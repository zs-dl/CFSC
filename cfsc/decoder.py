"""CFSC report-only server. No private values, truth, scenario or evaluation API."""
from dataclasses import dataclass,asdict
from functools import lru_cache
import hashlib
import math

import numpy as np


@dataclass(frozen=True)
class DecoderConfig:
    public_split_seed: int=20261109
    max_heads: int=8
    eigen_rtol: float=1e-10


ESTIMATORS=("ss_raw","ss_simplex","cfsc_affine","cfsc_simplex")


def simplex_projection(values):
    values=np.asarray(values,dtype=np.float64)
    ordered=np.sort(values)[::-1]
    cssv=np.cumsum(ordered)-1
    ranks=np.arange(1,len(values)+1)
    active=np.flatnonzero(ordered-cssv/ranks>0)
    threshold=cssv[active[-1]]/(active[-1]+1)
    return np.maximum(values-threshold,0)


def select_heads(f,max_heads=8):
    """Strict boundary rule, excluding ALL ties at the (max_heads+1)th value."""
    f=np.asarray(f,dtype=np.float64)
    if len(f)<=max_heads:
        return np.arange(len(f),dtype=np.int64)
    boundary=np.sort(f)[-(max_heads+1)]
    return np.flatnonzero(f>boundary).astype(np.int64)


def inclusion_coefficients(d,k,epsilon,maximum_order=3):
    theta=math.exp(-epsilon)
    one_minus=-math.expm1(-epsilon)
    denominator=k+(d-k)*theta
    A=np.zeros(maximum_order+1)
    B=np.zeros(maximum_order+1)
    D=np.zeros(maximum_order+1)
    A[0]=B[0]=1
    for r in range(1,maximum_order+1):
        if r>k or r>=d:
            continue
        u=math.prod((k-a)/(d-a) for a in range(r))
        A[r]=d*u/denominator
        B[r]=A[r]*(theta+one_minus*(k-r)/(d-r))
        D[r]=A[r]*one_minus*(d-k)/(d-r)
    return A,B,D


def star_moments(f,j,heads,d,k,epsilon):
    """Analytic third-order moments for C_(j,h), h != j."""
    heads=np.asarray(heads,dtype=np.int64)
    if np.any(heads==j):
        raise ValueError("A star cannot contain a self edge")
    if len(heads)==0:
        return np.zeros(0),np.zeros((0,0))
    if d<4 or k in (1,d-1):
        return np.zeros(len(heads)),np.zeros((len(heads),len(heads)))
    rho=(k-1)/(d-2)
    tau=k*(k-1)/((d-1)*(d-2))
    _,B,D=inclusion_coefficients(d,k,epsilon)
    @lru_cache(None)
    def moment_tuple(indices):
        if not indices:
            return 1.0
        r=len(indices)
        if r>=len(B):
            raise ArithmeticError("Star Gram unexpectedly requires more than third-order moments")
        return B[r]+D[r]*math.fsum(float(f[v]) for v in indices)
    def moment(*indices):
        return moment_tuple(tuple(sorted(set(indices))))
    m=len(heads)
    gram=np.empty((m,m),dtype=np.float64)
    for a,h in enumerate(heads):
        edge=(j,int(h))
        for b in range(a,m):
            other=(j,int(heads[b]))
            value=(moment(*edge,*other)
                   -rho*(sum(moment(*edge,v) for v in other)+sum(moment(*other,u) for u in edge))
                   +rho*rho*sum(moment(u,v) for u in edge for v in other)
                   +tau*(moment(*edge)+moment(*other))
                   -rho*tau*(sum(moment(u) for u in edge)+sum(moment(v) for v in other))
                   +tau*tau)
            gram[a,b]=gram[b,a]=value
    eta=(k-1)*(d-k-1)/(d-2)**2
    correlation=eta*(f[heads]-(1-f[j])/(d-1))
    return correlation,gram


def train_star_model(training_counts,training_n,d,k,epsilon,p,q,config):
    raw=(training_counts/training_n-q)/(p-q)
    f=simplex_projection(raw)
    heads=select_heads(f,config.max_heads)
    width=len(heads)
    H=np.zeros((d,width))
    G=np.zeros((d,width,width))
    eigenvalues=np.zeros((d,width))
    eigenvectors=np.zeros((d,width,width))
    coefficients=np.zeros((d,width))
    masks=np.ones((d,width),dtype=bool)
    ranks=np.zeros(d,dtype=np.int64)
    conditions=np.zeros(d)
    residuals=np.zeros(d)
    bounds=np.zeros(d)
    minimum_eigenvalue=0.0
    degenerate=d<4 or k in (1,d-1)
    for j in range(d):
        mask=heads!=j
        masks[j]=mask
        if degenerate or not mask.any():
            continue
        positions=np.flatnonzero(mask)
        h,gram=star_moments(f,j,heads[mask],d,k,epsilon)
        gram=(gram+gram.T)/2
        values,vectors=np.linalg.eigh(gram)
        maximum=float(values[-1])
        minimum_eigenvalue=min(minimum_eigenvalue,float(values[0]))
        # This is a roundoff check, not a regularization/utility parameter.
        psd_tolerance=64*np.finfo(np.float64).eps*max(1.0,float(np.linalg.norm(gram,ord=2)))
        if values[0] < -psd_tolerance:
            raise ArithmeticError(f"Analytic star Gram is not PSD for coordinate {j}: {values[0]}")
        positive=values>config.eigen_rtol*maximum if maximum>0 else np.zeros(len(values),dtype=bool)
        w=(vectors[:,positive]@((vectors[:,positive].T@h)/values[positive])
           if positive.any() else np.zeros(len(h)))
        H[j,positions]=h
        G[j][np.ix_(positions,positions)]=gram
        eigenvalues[j,:len(values)]=values
        eigenvectors[j][np.ix_(positions,np.arange(len(values)))]=vectors
        coefficients[j,positions]=w
        ranks[j]=int(positive.sum())
        if positive.any():
            conditions[j]=float(values[positive][-1]/values[positive][0])
        residuals[j]=float(np.linalg.norm(gram@w-h))
        m=len(h)
        bounds[j]=math.exp(epsilon)*(1 if m==d-1 else (d-1)/(d-1-m))
    arrays={"training_counts":training_counts,"training_raw":raw,"training_frequency":f,
            "heads":heads,"control_mask":masks,"H":H,"Gram":G,"eigenvalues":eigenvalues,
            "eigenvectors":eigenvectors,"coefficients":coefficients,"positive_rank":ranks,
            "positive_condition_number":conditions,"condition_bound":bounds,
            "normal_equation_residual":residuals}
    diagnostics={"training_n":training_n,"heads":heads.tolist(),"head_count":width,
                 "training_frequency_mass":float(f.sum()),"training_frequency_min":float(f.min()),
                 "minimum_gram_eigenvalue":minimum_eigenvalue,
                 "maximum_normal_equation_residual":float(residuals.max()),
                 "maximum_positive_condition_number":float(conditions.max()),
                 "control_identically_zero":degenerate,
                 "eigenvector_layout":"rows=head positions; first m columns are local eigenvectors; m=count(mask)",
                 "empty_head":width==0}
    return arrays,diagnostics


def apply_star_model(test_bits,model,d,k):
    """Accumulate co-inclusions in float64, never uint8 matrix arithmetic."""
    n_test=len(test_bits)
    heads=model["heads"]
    width=len(heads)
    counts=test_bits.sum(axis=0,dtype=np.int64)
    if d<4 or k in (1,d-1) or width==0:
        return np.zeros(d),{"test_counts":counts,"control_means":np.zeros((d,width)),
                            "correction":np.zeros(d)}, {"test_n":n_test,"co_inclusion_accumulator":"unused_zero_control_branch"}
    floating=test_bits.astype(np.float64)
    co_counts=floating.T@floating[:,heads]
    rho=(k-1)/(d-2)
    tau=k*(k-1)/((d-1)*(d-2))
    control_means=co_counts/n_test-rho*(counts[:,None]+counts[heads][None,:])/n_test+tau
    control_means=np.where(model["control_mask"],control_means,0.0)
    correction=np.sum(model["coefficients"]*control_means,axis=1)
    return correction,{"test_counts":counts,"co_inclusion_counts":co_counts,
                       "control_means":control_means,"correction":correction}, {
                       "test_n":n_test,"co_inclusion_accumulator":"float64_dot_of_binary_values"}


def decoder(reports,n,d,epsilon,k,p,q,*,config=None):
    config=DecoderConfig() if config is None else config
    if n<2 or d<2 or not 1<=k<d or not math.isfinite(epsilon) or epsilon<=0:
        raise ValueError("Require n>=2, d>=2, 1<=k<d, finite positive epsilon")
    if config.max_heads<1 or not 0<config.eigen_rtol<1:
        raise ValueError("Invalid public head/eigen settings")
    if not 0<=q<p<=1:
        raise ValueError("Invalid SS support probabilities")
    reports=np.asarray(reports)
    if reports.dtype!=np.uint8 or reports.shape!=(n,(d+7)//8):
        raise ValueError("Expected the authenticated uint8 little-endian SS bitmap")
    bits=np.unpackbits(reports,axis=1,count=d,bitorder="little")
    if not np.all(bits.sum(axis=1,dtype=np.int64)==k):
        raise ValueError("A decoded report does not contain exactly k public-domain bits")
    A,B,D=inclusion_coefficients(d,k,epsilon)
    if not np.allclose([p,q],[A[1],B[1]],rtol=1e-12,atol=1e-14):
        raise ValueError("Reported SS probabilities do not match the declared channel")
    counts=bits.sum(axis=0,dtype=np.int64)
    raw=(counts/n-q)/(p-q)
    permutation=np.random.Generator(np.random.PCG64(config.public_split_seed)).permutation(n)
    a_indices=permutation[:n//2]
    b_indices=permutation[n//2:]
    a_bits,b_bits=bits[a_indices],bits[b_indices]
    a_counts=a_bits.sum(axis=0,dtype=np.int64)
    b_counts=b_bits.sum(axis=0,dtype=np.int64)
    a_model,a_diag=train_star_model(a_counts,len(a_indices),d,k,epsilon,p,q,config)
    b_model,b_diag=train_star_model(b_counts,len(b_indices),d,k,epsilon,p,q,config)
    # The coefficient model and the application reports always come from
    # disjoint folds. Actual fold sizes, not one half, weight the two outputs.
    a_correction,a_apply,a_apply_diag=apply_star_model(a_bits,b_model,d,k)
    b_correction,b_apply,b_apply_diag=apply_star_model(b_bits,a_model,d,k)
    combined=(len(a_indices)*a_correction+len(b_indices)*b_correction)/n
    centered=combined-combined.mean()
    affine=raw-centered
    estimates={"ss_raw":raw,"ss_simplex":simplex_projection(raw),
               "cfsc_affine":affine,"cfsc_simplex":simplex_projection(affine)}
    arrays={"report_counts":counts,"fold_a_indices":a_indices,"fold_b_indices":b_indices,
            "combined_correction":combined,"centered_correction":centered,
            "inclusion_A":A,"inclusion_B":B,"inclusion_D":D}
    for prefix,model in [("train_a",a_model),("train_b",b_model),("apply_a_from_b",a_apply),("apply_b_from_a",b_apply)]:
        arrays.update({prefix+"_"+name:value for name,value in model.items()})
    def index_hash(ix):
        return hashlib.sha256(np.asarray(ix,dtype="<i8").tobytes()).hexdigest()
    diagnostics={"config":asdict(config),"n":n,"d":d,"epsilon":epsilon,"k":k,"p":p,"q":q,
                 "fold_sizes":{"a":len(a_indices),"b":len(b_indices)},
                 "fold_weights":{"a":len(a_indices)/n,"b":len(b_indices)/n},
                 "fold_hashes":{"a":index_hash(a_indices),"b":index_hash(b_indices),"permutation":index_hash(permutation)},
                 "bit_order":"little","padding_bits":"ignored beyond d",
                 "row_cardinality_verified":True,"counts_dtype":str(counts.dtype),
                 "train_a":a_diag,"train_b":b_diag,
                 "apply_a_from_b":a_apply_diag,"apply_b_from_a":b_apply_diag,
                 "combined_correction_mass":float(combined.sum()),
                 "centered_correction_mass":float(centered.sum()),
                 "raw_mass":float(raw.sum()),"affine_mass":float(affine.sum()),
                 "estimator_status":{name:"success" for name in ESTIMATORS}}
    for name,value in estimates.items():
        if not np.isfinite(value).all() or abs(value.sum()-1)>1e-9:
            raise ArithmeticError("Invalid finite/mass output: "+name)
        if name.endswith("simplex") and value.min() < -1e-13:
            raise ArithmeticError("Invalid simplex output: "+name)
    return {"estimates":estimates,"arrays":arrays,"diagnostics":diagnostics}
