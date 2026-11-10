"""Report-only server API: original values are not accepted by this module."""
import numpy as np
from .channel import channel_parameters
from .decoder import DecoderConfig, decoder
from .ss_mle import MLEConfig, solve_ss_mle


def decode_reports(reports, d, epsilon, k=None, *, config=None, mle_config=None, include_mle=True):
    public = channel_parameters(d, epsilon, k)
    reports = np.asarray(reports)
    n = len(reports)
    result = decoder(reports, n, d, epsilon, public['k'], public['p'], public['q'], config=config)
    if include_mle:
        if mle_config is None:
            mle_config = MLEConfig(gap_tolerance=1e-11)
        mle = solve_ss_mle(reports, n, d, epsilon, public['k'], config=mle_config)
        result['estimates']['ss_mle'] = mle['estimate']
        result['diagnostics']['ss_mle'] = mle['diagnostics']
    return result
