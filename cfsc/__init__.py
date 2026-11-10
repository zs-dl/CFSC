"""Cross-Fitted Star Control for fixed-size locally private subset reports.

Public objects are imported lazily so the CLI can configure its thread limit
before importing NumPy. Importing this package changes no process settings.
"""
from importlib import import_module

__version__ = '1.0.0'
_EXPORTS = {'decode_reports': '.api', 'channel_parameters': '.channel',
            'sample_reports': '.channel', 'synthetic_categories': '.channel',
            'load_categories': '.channel', 'DecoderConfig': '.decoder', 'MLEConfig': '.ss_mle'}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
