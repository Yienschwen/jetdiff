try:
    from .cjet import CJet as Jet
except ImportError:
    from .pyjet import PyJet as Jet
