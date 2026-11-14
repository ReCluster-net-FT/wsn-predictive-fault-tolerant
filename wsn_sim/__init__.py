from importlib import import_module

from .network import Network, RadioParams

__all__ = [
    "Network", "RadioParams", "moga", "lsa", "simulate", "wsn_dataset",
    "graphsage_embed", "moga_embedding", "ns3_export",
]


def __getattr__(name):
    if name in __all__[2:]:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
