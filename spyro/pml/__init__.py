import importlib
import pkgutil

__all__ = []
for module_info in pkgutil.walk_packages(__path__):
    module_name = module_info.name
    __all__.append(module_name)
    module = importlib.import_module(f"{__name__}.{module_name}")
    globals()[module_name] = module
