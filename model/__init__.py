import os
import importlib

current_dir = os.path.dirname(__file__)
modules = [f[:-3] for f in os.listdir(current_dir) if f.endswith('.py') and f != '__init__.py']

for module in modules:
    importlib.import_module(f'.{module}', package='model')

for module in modules:
    mod = importlib.import_module(f'.{module}', package='model')
    for attr in dir(mod):
        if not attr.startswith('_'):
            globals()[attr] = getattr(mod, attr)