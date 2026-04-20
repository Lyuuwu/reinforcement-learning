import importlib

def load_experiment(name: str) -> dict:
    module = importlib.import_module(f'experiments.{name}')
    return module.build()