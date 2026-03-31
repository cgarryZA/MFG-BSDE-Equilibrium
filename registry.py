# registry.py

EQUATION_REGISTRY = {}

def register_equation(name):
    """Decorator to automatically register an equation class."""
    def decorator(cls):
        EQUATION_REGISTRY[name] = cls
        return cls
    return decorator