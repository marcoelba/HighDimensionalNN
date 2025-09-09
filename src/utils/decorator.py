# decorator function to automatically compose docstrings


def compose_docstring(*component_classes):
    """Decorator to compose docstring from multiple component classes."""
    
    def decorator(cls):
        base_doc = cls.__doc__ or ""
        component_docs = []
        
        for comp_class in component_classes:
            if comp_class.__doc__:
                comp_name = comp_class.__name__
                comp_doc = comp_class.__doc__.strip()
                component_docs.append(f"{comp_name}:\n{comp_doc}")
        
        if component_docs:
            components_section = "\n\nComponents:\n" + "\n\n".join(component_docs)
            cls.__doc__ = base_doc + components_section
        
        return cls
    return decorator
