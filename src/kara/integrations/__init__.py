"""
Framework integrations for kara-toolkit.
"""

# Import integrations only if the required packages are available
try:
    from .langchain import KARATextSplitter

    __all__ = ["KARATextSplitter"]
except ImportError:
    __all__ = []
