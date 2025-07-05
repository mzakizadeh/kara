"""
Framework integrations for kara-python.
"""

# Import integrations only if the required packages are available
try:
    from .langchain import KARATextSplitter, LangChainKARAUpdater

    __all__ = ["LangChainKARAUpdater", "KARATextSplitter"]
except ImportError:
    __all__ = []
