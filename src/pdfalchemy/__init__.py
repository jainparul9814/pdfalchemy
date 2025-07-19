"""
PDFAlchemy - A Python library for advanced PDF manipulation and processing.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import PDFProcessor, PNGConversionInput, PNGConversionOutput, ImageExtractionInput, ImageExtractionOutput
from .config import ExtractionConfig

__all__ = [
    "PDFProcessor",
    "PNGConversionInput",
    "PNGConversionOutput",
    "ImageExtractionInput",
    "ImageExtractionOutput",
    "ExtractionConfig",
    "__version__",
] 