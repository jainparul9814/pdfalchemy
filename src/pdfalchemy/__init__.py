"""
PDFAlchemy - A Python library for advanced PDF manipulation and processing.
"""

__version__ = "0.1.0"
__author__ = "Parul Jain"
__email__ = "jainparul9814@gmail.com"

from .core import PDFProcessor, PNGConversionInput, PNGConversionOutput, ImageExtractionInput, ImageExtractionOutput

__all__ = [
    "PDFProcessor",
    "PNGConversionInput",
    "PNGConversionOutput",
    "ImageExtractionInput",
    "ImageExtractionOutput",
    "__version__",
] 