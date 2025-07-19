"""
Configuration classes for PDF processing.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExtractionConfig:
    """
    Configuration for PDF extraction settings.
    """
    
    # Text extraction settings
    extract_text: bool = True
    extract_tables: bool = True
    extract_images: bool = False
    extract_metadata: bool = True
    
    # OCR settings
    ocr_enabled: bool = False
    ocr_language: str = "eng"
    ocr_confidence_threshold: float = 0.8
    
    # Processing settings
    language: str = "en"
    preserve_formatting: bool = True
    remove_headers_footers: bool = False
    
    # Table extraction settings
    table_detection_method: str = "auto"  # "auto", "stream", "lattice"
    table_vertical_strategy: str = "text"
    table_horizontal_strategy: str = "text"
    
    # Performance settings
    max_pages: Optional[int] = None
    parallel_processing: bool = False
    chunk_size: int = 1000
    
    def __post_init__(self):
        """Validate configuration settings."""
        if self.ocr_confidence_threshold < 0 or self.ocr_confidence_threshold > 1:
            raise ValueError("OCR confidence threshold must be between 0 and 1")
        
        if self.table_detection_method not in ["auto", "stream", "lattice"]:
            raise ValueError("Table detection method must be 'auto', 'stream', or 'lattice'")
        
        if self.table_vertical_strategy not in ["text", "lines", "explicit"]:
            raise ValueError("Table vertical strategy must be 'text', 'lines', or 'explicit'")
        
        if self.table_horizontal_strategy not in ["text", "lines", "explicit"]:
            raise ValueError("Table horizontal strategy must be 'text', 'lines', or 'explicit'") 