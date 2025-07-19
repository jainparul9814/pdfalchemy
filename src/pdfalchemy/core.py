"""
Core PDF processing functionality.
"""

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pdf2image import convert_from_bytes
from pydantic import BaseModel, Field, field_validator
import cv2
import numpy as np
from PIL import Image

class PNGConversionInput(BaseModel):
    """Input model for PDF to PNG conversion."""
    
    pdf_bytes: bytes = Field(..., description="PDF data as byte array")
    dpi: int = Field(default=200, ge=72, le=1200, description="Resolution in DPI (dots per inch)")
    first_page: Optional[int] = Field(default=None, ge=1, description="First page to convert (1-indexed)")
    last_page: Optional[int] = Field(default=None, ge=1, description="Last page to convert (1-indexed)")
    
    @field_validator('pdf_bytes')
    @classmethod
    def validate_pdf_bytes(cls, v):
        """Validate that the byte array contains valid PDF data."""
        if not v.startswith(b'%PDF'):
            raise ValueError("Invalid PDF data: byte array does not start with PDF header")
        return v
    
    @field_validator('last_page')
    @classmethod
    def validate_page_range(cls, v, info):
        """Validate that last_page is greater than or equal to first_page."""
        if v is not None and info.data and info.data.get('first_page') is not None:
            if v < info.data['first_page']:
                raise ValueError("last_page must be greater than or equal to first_page")
        return v

class PNGConversionOutput(BaseModel):
    """Output model for PDF to PNG conversion."""
    
    png_images: List[bytes] = Field(..., description="List of PNG images as byte arrays")
    total_pages: int = Field(..., description="Total number of pages converted")
    dpi_used: int = Field(..., description="DPI used for conversion")
    page_range: Optional[str] = Field(default=None, description="Page range converted (e.g., '1-5')")
    
    @property
    def total_size_bytes(self) -> int:
        """Calculate total size of all PNG images in bytes."""
        return sum(len(img) for img in self.png_images)
    
    def get_page_info(self, page_index: int) -> Dict[str, Any]:
        """Get information about a specific page."""
        if page_index >= len(self.png_images):
            raise IndexError(f"Page index {page_index} out of range")
        
        return {
            "page_number": page_index + 1,
            "size_bytes": len(self.png_images[page_index]),
            "png_data": self.png_images[page_index]
        }

class ImageExtractionInput(BaseModel):
    """Input model for image extraction from PNG using flood fill."""
    
    png_bytes: bytes = Field(..., description="PNG image data as byte array")
    min_width: int = Field(default=50, ge=1, description="Minimum width in pixels for extracted images")
    min_height: int = Field(default=50, ge=1, description="Minimum height in pixels for extracted images")
    max_width: Optional[int] = Field(default=None, ge=1, description="Maximum width in pixels for extracted images")
    max_height: Optional[int] = Field(default=None, ge=1, description="Maximum height in pixels for extracted images")
    min_aspect_ratio: Optional[float] = Field(default=None, ge=0.1, description="Minimum aspect ratio (width/height) for extracted images")
    max_aspect_ratio: Optional[float] = Field(default=None, ge=0.1, description="Maximum aspect ratio (width/height) for extracted images")
    flood_fill_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Threshold for flood fill algorithm")
    noise_reduction: bool = Field(default=True, description="Enable noise reduction")
    separate_connected_regions: bool = Field(default=True, description="Attempt to separate connected regions using morphological operations")
    sort_order: str = Field(default="top-bottom", description="Sort order for extracted images: 'top-bottom' (default), 'left-right', 'reading-order'")
    
    @field_validator('png_bytes')
    @classmethod
    def validate_png_bytes(cls, v):
        """Validate that the byte array contains valid PNG data."""
        if not v.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError("Invalid PNG data: byte array does not start with PNG header")
        return v
    
    @field_validator('max_width')
    @classmethod
    def validate_max_width(cls, v, info):
        """Validate that max_width is greater than min_width."""
        if v is not None and info.data and info.data.get('min_width') is not None:
            if v < info.data['min_width']:
                raise ValueError("max_width must be greater than or equal to min_width")
        return v
    
    @field_validator('max_height')
    @classmethod
    def validate_max_height(cls, v, info):
        """Validate that max_height is greater than min_height."""
        if v is not None and info.data and info.data.get('min_height') is not None:
            if v < info.data['min_height']:
                raise ValueError("max_height must be greater than or equal to min_height")
        return v
    
    @field_validator('max_aspect_ratio')
    @classmethod
    def validate_aspect_ratio(cls, v, info):
        """Validate that max_aspect_ratio is greater than or equal to min_aspect_ratio."""
        if v is not None and info.data and info.data.get('min_aspect_ratio') is not None:
            if v < info.data['min_aspect_ratio']:
                raise ValueError("max_aspect_ratio must be greater than or equal to min_aspect_ratio")
        return v
    
    @field_validator('sort_order')
    @classmethod
    def validate_sort_order(cls, v):
        """Validate sort order option."""
        valid_orders = ['top-bottom', 'left-right', 'reading-order']
        if v not in valid_orders:
            raise ValueError(f"sort_order must be one of: {', '.join(valid_orders)}")
        return v

class ImageExtractionOutput(BaseModel):
    """Output model for image extraction from PNG."""
    
    extracted_images: List[str] = Field(..., description="List of base64 encoded extracted images")
    total_images: int = Field(..., description="Total number of extracted images")
    filtered_count: int = Field(..., description="Number of images filtered out due to size constraints")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    @property
    def total_size_bytes(self) -> int:
        """Calculate total size of all extracted images in bytes."""
        total = 0
        for img_base64 in self.extracted_images:
            # Decode base64 to get original size
            img_bytes = base64.b64decode(img_base64)
            total += len(img_bytes)
        return total

class PDFProcessor:
    """
    Main class for processing individual PDF documents.
    """
    
    def __init__(self):
        """
        Initialize PDF processor.
        """
        pass
    
    def _validate_pdf_bytes(self, pdf_bytes: bytes):
        """Validate that the byte array contains valid PDF data."""
        if not pdf_bytes.startswith(b'%PDF'):
            raise ValueError("Invalid PDF data: byte array does not start with PDF header")
    
    def to_png(self, input_data: PNGConversionInput) -> PNGConversionOutput:
        """
        Convert PDF pages to PNG images as byte arrays.
        
        Args:
            input_data: PNGConversionInput model containing PDF data and conversion parameters
            
        Returns:
            PNGConversionOutput model containing PNG images and metadata
            
        Raises:
            RuntimeError: If conversion fails
        """
        try:
            # Convert PDF to images
            images = convert_from_bytes(
                input_data.pdf_bytes,
                dpi=input_data.dpi,
                first_page=input_data.first_page,
                last_page=input_data.last_page
            )
            
            # Convert images to byte arrays
            png_bytes_list = []
            
            for image in images:
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="PNG")
                png_bytes_list.append(img_buffer.getvalue())
                img_buffer.close()
            
            # Create page range string
            page_range = None
            if input_data.first_page is not None or input_data.last_page is not None:
                start = input_data.first_page or 1
                end = input_data.last_page or len(images)
                page_range = f"{start}-{end}"
            
            return PNGConversionOutput(
                png_images=png_bytes_list,
                total_pages=len(png_bytes_list),
                dpi_used=input_data.dpi,
                page_range=page_range
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to PNG: {e}")
    
    def to_png_bytes(self, input_data: PNGConversionInput) -> List[bytes]:
        """
        Convert PDF pages to PNG images as byte arrays.
        
        Args:
            input_data: PNGConversionInput model containing PDF data and conversion parameters
            
        Returns:
            List of byte arrays containing PNG image data
            
        Raises:
            RuntimeError: If conversion fails
        """
        try:
            # Convert PDF to images
            images = convert_from_bytes(
                input_data.pdf_bytes,
                dpi=input_data.dpi,
                first_page=input_data.first_page,
                last_page=input_data.last_page
            )
            
            # Convert images to byte arrays
            png_bytes_list = []
            
            for image in images:
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="PNG")
                png_bytes_list.append(img_buffer.getvalue())
                img_buffer.close()
            
            return png_bytes_list
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to PNG bytes: {e}")
    
    def to_png_base64(self, input_data: PNGConversionInput) -> List[str]:
        """
        Convert PDF pages to PNG images as base64 encoded strings.
        
        Args:
            input_data: PNGConversionInput model containing PDF data and conversion parameters
            
        Returns:
            List of base64 encoded strings containing PNG image data
            
        Raises:
            RuntimeError: If conversion fails
        """
        try:
            # Get PNG bytes first
            png_bytes_list = self.to_png_bytes(input_data)
            
            # Convert to base64
            base64_list = []
            for png_bytes in png_bytes_list:
                base64_str = base64.b64encode(png_bytes).decode('utf-8')
                base64_list.append(base64_str)
            
            return base64_list
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to PNG base64: {e}")

    def extract_images_from_png(self, input_data: ImageExtractionInput) -> ImageExtractionOutput:
        """
        Extract images from PNG using a flood fill algorithm with size filtering.

        Args:
            input_data: ImageExtractionInput model containing PNG data and extraction parameters

        Returns:
            ImageExtractionOutput model containing base64 encoded extracted images

        Raises:
            RuntimeError: If extraction fails
        """
        import time
        start_time = time.time()

        try:
            # Load the PNG into an OpenCV image for processing
            img_array = np.frombuffer(input_data.png_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("Failed to decode PNG bytes to image.")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Optional noise reduction
            if input_data.noise_reduction:
                gray = cv2.medianBlur(gray, 3)

            # Threshold the image, assuming images are non-white
            _, thresh = cv2.threshold(
                gray,
                int(255 * (1 - input_data.flood_fill_threshold)),
                255,
                cv2.THRESH_BINARY_INV
            )

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Store all valid regions with their positions for ordering
            valid_regions = []
            filtered_count = 0

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                # Filtering by size
                if w < input_data.min_width or h < input_data.min_height:
                    filtered_count += 1
                    continue
                if input_data.max_width and w > input_data.max_width:
                    filtered_count += 1
                    continue
                if input_data.max_height and h > input_data.max_height:
                    filtered_count += 1
                    continue

                # Filtering by aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                if input_data.min_aspect_ratio and aspect_ratio < input_data.min_aspect_ratio:
                    filtered_count += 1
                    continue
                if input_data.max_aspect_ratio and aspect_ratio > input_data.max_aspect_ratio:
                    filtered_count += 1
                    continue

                # Check if this contour contains multiple disconnected regions
                # by analyzing the contour area vs bounding box area
                contour_area = cv2.contourArea(cnt)
                bounding_box_area = w * h
                area_ratio = contour_area / bounding_box_area if bounding_box_area > 0 else 0
                
                # If area ratio is low, it might be multiple disconnected regions
                # Try to separate them using morphological operations
                if input_data.separate_connected_regions and area_ratio < 0.6:  # More conservative threshold
                    # Create a mask for this contour
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [cnt], 0, 255, -1)
                    
                    # Try multiple approaches to separate regions
                    separated_regions = []
                    
                    # Approach 1: Gentle morphological operations
                    kernel_small = np.ones((2, 2), np.uint8)
                    mask_eroded = cv2.erode(mask, kernel_small, iterations=1)
                    mask_dilated = cv2.dilate(mask_eroded, kernel_small, iterations=1)
                    
                    sub_contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # If we got multiple regions, use them
                    if len(sub_contours) > 1:
                        separated_regions = sub_contours
                    else:
                        # Approach 2: Try distance transform to separate close regions
                        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                        _, sure_fg = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)
                        sure_fg = np.uint8(sure_fg)
                        
                        # Find markers for watershed
                        _, markers = cv2.connectedComponents(sure_fg)
                        markers = markers + 1
                        
                        # Apply watershed
                        cv2.watershed(img, markers)
                        
                        # Extract regions from watershed result
                        for marker_id in range(2, markers.max() + 1):
                            marker_mask = np.zeros(mask.shape, dtype=np.uint8)
                            marker_mask[markers == marker_id] = 255
                            
                            # Find contours in this marker
                            marker_contours, _ = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if marker_contours:
                                separated_regions.extend(marker_contours)
                    
                    # Process separated regions
                    if separated_regions:
                        for sub_cnt in separated_regions:
                            sub_x, sub_y, sub_w, sub_h = cv2.boundingRect(sub_cnt)
                            
                            # Apply the same filtering to sub-contours
                            if sub_w < input_data.min_width or sub_h < input_data.min_height:
                                filtered_count += 1
                                continue
                            if input_data.max_width and sub_w > input_data.max_width:
                                filtered_count += 1
                                continue
                            if input_data.max_height and sub_h > input_data.max_height:
                                filtered_count += 1
                                continue
                            
                            sub_aspect_ratio = sub_w / sub_h if sub_h > 0 else 0
                            if input_data.min_aspect_ratio and sub_aspect_ratio < input_data.min_aspect_ratio:
                                filtered_count += 1
                                continue
                            if input_data.max_aspect_ratio and sub_aspect_ratio > input_data.max_aspect_ratio:
                                filtered_count += 1
                                continue
                            
                            # Extract sub-region
                            sub_roi = img[sub_y:sub_y+sub_h, sub_x:sub_x+sub_w]
                            
                            # Store sub-region with position for ordering
                            is_success, buffer = cv2.imencode(".png", sub_roi)
                            if not is_success:
                                continue
                            roi_bytes = buffer.tobytes()
                            img_base64 = base64.b64encode(roi_bytes).decode("utf-8")
                            valid_regions.append({
                                'img_base64': img_base64,
                                'x': sub_x,
                                'y': sub_y,
                                'w': sub_w,
                                'h': sub_h
                            })
                    else:
                        # If separation failed, fall back to original contour
                        roi = img[y:y+h, x:x+w]
                        is_success, buffer = cv2.imencode(".png", roi)
                        if not is_success:
                            continue
                        roi_bytes = buffer.tobytes()
                        img_base64 = base64.b64encode(roi_bytes).decode("utf-8")
                        valid_regions.append({
                            'img_base64': img_base64,
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h
                        })
                else:
                    # Extract region of interest (ROI) for single connected region
                    roi = img[y:y+h, x:x+w]

                    # Encode ROI back as PNG and base64 for output
                    is_success, buffer = cv2.imencode(".png", roi)
                    if not is_success:
                        continue
                    roi_bytes = buffer.tobytes()
                    img_base64 = base64.b64encode(roi_bytes).decode("utf-8")
                    valid_regions.append({
                        'img_base64': img_base64,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h
                    })

            # Sort regions by position based on sort_order
            if input_data.sort_order == "top-bottom":
                # Sort by y-coordinate first (top to bottom), then by x-coordinate (left to right)
                valid_regions.sort(key=lambda region: (region['y'], region['x']))
            elif input_data.sort_order == "left-right":
                # Sort by x-coordinate first (left to right), then by y-coordinate (top to bottom)
                valid_regions.sort(key=lambda region: (region['x'], region['y']))
            elif input_data.sort_order == "reading-order":
                # Sort by reading order: top-to-bottom, left-to-right, but with row detection
                # Group regions by approximate row (within 20% of image height)
                img_height = img.shape[0]
                row_threshold = img_height * 0.2
                
                # Group regions by row
                rows = {}
                for region in valid_regions:
                    row_key = int(region['y'] / row_threshold)
                    if row_key not in rows:
                        rows[row_key] = []
                    rows[row_key].append(region)
                
                # Sort each row by x-coordinate, then sort rows by y-coordinate
                sorted_regions = []
                for row_key in sorted(rows.keys()):
                    rows[row_key].sort(key=lambda region: region['x'])
                    sorted_regions.extend(rows[row_key])
                
                valid_regions = sorted_regions
            
            # Extract the sorted images
            extracted_images = [region['img_base64'] for region in valid_regions]
            
            processing_time = (time.time() - start_time) * 1000
            return ImageExtractionOutput(
                extracted_images=extracted_images,
                total_images=len(extracted_images),
                filtered_count=filtered_count,
                processing_time_ms=processing_time
            )
        except Exception as e:
            raise RuntimeError(f"Failed to extract images from PNG: {e}")

