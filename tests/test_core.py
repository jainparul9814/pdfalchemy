"""
Tests for core PDF processing functionality.
"""

import base64
import io
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from pdfalchemy.core import (
    PNGConversionInput, PNGConversionOutput,
    ImageExtractionInput, ImageExtractionOutput,
    PDFProcessor
)


class TestPNGConversionInput:
    """Test cases for PNGConversionInput model."""
    
    def test_valid_input(self):
        """Test valid PNGConversionInput creation."""
        pdf_bytes = b"%PDF-1.4\n%Test PDF content"
        input_data = PNGConversionInput(
            pdf_bytes=pdf_bytes,
            dpi=300,
            first_page=1,
            last_page=5
        )
        
        assert input_data.pdf_bytes == pdf_bytes
        assert input_data.dpi == 300
        assert input_data.first_page == 1
        assert input_data.last_page == 5
    
    def test_default_values(self):
        """Test PNGConversionInput with default values."""
        pdf_bytes = b"%PDF-1.4\n%Test PDF content"
        input_data = PNGConversionInput(pdf_bytes=pdf_bytes)
        
        assert input_data.dpi == 200
        assert input_data.first_page is None
        assert input_data.last_page is None
    
    def test_invalid_pdf_bytes(self):
        """Test validation of invalid PDF bytes."""
        with pytest.raises(ValueError, match="Invalid PDF data"):
            PNGConversionInput(pdf_bytes=b"Not a PDF")
    
    def test_dpi_constraints(self):
        """Test DPI constraints."""
        pdf_bytes = b"%PDF-1.4\n%Test PDF content"
        
        # Test minimum DPI
        with pytest.raises(ValueError):
            PNGConversionInput(pdf_bytes=pdf_bytes, dpi=50)
        
        # Test maximum DPI
        with pytest.raises(ValueError):
            PNGConversionInput(pdf_bytes=pdf_bytes, dpi=1500)
    
    def test_page_range_validation(self):
        """Test page range validation."""
        pdf_bytes = b"%PDF-1.4\n%Test PDF content"
        
        # Test invalid page range
        with pytest.raises(ValueError, match="last_page must be greater than or equal to first_page"):
            PNGConversionInput(
                pdf_bytes=pdf_bytes,
                first_page=5,
                last_page=3
            )
    
    def test_page_number_constraints(self):
        """Test page number constraints."""
        pdf_bytes = b"%PDF-1.4\n%Test PDF content"
        
        # Test invalid first_page
        with pytest.raises(ValueError):
            PNGConversionInput(pdf_bytes=pdf_bytes, first_page=0)
        
        # Test invalid last_page
        with pytest.raises(ValueError):
            PNGConversionInput(pdf_bytes=pdf_bytes, last_page=0)


class TestPNGConversionOutput:
    """Test cases for PNGConversionOutput model."""
    
    def test_valid_output(self):
        """Test valid PNGConversionOutput creation."""
        # Create mock PNG images
        png1 = self._create_mock_png(100, 100)
        png2 = self._create_mock_png(200, 150)
        
        output = PNGConversionOutput(
            png_images=[png1, png2],
            total_pages=2,
            dpi_used=300,
            page_range="1-2"
        )
        
        assert len(output.png_images) == 2
        assert output.total_pages == 2
        assert output.dpi_used == 300
        assert output.page_range == "1-2"
    
    def test_total_size_bytes(self):
        """Test total_size_bytes property."""
        png1 = self._create_mock_png(100, 100)
        png2 = self._create_mock_png(200, 150)
        
        output = PNGConversionOutput(
            png_images=[png1, png2],
            total_pages=2,
            dpi_used=300
        )
        
        expected_size = len(png1) + len(png2)
        assert output.total_size_bytes == expected_size
    
    def test_get_page_info(self):
        """Test get_page_info method."""
        png1 = self._create_mock_png(100, 100)
        png2 = self._create_mock_png(200, 150)
        
        output = PNGConversionOutput(
            png_images=[png1, png2],
            total_pages=2,
            dpi_used=300
        )
        
        # Test valid page info
        page_info = output.get_page_info(0)
        assert page_info["page_number"] == 1
        assert page_info["size_bytes"] == len(png1)
        assert page_info["png_data"] == png1
        
        # Test invalid page index
        with pytest.raises(IndexError):
            output.get_page_info(5)
    
    def _create_mock_png(self, width, height):
        """Helper method to create mock PNG data."""
        img = Image.new('RGB', (width, height), color='white')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()


class TestImageExtractionInput:
    """Test cases for ImageExtractionInput model."""
    
    def test_valid_input(self):
        """Test valid ImageExtractionInput creation."""
        png_bytes = self._create_mock_png(800, 600)
        input_data = ImageExtractionInput(
            png_bytes=png_bytes,
            min_width=50,
            min_height=50,
            max_width=500,
            max_height=400,
            min_aspect_ratio=0.5,
            max_aspect_ratio=2.0,
            flood_fill_threshold=0.2,
            noise_reduction=True,
            separate_connected_regions=True
        )
        
        assert input_data.png_bytes == png_bytes
        assert input_data.min_width == 50
        assert input_data.min_height == 50
        assert input_data.max_width == 500
        assert input_data.max_height == 400
        assert input_data.min_aspect_ratio == 0.5
        assert input_data.max_aspect_ratio == 2.0
        assert input_data.flood_fill_threshold == 0.2
        assert input_data.noise_reduction is True
        assert input_data.separate_connected_regions is True
    
    def test_default_values(self):
        """Test ImageExtractionInput with default values."""
        png_bytes = self._create_mock_png(800, 600)
        input_data = ImageExtractionInput(png_bytes=png_bytes)
        
        assert input_data.min_width == 50
        assert input_data.min_height == 50
        assert input_data.max_width is None
        assert input_data.max_height is None
        assert input_data.min_aspect_ratio is None
        assert input_data.max_aspect_ratio is None
        assert input_data.flood_fill_threshold == 0.1
        assert input_data.noise_reduction is True
        assert input_data.separate_connected_regions is True
    
    def test_invalid_png_bytes(self):
        """Test validation of invalid PNG bytes."""
        with pytest.raises(ValueError, match="Invalid PNG data"):
            ImageExtractionInput(png_bytes=b"Not a PNG")
    
    def test_size_constraints(self):
        """Test size constraint validation."""
        png_bytes = self._create_mock_png(800, 600)
        
        # Test max_width < min_width
        with pytest.raises(ValueError, match="max_width must be greater than or equal to min_width"):
            ImageExtractionInput(
                png_bytes=png_bytes,
                min_width=100,
                max_width=50
            )
        
        # Test max_height < min_height
        with pytest.raises(ValueError, match="max_height must be greater than or equal to min_height"):
            ImageExtractionInput(
                png_bytes=png_bytes,
                min_height=100,
                max_height=50
            )
    
    def test_aspect_ratio_constraints(self):
        """Test aspect ratio constraint validation."""
        png_bytes = self._create_mock_png(800, 600)
        
        # Test max_aspect_ratio < min_aspect_ratio
        with pytest.raises(ValueError, match="max_aspect_ratio must be greater than or equal to min_aspect_ratio"):
            ImageExtractionInput(
                png_bytes=png_bytes,
                min_aspect_ratio=2.0,
                max_aspect_ratio=1.0
            )
    
    def test_threshold_constraints(self):
        """Test flood fill threshold constraints."""
        png_bytes = self._create_mock_png(800, 600)
        
        # Test threshold too low
        with pytest.raises(ValueError):
            ImageExtractionInput(png_bytes=png_bytes, flood_fill_threshold=-0.1)
        
        # Test threshold too high
        with pytest.raises(ValueError):
            ImageExtractionInput(png_bytes=png_bytes, flood_fill_threshold=1.5)
    
    def test_sort_order_validation(self):
        """Test sort order validation."""
        png_bytes = self._create_mock_png(800, 600)
        
        # Test valid sort orders
        valid_orders = ["top-bottom", "left-right", "reading-order"]
        for order in valid_orders:
            input_data = ImageExtractionInput(png_bytes=png_bytes, sort_order=order)
            assert input_data.sort_order == order
        
        # Test invalid sort order
        with pytest.raises(ValueError):
            ImageExtractionInput(png_bytes=png_bytes, sort_order="invalid-order")
    
    def _create_mock_png(self, width, height):
        """Helper method to create mock PNG data."""
        img = Image.new('RGB', (width, height), color='white')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()


class TestImageExtractionOutput:
    """Test cases for ImageExtractionOutput model."""
    
    def test_valid_output(self):
        """Test valid ImageExtractionOutput creation."""
        # Create mock base64 images
        img1 = base64.b64encode(b"mock_png_data_1").decode('utf-8')
        img2 = base64.b64encode(b"mock_png_data_2").decode('utf-8')
        
        output = ImageExtractionOutput(
            extracted_images=[img1, img2],
            total_images=2,
            filtered_count=5,
            processing_time_ms=150.5
        )
        
        assert len(output.extracted_images) == 2
        assert output.total_images == 2
        assert output.filtered_count == 5
        assert output.processing_time_ms == 150.5
    
    def test_total_size_bytes(self):
        """Test total_size_bytes property."""
        img1 = base64.b64encode(b"mock_png_data_1").decode('utf-8')
        img2 = base64.b64encode(b"mock_png_data_2").decode('utf-8')
        
        output = ImageExtractionOutput(
            extracted_images=[img1, img2],
            total_images=2,
            filtered_count=0,
            processing_time_ms=100.0
        )
        
        expected_size = len(b"mock_png_data_1") + len(b"mock_png_data_2")
        assert output.total_size_bytes == expected_size


class TestPDFProcessor:
    """Test cases for PDFProcessor class."""
    
    def test_init(self):
        """Test PDFProcessor initialization."""
        processor = PDFProcessor()
        assert processor is not None
    
    def test_validate_pdf_bytes(self):
        """Test PDF bytes validation."""
        processor = PDFProcessor()
        
        # Test valid PDF
        valid_pdf = b"%PDF-1.4\n%Test PDF content"
        processor._validate_pdf_bytes(valid_pdf)  # Should not raise
        
        # Test invalid PDF
        with pytest.raises(ValueError, match="Invalid PDF data"):
            processor._validate_pdf_bytes(b"Not a PDF")
    
    @patch('pdfalchemy.core.convert_from_bytes')
    def test_to_png_success(self, mock_convert):
        """Test successful PDF to PNG conversion."""
        # Mock PIL images
        mock_img1 = Mock()
        mock_img2 = Mock()
        
        # Mock image save
        buffer1 = io.BytesIO()
        buffer2 = io.BytesIO()
        mock_img1.save.return_value = None
        mock_img2.save.return_value = None
        
        # Mock convert_from_bytes
        mock_convert.return_value = [mock_img1, mock_img2]
        
        processor = PDFProcessor()
        input_data = PNGConversionInput(
            pdf_bytes=b"%PDF-1.4\n%Test PDF content",
            dpi=300,
            first_page=1,
            last_page=2
        )
        
        # Mock the save method to return bytes
        def mock_save(buffer, format):
            buffer.write(b"mock_png_data")
        
        mock_img1.save.side_effect = mock_save
        mock_img2.save.side_effect = mock_save
        
        result = processor.to_png(input_data)
        
        assert isinstance(result, PNGConversionOutput)
        assert result.total_pages == 2
        assert result.dpi_used == 300
        assert result.page_range == "1-2"
        assert len(result.png_images) == 2
        
        mock_convert.assert_called_once_with(
            input_data.pdf_bytes,
            dpi=300,
            first_page=1,
            last_page=2
        )
    
    @patch('pdfalchemy.core.convert_from_bytes')
    def test_to_png_failure(self, mock_convert):
        """Test PDF to PNG conversion failure."""
        mock_convert.side_effect = Exception("Conversion failed")
        
        processor = PDFProcessor()
        input_data = PNGConversionInput(
            pdf_bytes=b"%PDF-1.4\n%Test PDF content"
        )
        
        with pytest.raises(RuntimeError, match="Failed to convert PDF to PNG"):
            processor.to_png(input_data)
    
    @patch('pdfalchemy.core.convert_from_bytes')
    def test_to_png_bytes(self, mock_convert):
        """Test to_png_bytes method."""
        # Mock PIL images
        mock_img1 = Mock()
        mock_img2 = Mock()
        
        # Mock convert_from_bytes
        mock_convert.return_value = [mock_img1, mock_img2]
        
        processor = PDFProcessor()
        input_data = PNGConversionInput(
            pdf_bytes=b"%PDF-1.4\n%Test PDF content"
        )
        
        # Mock the save method to return bytes
        def mock_save(buffer, format):
            buffer.write(b"mock_png_data")
        
        mock_img1.save.side_effect = mock_save
        mock_img2.save.side_effect = mock_save
        
        result = processor.to_png_bytes(input_data)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(img, bytes) for img in result)
    
    @patch('pdfalchemy.core.convert_from_bytes')
    def test_to_png_base64(self, mock_convert):
        """Test to_png_base64 method."""
        # Mock PIL images
        mock_img1 = Mock()
        mock_img2 = Mock()
        
        # Mock convert_from_bytes
        mock_convert.return_value = [mock_img1, mock_img2]
        
        processor = PDFProcessor()
        input_data = PNGConversionInput(
            pdf_bytes=b"%PDF-1.4\n%Test PDF content"
        )
        
        # Mock the save method to return bytes
        def mock_save(buffer, format):
            buffer.write(b"mock_png_data")
        
        mock_img1.save.side_effect = mock_save
        mock_img2.save.side_effect = mock_save
        
        result = processor.to_png_base64(input_data)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(img, str) for img in result)
        
        # Verify base64 encoding
        for img_str in result:
            decoded = base64.b64decode(img_str)
            assert decoded == b"mock_png_data"
    
    @patch('cv2.imdecode')
    @patch('cv2.cvtColor')
    @patch('cv2.threshold')
    @patch('cv2.findContours')
    @patch('cv2.boundingRect')
    @patch('cv2.contourArea')
    @patch('cv2.imencode')
    def test_extract_images_from_png_success(self, mock_imencode, mock_contour_area, 
                                           mock_bounding_rect, mock_find_contours,
                                           mock_threshold, mock_cvt_color, mock_imdecode):
        """Test successful image extraction from PNG."""
        # Create mock PNG data
        png_bytes = self._create_mock_png(800, 600)
        
        # Mock OpenCV operations
        mock_img = np.zeros((600, 800, 3), dtype=np.uint8)
        mock_imdecode.return_value = mock_img
        
        mock_gray = np.zeros((600, 800), dtype=np.uint8)
        mock_cvt_color.return_value = mock_gray
        
        mock_threshold.return_value = (None, np.zeros((600, 800), dtype=np.uint8))
        
        # Mock contour finding
        mock_contour = np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]])
        mock_find_contours.return_value = ([mock_contour], None)
        
        # Mock bounding rectangle
        mock_bounding_rect.return_value = (100, 100, 100, 100)
        
        # Mock contour area
        mock_contour_area.return_value = 10000
        
        # Mock image encoding
        mock_imencode.return_value = (True, np.array(b"mock_extracted_image"))
        
        processor = PDFProcessor()
        input_data = ImageExtractionInput(
            png_bytes=png_bytes,
            min_width=50,
            min_height=50
        )
        
        result = processor.extract_images_from_png(input_data)
        
        assert isinstance(result, ImageExtractionOutput)
        assert result.total_images >= 0
        assert result.filtered_count >= 0
        assert result.processing_time_ms >= 0
    
    @patch('cv2.imdecode')
    def test_extract_images_from_png_decode_failure(self, mock_imdecode):
        """Test image extraction with PNG decode failure."""
        mock_imdecode.return_value = None
        
        processor = PDFProcessor()
        # Use valid PNG header but invalid content
        png_bytes = b'\x89PNG\r\n\x1a\n' + b"invalid_png_content"
        input_data = ImageExtractionInput(png_bytes=png_bytes)
        
        with pytest.raises(RuntimeError, match="Failed to decode PNG bytes to image"):
            processor.extract_images_from_png(input_data)
    
    def test_extract_images_from_png_general_failure(self):
        """Test image extraction with general failure."""
        processor = PDFProcessor()
        # Use valid PNG header but invalid content that will cause processing to fail
        png_bytes = b'\x89PNG\r\n\x1a\n' + b"invalid_data"
        input_data = ImageExtractionInput(png_bytes=png_bytes)
        
        with pytest.raises(RuntimeError, match="Failed to extract images from PNG"):
            processor.extract_images_from_png(input_data)
    
    def _create_mock_png(self, width, height):
        """Helper method to create mock PNG data."""
        img = Image.new('RGB', (width, height), color='white')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()


class TestIntegration:
    """Integration tests for the core functionality."""
    
    @patch('pdfalchemy.core.convert_from_bytes')
    @patch('cv2.imdecode')
    @patch('cv2.cvtColor')
    @patch('cv2.threshold')
    @patch('cv2.findContours')
    @patch('cv2.boundingRect')
    @patch('cv2.contourArea')
    @patch('cv2.imencode')
    def test_full_pipeline(self, mock_imencode, mock_contour_area, mock_bounding_rect,
                          mock_find_contours, mock_threshold, mock_cvt_color, 
                          mock_imdecode, mock_convert):
        """Test the full PDF to extracted images pipeline."""
        # Mock PDF to PNG conversion
        mock_img = Mock()
        mock_convert.return_value = [mock_img]
        
        def mock_save(buffer, format):
            # Create a minimal valid PNG
            png_header = b'\x89PNG\r\n\x1a\n'
            png_data = png_header + b"mock_png_data"
            buffer.write(png_data)
        mock_img.save.side_effect = mock_save
        
        # Mock image extraction
        mock_img_array = np.zeros((600, 800, 3), dtype=np.uint8)
        mock_imdecode.return_value = mock_img_array
        
        mock_gray = np.zeros((600, 800), dtype=np.uint8)
        mock_cvt_color.return_value = mock_gray
        
        mock_threshold.return_value = (None, np.zeros((600, 800), dtype=np.uint8))
        
        mock_contour = np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]])
        mock_find_contours.return_value = ([mock_contour], None)
        
        mock_bounding_rect.return_value = (100, 100, 100, 100)
        mock_contour_area.return_value = 10000
        mock_imencode.return_value = (True, np.array(b"mock_extracted_image"))
        
        processor = PDFProcessor()
        
        # Step 1: Convert PDF to PNG
        png_input = PNGConversionInput(
            pdf_bytes=b"%PDF-1.4\n%Test PDF content",
            dpi=300
        )
        png_result = processor.to_png(png_input)
        
        # Step 2: Extract images from PNG
        extraction_input = ImageExtractionInput(
            png_bytes=png_result.png_images[0],
            min_width=50,
            min_height=50
        )
        extraction_result = processor.extract_images_from_png(extraction_input)
        
        assert isinstance(png_result, PNGConversionOutput)
        assert isinstance(extraction_result, ImageExtractionOutput)
        assert png_result.total_pages == 1
        assert extraction_result.total_images >= 0 