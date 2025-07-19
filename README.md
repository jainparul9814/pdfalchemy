# PDFAlchemy

[![PyPI version](https://badge.fury.io/py/pdfalchemy.svg)](https://badge.fury.io/py/pdfalchemy)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A powerful Python library for advanced PDF processing with focus on image extraction and conversion capabilities.

## Features

- **PDF to PNG Conversion**: Convert PDF pages to high-quality PNG images with customizable DPI
- **Image Extraction**: Extract individual images from PDF pages using advanced computer vision algorithms
- **Flood Fill Algorithm**: Intelligent image detection and separation using morphological operations
- **Size and Aspect Ratio Filtering**: Filter extracted images based on customizable criteria
- **Base64 Encoding**: Convert PDF pages to base64-encoded PNG strings for web applications
- **Command Line Interface**: Easy-to-use CLI for batch processing and automation
- **Type Safety**: Full type hints and Pydantic validation for robust data handling
- **Comprehensive Testing**: Extensive test suite with 28+ test cases

## Installation

### From PyPI (Recommended)

```bash
pip install pdfalchemy
```

### From Source

```bash
git clone https://github.com/jainparul9814/pdfalchemy.git
cd pdfalchemy

# Install base dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

## Quick Start

### Python API

```python
from pdfalchemy import PDFProcessor, PNGConversionInput, ImageExtractionInput

# Initialize processor
processor = PDFProcessor()

# Convert PDF to PNG
with open("document.pdf", "rb") as f:
    pdf_bytes = f.read()

png_input = PNGConversionInput(
    pdf_bytes=pdf_bytes,
    dpi=300,  # High resolution
    first_page=1,
    last_page=5
)

png_result = processor.to_png(png_input)
print(f"Converted {png_result.total_pages} pages")

# Extract images from PNG
for i, png_bytes in enumerate(png_result.png_images):
    extraction_input = ImageExtractionInput(
        png_bytes=png_bytes,
        min_width=50,
        min_height=50,
        flood_fill_threshold=0.2,
        noise_reduction=True
    )
    
    extraction_result = processor.extract_images_from_png(extraction_input)
    print(f"Page {i+1}: Extracted {extraction_result.total_images} images")
```

### Command Line Interface

```bash
# Convert PDF to PNG images
pdfalchemy to-png document.pdf --output ./images/ --dpi 300

# Convert specific pages (range, list, or single page)
pdfalchemy to-png document.pdf --pages 1-5 --dpi 200
pdfalchemy to-png document.pdf --pages 1,3,5 --dpi 200
pdfalchemy to-png document.pdf --pages 3 --dpi 200

# Convert to base64 for web applications
pdfalchemy to-base64 document.pdf --dpi 200 --output images.json

# Extract individual images from PDF pages
pdfalchemy extract-images document.pdf --output ./extracted/ --min-size 100x100

# Extract images with custom filters
pdfalchemy extract-images document.pdf --min-width 50 --max-width 800 --aspect-ratio 0.5-2.0

# Extract images with advanced options
pdfalchemy extract-images document.pdf \
  --output ./extracted/ \
  --dpi 300 \
  --pages 1-5 \
  --min-size 100x100 \
  --max-size 800x600 \
  --aspect-ratio 0.5-2.0 \
  --threshold 0.15 \
  --format json \
  --summary

# Get help for any command
pdfalchemy extract-images --help
```

## Advanced Usage

### Image Extraction with Custom Filters

```python
from pdfalchemy import PDFProcessor, ImageExtractionInput

processor = PDFProcessor()

# Configure image extraction with specific criteria
extraction_input = ImageExtractionInput(
    png_bytes=png_bytes,
    min_width=100,           # Minimum width in pixels
    min_height=100,          # Minimum height in pixels
    max_width=800,           # Maximum width in pixels
    max_height=600,          # Maximum height in pixels
    min_aspect_ratio=0.5,    # Minimum aspect ratio (width/height)
    max_aspect_ratio=2.0,    # Maximum aspect ratio
    flood_fill_threshold=0.15,  # Threshold for flood fill algorithm
    noise_reduction=True,    # Enable noise reduction
    separate_connected_regions=True  # Separate connected regions
)

result = processor.extract_images_from_png(extraction_input)
print(f"Extracted {result.total_images} images")
print(f"Filtered out {result.filtered_count} images")
print(f"Processing time: {result.processing_time_ms:.2f} ms")
```

### Batch Processing

```python
from pathlib import Path
from pdfalchemy import PDFProcessor, PNGConversionInput

processor = PDFProcessor()
pdf_files = Path("./pdfs/").glob("*.pdf")

for pdf_file in pdf_files:
    print(f"Processing {pdf_file}")
    
    with open(pdf_file, "rb") as f:
        pdf_bytes = f.read()
    
    png_input = PNGConversionInput(
        pdf_bytes=pdf_bytes,
        dpi=200
    )
    
    result = processor.to_png(png_input)
    print(f"  Converted {result.total_pages} pages")
```

### Base64 Conversion for Web Applications

```python
from pdfalchemy import PDFProcessor, PNGConversionInput

processor = PDFProcessor()

with open("document.pdf", "rb") as f:
    pdf_bytes = f.read()

png_input = PNGConversionInput(
    pdf_bytes=pdf_bytes,
    dpi=200
)

# Get base64 encoded PNG images
base64_images = processor.to_png_base64(png_input)

# Use in web applications
for i, base64_str in enumerate(base64_images):
    html_img_tag = f'<img src="data:image/png;base64,{base64_str}" alt="Page {i+1}">'
    print(html_img_tag)
```

## Command Line Interface

PDFAlchemy provides a powerful command-line interface for batch processing and automation.

### Available Commands

#### `to-png` - Convert PDF to PNG Images
```bash
pdfalchemy to-png <pdf_file> [options]
```

**Options:**
- `--output, -o`: Output directory for PNG files
- `--dpi`: DPI resolution (default: 200, range: 72-1200)
- `--pages`: Page range (e.g., '1-5', '1,3,5', or '3')

**Examples:**
```bash
# Convert all pages
pdfalchemy to-png document.pdf --output ./images/

# Convert specific pages with high resolution
pdfalchemy to-png document.pdf --dpi 300 --pages 1-5 --output ./high_res/

# Convert single page
pdfalchemy to-png document.pdf --pages 3 --output ./single_page/
```

#### `to-base64` - Convert PDF to Base64 Encoded PNG
```bash
pdfalchemy to-base64 <pdf_file> [options]
```

**Options:**
- `--output, -o`: Output file for base64 data (JSON format)
- `--dpi`: DPI resolution (default: 200)
- `--pages`: Page range (e.g., '1-5', '1,3,5', or '3')

**Examples:**
```bash
# Convert to base64 for web applications
pdfalchemy to-base64 document.pdf --dpi 200 --output images.json

# Convert specific pages
pdfalchemy to-base64 document.pdf --pages 1-3 --output selected_pages.json
```

#### `extract-images` - Extract Individual Images from PDF
```bash
pdfalchemy extract-images <pdf_file> [options]
```

**Basic Options:**
- `--output, -o`: Output directory for extracted images
- `--dpi`: DPI resolution for conversion (default: 200)
- `--pages`: Page range (e.g., '1-5', '1,3,5', or '3')

**Size Filtering:**
- `--min-size`: Minimum size in pixels (e.g., '100x100')
- `--max-size`: Maximum size in pixels (e.g., '800x600')
- `--min-width`: Minimum width in pixels
- `--min-height`: Minimum height in pixels
- `--max-width`: Maximum width in pixels
- `--max-height`: Maximum height in pixels

**Advanced Filtering:**
- `--aspect-ratio`: Aspect ratio range (e.g., '0.5-2.0')
- `--threshold`: Flood fill threshold (0.0-1.0, default: 0.1)
- `--no-noise-reduction`: Disable noise reduction
- `--no-separate-regions`: Disable connected region separation
- `--sort-order`: Sort order for extracted images ('top-bottom', 'left-right', 'reading-order', default: 'top-bottom')

**Output Options:**
- `--format`: Output format ('png' or 'json', default: 'png')
- `--summary`: Show detailed extraction summary

**Examples:**
```bash
# Basic image extraction
pdfalchemy extract-images document.pdf --output ./extracted/ --min-size 100x100

# Advanced filtering with custom sort order
pdfalchemy extract-images document.pdf \
  --output ./filtered/ \
  --min-width 50 \
  --max-width 800 \
  --aspect-ratio 0.5-2.0 \
  --threshold 0.15 \
  --sort-order reading-order

# JSON output with summary
pdfalchemy extract-images document.pdf \
  --output ./json_output/ \
  --format json \
  --summary \
  --pages 1-5

# High-resolution extraction with custom filters
pdfalchemy extract-images document.pdf \
  --dpi 300 \
  --output ./high_res_extracted/ \
  --min-size 200x200 \
  --max-size 1200x800 \
  --aspect-ratio 0.8-1.5 \
  --threshold 0.2 \
  --no-noise-reduction
```

### Page Range Formats

The `--pages` option supports multiple formats:
- **Range**: `1-5` (pages 1 through 5)
- **List**: `1,3,5` (pages 1, 3, and 5)
- **Single**: `3` (page 3 only)

### Output Formats

#### PNG Format
- Saves individual PNG files for each extracted image
- File naming: `page_001_image_001.png`, `page_001_image_002.png`, etc.
- Suitable for visual inspection and further processing

#### JSON Format
- Saves all extracted images as base64-encoded data in a JSON file
- Includes metadata: page number, image index, size in bytes
- Suitable for web applications and programmatic access

### Sort Order Options

The `--sort-order` parameter controls how extracted images are ordered:

- **`top-bottom`** (default): Sort by y-coordinate first (top to bottom), then by x-coordinate (left to right)
- **`left-right`**: Sort by x-coordinate first (left to right), then by y-coordinate (top to bottom)
- **`reading-order`**: Group images by approximate rows and sort each row left-to-right, then sort rows top-to-bottom

### Performance Tips

1. **Use appropriate DPI**: Higher DPI provides better quality but increases processing time
2. **Filter early**: Use size and aspect ratio filters to reduce processing overhead
3. **Batch processing**: Process multiple files in scripts for automation
4. **Memory management**: For large PDFs, consider processing page ranges

## Configuration

PDFAlchemy uses Pydantic models for configuration and validation. All input and output models include comprehensive validation and type checking.

## Data Models

### PNGConversionInput
- `pdf_bytes`: PDF data as byte array
- `dpi`: Resolution in DPI (72-1200, default: 200)
- `first_page`: First page to convert (1-indexed, optional)
- `last_page`: Last page to convert (1-indexed, optional)

### PNGConversionOutput
- `png_images`: List of PNG images as byte arrays
- `total_pages`: Total number of pages converted
- `dpi_used`: DPI used for conversion
- `page_range`: Page range converted (e.g., '1-5')
- `total_size_bytes`: Total size of all PNG images

### ImageExtractionInput
- `png_bytes`: PNG image data as byte array
- `min_width/min_height`: Minimum dimensions for extracted images
- `max_width/max_height`: Maximum dimensions for extracted images
- `min_aspect_ratio/max_aspect_ratio`: Aspect ratio constraints
- `flood_fill_threshold`: Threshold for flood fill algorithm (0.0-1.0)
- `noise_reduction`: Enable noise reduction
- `separate_connected_regions`: Attempt to separate connected regions
- `sort_order`: Sort order for extracted images ('top-bottom', 'left-right', 'reading-order')

### ImageExtractionOutput
- `extracted_images`: List of base64 encoded extracted images
- `total_images`: Total number of extracted images
- `filtered_count`: Number of images filtered out
- `processing_time_ms`: Processing time in milliseconds
- `total_size_bytes`: Total size of all extracted images

## Development

### Setup Development Environment

```bash
git clone https://github.com/jainparul9814/pdfalchemy.git
cd pdfalchemy

# Install all dependencies
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .

# Setup pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run core tests with verbose output
pytest tests/test_core.py -v

# Run with coverage
pytest --cov=src.pdfalchemy.core --cov-report=term-missing
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Sample Scripts

Check the `sample_test_scripts/` directory for working examples:

```bash
python sample_test_scripts/test_image_extraction.py
```

## Dependencies

### Core Dependencies
- `pydantic>=2.0.0`: Data validation and settings management
- `pdf2image>=1.16.0`: PDF to image conversion
- `opencv-python>=4.8.0`: Computer vision for image processing
- `Pillow>=9.0.0`: Image processing
- `numpy>=1.21.0`: Numerical computing

### Development Dependencies
- `pytest>=7.0.0`: Testing framework
- `black>=23.0.0`: Code formatting
- `isort>=5.12.0`: Import sorting
- `flake8>=6.0.0`: Linting
- `mypy>=1.0.0`: Type checking

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/jainparul9814/pdfalchemy/issues)
- **Author**: Parul Jain (jainparul9814@gmail.com)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history. 