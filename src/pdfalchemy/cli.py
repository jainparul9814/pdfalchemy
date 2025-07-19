"""
Command-line interface for PDFAlchemy.
"""

import argparse
import json
import sys
import base64
from pathlib import Path
from typing import List, Optional, Tuple
from pdfalchemy.core import PDFProcessor, PNGConversionInput, ImageExtractionInput


def parse_page_range(page_range: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse page range string into first_page and last_page.
    
    Args:
        page_range: String like "1-5", "1,3,5", or "3"
        
    Returns:
        Tuple of (first_page, last_page) or (None, None) if invalid
    """
    if not page_range:
        return None, None
    
    try:
        if "-" in page_range:
            # Range format: "1-5"
            parts = page_range.split("-")
            if len(parts) == 2:
                first_page = int(parts[0].strip())
                last_page = int(parts[1].strip())
                return first_page, last_page
        elif "," in page_range:
            # List format: "1,3,5" - we'll use the first and last
            pages = [int(p.strip()) for p in page_range.split(",")]
            if pages:
                return min(pages), max(pages)
        else:
            # Single page: "3"
            page = int(page_range.strip())
            return page, page
    except ValueError:
        pass
    
    return None, None


def main(args: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="PDFAlchemy - Advanced PDF processing and image extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert PDF to PNG images
  pdfalchemy to-png document.pdf --output ./images/
  pdfalchemy to-png document.pdf --dpi 300 --pages 1-5
  
  # Convert to base64 for web applications
  pdfalchemy to-base64 document.pdf --dpi 200 --output images.json
  
  # Extract images from PDF pages
  pdfalchemy extract-images document.pdf --output ./extracted/ --min-size 100x100
  
  # Extract images with custom filters
  pdfalchemy extract-images document.pdf --min-width 50 --max-width 800 --aspect-ratio 0.5-2.0
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # to-png command
    png_parser = subparsers.add_parser("to-png", help="Convert PDF to PNG images")
    png_parser.add_argument("file", help="PDF file to process")
    png_parser.add_argument("--output", "-o", help="Output directory for PNG files")
    png_parser.add_argument("--dpi", type=int, default=200, help="DPI resolution (default: 200)")
    png_parser.add_argument("--pages", help="Page range (e.g., '1-5', '1,3,5', or '3')")
    
    # to-base64 command
    base64_parser = subparsers.add_parser("to-base64", help="Convert PDF to base64 encoded PNG images")
    base64_parser.add_argument("file", help="PDF file to process")
    base64_parser.add_argument("--dpi", type=int, default=200, help="DPI resolution (default: 200)")
    base64_parser.add_argument("--pages", help="Page range (e.g., '1-5', '1,3,5', or '3')")
    base64_parser.add_argument("--output", "-o", help="Output file for base64 data")
    
    # extract-images command
    extract_parser = subparsers.add_parser("extract-images", help="Extract individual images from PDF pages")
    extract_parser.add_argument("file", help="PDF file to process")
    extract_parser.add_argument("--output", "-o", help="Output directory for extracted images")
    extract_parser.add_argument("--dpi", type=int, default=200, help="DPI resolution for conversion (default: 200)")
    extract_parser.add_argument("--pages", help="Page range (e.g., '1-5', '1,3,5', or '3')")
    
    # Image extraction filters
    extract_parser.add_argument("--min-size", help="Minimum size in pixels (e.g., '100x100')")
    extract_parser.add_argument("--max-size", help="Maximum size in pixels (e.g., '800x600')")
    extract_parser.add_argument("--min-width", type=int, help="Minimum width in pixels")
    extract_parser.add_argument("--min-height", type=int, help="Minimum height in pixels")
    extract_parser.add_argument("--max-width", type=int, help="Maximum width in pixels")
    extract_parser.add_argument("--max-height", type=int, help="Maximum height in pixels")
    extract_parser.add_argument("--aspect-ratio", help="Aspect ratio range (e.g., '0.5-2.0')")
    extract_parser.add_argument("--threshold", type=float, default=0.1, help="Flood fill threshold (0.0-1.0, default: 0.1)")
    extract_parser.add_argument("--no-noise-reduction", action="store_true", help="Disable noise reduction")
    extract_parser.add_argument("--no-separate-regions", action="store_true", help="Disable connected region separation")
    extract_parser.add_argument("--sort-order", choices=["top-bottom", "left-right", "reading-order"], 
                               default="top-bottom", help="Sort order for extracted images (default: top-bottom)")
    
    # Output format options
    extract_parser.add_argument("--format", choices=["png", "json"], default="png", 
                               help="Output format (default: png)")
    extract_parser.add_argument("--summary", action="store_true", help="Show extraction summary")
    
    parsed_args = parser.parse_args(args)
    
    if not parsed_args.command:
        parser.print_help()
        return 1
    
    try:
        if parsed_args.command == "to-png":
            return _handle_to_png(parsed_args)
        elif parsed_args.command == "to-base64":
            return _handle_to_base64(parsed_args)
        elif parsed_args.command == "extract-images":
            return _handle_extract_images(parsed_args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def _handle_to_png(args) -> int:
    """Handle to-png command."""
    try:
        # Read PDF file
        with open(args.file, 'rb') as f:
            pdf_bytes = f.read()
        
        # Parse page range
        first_page, last_page = parse_page_range(args.pages)
        
        # Create input model
        input_data = PNGConversionInput(
            pdf_bytes=pdf_bytes,
            dpi=args.dpi,
            first_page=first_page,
            last_page=last_page
        )
        
        # Process PDF
        processor = PDFProcessor()
        result = processor.to_png(input_data)
        
        # Save PNG files
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, png_bytes in enumerate(result.png_images):
                output_file = output_dir / f"page_{i+1:03d}.png"
                with open(output_file, 'wb') as f:
                    f.write(png_bytes)
            
            print(f"Converted {result.total_pages} pages to {output_dir}")
            if result.page_range:
                print(f"Page range: {result.page_range}")
            print(f"DPI used: {result.dpi_used}")
            print(f"Total size: {result.total_size_bytes:,} bytes")
        else:
            print(f"Converted {result.total_pages} pages (use --output to save files)")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _handle_to_base64(args) -> int:
    """Handle to-base64 command."""
    try:
        # Read PDF file
        with open(args.file, 'rb') as f:
            pdf_bytes = f.read()
        
        # Parse page range
        first_page, last_page = parse_page_range(args.pages)
        
        # Create input model
        input_data = PNGConversionInput(
            pdf_bytes=pdf_bytes,
            dpi=args.dpi,
            first_page=first_page,
            last_page=last_page
        )
        
        # Process PDF
        processor = PDFProcessor()
        base64_images = processor.to_png_base64(input_data)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(base64_images, f, indent=2)
            print(f"Converted {len(base64_images)} pages to {args.output}")
        else:
            for i, base64_str in enumerate(base64_images):
                print(f"Page {i+1}: {base64_str[:50]}...")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _parse_size_constraint(size_str: str) -> Tuple[int, int]:
    """Parse size constraint string like '100x100'."""
    if not size_str:
        return None, None
    
    try:
        width, height = map(int, size_str.split('x'))
        return width, height
    except (ValueError, AttributeError):
        return None, None


def _parse_aspect_ratio_constraint(ratio_str: str) -> Tuple[float, float]:
    """Parse aspect ratio constraint string like '0.5-2.0'."""
    if not ratio_str:
        return None, None
    
    try:
        if "-" in ratio_str:
            min_ratio, max_ratio = map(float, ratio_str.split("-"))
            return min_ratio, max_ratio
        else:
            ratio = float(ratio_str)
            return ratio, ratio
    except (ValueError, AttributeError):
        return None, None


def _handle_extract_images(args) -> int:
    """Handle extract-images command."""
    try:
        # Read PDF file
        with open(args.file, 'rb') as f:
            pdf_bytes = f.read()
        
        # Parse page range
        first_page, last_page = parse_page_range(args.pages)
        
        # Create PNG conversion input
        png_input = PNGConversionInput(
            pdf_bytes=pdf_bytes,
            dpi=args.dpi,
            first_page=first_page,
            last_page=last_page
        )
        
        # Process PDF to PNG
        processor = PDFProcessor()
        png_result = processor.to_png(png_input)
        
        # Parse size constraints
        min_width, min_height = _parse_size_constraint(args.min_size)
        max_width, max_height = _parse_size_constraint(args.max_size)
        
        # Override with individual constraints if provided
        if args.min_width is not None:
            min_width = args.min_width
        if args.min_height is not None:
            min_height = args.min_height
        if args.max_width is not None:
            max_width = args.max_width
        if args.max_height is not None:
            max_height = args.max_height
        
        # Parse aspect ratio constraints
        min_aspect_ratio, max_aspect_ratio = _parse_aspect_ratio_constraint(args.aspect_ratio)
        
        # Extract images from each PNG
        total_extracted = 0
        total_filtered = 0
        all_extracted_images = []
        
        for i, png_bytes in enumerate(png_result.png_images):
            extraction_input = ImageExtractionInput(
                png_bytes=png_bytes,
                min_width=min_width or 50,
                min_height=min_height or 50,
                max_width=max_width,
                max_height=max_height,
                min_aspect_ratio=min_aspect_ratio,
                max_aspect_ratio=max_aspect_ratio,
                flood_fill_threshold=args.threshold,
                noise_reduction=not args.no_noise_reduction,
                separate_connected_regions=not args.no_separate_regions,
                sort_order=args.sort_order
            )
            
            extraction_result = processor.extract_images_from_png(extraction_input)
            total_extracted += extraction_result.total_images
            total_filtered += extraction_result.filtered_count
            
            # Save extracted images
            if args.output and extraction_result.extracted_images:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                for j, img_base64 in enumerate(extraction_result.extracted_images):
                    img_bytes = base64.b64decode(img_base64)
                    
                    if args.format == "png":
                        output_file = output_dir / f"page_{i+1:03d}_image_{j+1:03d}.png"
                        with open(output_file, 'wb') as f:
                            f.write(img_bytes)
                    else:  # json format
                        all_extracted_images.append({
                            "page": i + 1,
                            "image_index": j + 1,
                            "base64_data": img_base64,
                            "size_bytes": len(img_bytes)
                        })
            
            if args.summary:
                print(f"Page {i+1}: Extracted {extraction_result.total_images} images, "
                      f"filtered {extraction_result.filtered_count} images "
                      f"({extraction_result.processing_time_ms:.1f}ms)")
        
        # Save JSON output if requested
        if args.output and args.format == "json" and all_extracted_images:
            output_file = Path(args.output) / "extracted_images.json"
            with open(output_file, 'w') as f:
                json.dump(all_extracted_images, f, indent=2)
            print(f"Saved {len(all_extracted_images)} images to {output_file}")
        
        # Print summary
        print(f"\nExtraction Summary:")
        print(f"  Total pages processed: {png_result.total_pages}")
        print(f"  Total images extracted: {total_extracted}")
        print(f"  Total images filtered: {total_filtered}")
        print(f"  Output format: {args.format}")
        if args.output:
            print(f"  Output directory: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 