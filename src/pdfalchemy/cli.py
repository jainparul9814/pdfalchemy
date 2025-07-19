"""
Command-line interface for PDFAlchemy.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

try:
    from .core import PDFProcessor, PNGConversionInput
    from .config import ExtractionConfig
except ImportError:
    # Handle direct script execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pdfalchemy.core import PDFProcessor, PNGConversionInput
    from pdfalchemy.config import ExtractionConfig


def main(args: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="PDFAlchemy - Advanced PDF processing and extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pdfalchemy to-png document.pdf --output ./images/
  pdfalchemy to-png document.pdf --dpi 300 --pages 1-5
  pdfalchemy to-base64 document.pdf --dpi 200
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # to-png command
    png_parser = subparsers.add_parser("to-png", help="Convert PDF to PNG images")
    png_parser.add_argument("file", help="PDF file to process")
    png_parser.add_argument("--output", "-o", help="Output directory for PNG files")
    png_parser.add_argument("--dpi", type=int, default=200, help="DPI resolution (default: 200)")
    png_parser.add_argument("--pages", help="Page range (e.g., '1-5' or '1,3,5')")
    
    # to-base64 command
    base64_parser = subparsers.add_parser("to-base64", help="Convert PDF to base64 encoded PNG images")
    base64_parser.add_argument("file", help="PDF file to process")
    base64_parser.add_argument("--dpi", type=int, default=200, help="DPI resolution (default: 200)")
    base64_parser.add_argument("--pages", help="Page range (e.g., '1-5' or '1,3,5')")
    base64_parser.add_argument("--output", "-o", help="Output file for base64 data")
    
    parsed_args = parser.parse_args(args)
    
    if not parsed_args.command:
        parser.print_help()
        return 1
    
    try:
        if parsed_args.command == "to-png":
            return _handle_to_png(parsed_args)
        elif parsed_args.command == "to-base64":
            return _handle_to_base64(parsed_args)
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
        
        # Create input model
        input_data = PNGConversionInput(
            pdf_bytes=pdf_bytes,
            dpi=args.dpi
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
        
        # Create input model
        input_data = PNGConversionInput(
            pdf_bytes=pdf_bytes,
            dpi=args.dpi
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


if __name__ == "__main__":
    sys.exit(main()) 