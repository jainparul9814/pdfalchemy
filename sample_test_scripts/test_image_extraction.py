#!/usr/bin/env python3
"""
Test script for PDF to PNG conversion and image extraction.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path so we can import pdfalchemy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pdfalchemy import PDFProcessor, PNGConversionInput, ImageExtractionInput, ImageExtractionOutput


def test_pdf_to_png_and_extract_images(pdf_path: str):
    """
    Test the complete workflow: PDF -> PNG -> Image Extraction
    
    Args:
        pdf_path: Path to the PDF file to process
    """
    print(f"Processing PDF: {pdf_path}")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} does not exist")
        return
    
    try:
        # Step 1: Read PDF file and convert to bytes
        print("Step 1: Reading PDF file...")
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        print(f"PDF size: {len(pdf_bytes)} bytes")
        
        # Step 2: Convert PDF to PNG
        print("Step 2: Converting PDF to PNG...")
        processor = PDFProcessor()
        
        png_input = PNGConversionInput(
            pdf_bytes=pdf_bytes,
            dpi=200,  # Medium resolution
        )
        
        png_output = processor.to_png(png_input)
        print(f"Converted {png_output.total_pages} page(s) to PNG")
        print(f"Total PNG size: {png_output.total_size_bytes} bytes")
        
        # Step 3: Extract images from each PNG page
        total_extracted = 0
        total_filtered = 0
        
        if png_output.png_images:
            print("Step 3: Extracting images from PNG pages...")
            
            # Process each page
            for page_idx, png_bytes in enumerate(png_output.png_images):
                page_num = page_idx + 1
                print(f"\nProcessing page {page_num}...")
                
                extraction_input = ImageExtractionInput(
                    png_bytes=png_bytes,
                    min_width=50,    # Minimum 50px width
                    min_height=50,   # Minimum 50px height
                    flood_fill_threshold=0.2,  # 20% threshold
                    noise_reduction=True,
                    min_aspect_ratio=0.6,
                    max_aspect_ratio=2.0,
                    separate_connected_regions=True,
                    sort_order="reading-order"  # Sort in reading order (top-bottom, left-right)
                )
                
                extraction_output = processor.extract_images_from_png(extraction_input)
                
                print(f"Page {page_num} Results:")
                print(f"  - Images extracted: {extraction_output.total_images}")
                print(f"  - Images filtered out: {extraction_output.filtered_count}")
                print(f"  - Processing time: {extraction_output.processing_time_ms:.2f} ms")
                print(f"  - Extracted size: {extraction_output.total_size_bytes} bytes")
                
                total_extracted += extraction_output.total_images
                total_filtered += extraction_output.filtered_count
                
                # Save extracted images to files for inspection
                if extraction_output.extracted_images:
                    output_dir = Path("extracted_images")
                    output_dir.mkdir(exist_ok=True)
                    
                    print(f"  - Saving {len(extraction_output.extracted_images)} images from page {page_num}")
                    for img_idx, img_base64 in enumerate(extraction_output.extracted_images):
                        import base64
                        img_bytes = base64.b64decode(img_base64)
                        
                        # Use page number and image index for proper ordering
                        output_path = output_dir / f"page_{page_num:03d}_image_{img_idx+1:03d}.png"
                        
                        with open(output_path, 'wb') as f:
                            f.write(img_bytes)
                        
                        print(f"    - Saved: {output_path}")
                else:
                    print(f"  - No images extracted from page {page_num}")
        else:
            print("No PNG images were generated")
        
        # Print summary
        print(f"\n=== EXTRACTION SUMMARY ===")
        print(f"Total pages processed: {png_output.total_pages}")
        print(f"Total images extracted: {total_extracted}")
        print(f"Total images filtered: {total_filtered}")
        print(f"Output directory: extracted_images/")
        if total_extracted > 0:
            print(f"Files saved with format: page_XXX_image_YYY.png")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run the test."""
    # Use hardcoded path for testing
    pdf_path = "<your_pdf_path>"
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} does not exist")
        print("Please update the pdf_path variable in the script with a valid PDF file path")
        sys.exit(1)
    
    test_pdf_to_png_and_extract_images(pdf_path)


if __name__ == "__main__":
    main() 