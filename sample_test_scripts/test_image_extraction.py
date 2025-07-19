#!/usr/bin/env python3
"""
Test script for PDF to PNG conversion and image extraction.
"""

import sys
import os
import base64
import time
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
        
        # Step 3: Extract images from the first PNG
        count = 0
        if png_output.png_images:
            print("Step 3: Extracting images from PNG...")
            
            # Use the first PNG image
            for png_bytes in png_output.png_images:
            
              extraction_input = ImageExtractionInput(
                  png_bytes=png_bytes,
                  min_width=50,    # Minimum 50px width
                  min_height=50,   # Minimum 50px height
                  flood_fill_threshold=0.2,  # 20% threshold
                  noise_reduction=True,
                  min_aspect_ratio=0.6,
                  max_aspect_ratio=2.0,
                  separate_connected_regions=True
              )
              
              extraction_output = processor.extract_images_from_png(extraction_input)
              
              print(f"Extraction Results:")
              print(f"  - Total images extracted: {extraction_output.total_images}")
              print(f"  - Images filtered out: {extraction_output.filtered_count}")
              print(f"  - Processing time: {extraction_output.processing_time_ms:.2f} ms")
              print(f"  - Total extracted size: {extraction_output.total_size_bytes} bytes")
              
              # Save extracted images to files for inspection
              if extraction_output.extracted_images:
                  output_dir = Path("extracted_images")
                  output_dir.mkdir(exist_ok=True)
                  
                  print(f"Saving extracted images to: {output_dir}")
                  for i, img_base64 in enumerate(extraction_output.extracted_images):
                      import base64
                      img_bytes = base64.b64decode(img_base64)
                      output_path = output_dir / f"extracted_image_{count}.png"
                      count += 1
                      
                      with open(output_path, 'wb') as f:
                          f.write(img_bytes)
                      
                      print(f"  - Saved: {output_path}")
              else:
                  print("  - No images were extracted")
        else:
            print("No PNG images were generated")
            
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