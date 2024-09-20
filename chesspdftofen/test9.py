import cv2
import numpy as np
import os
from segment_boards import segment_boards
from pdf2image import convert_from_path

# Define paths
pdf_path = r'C:\Users\boldr\Downloads\chesspdftofen-master\chesspdfpage.pdf'
output_dir = r'C:\Users\boldr\Downloads\chesspdftofen-master\cropped_boards'
dpi = 600  # DPI for good quality

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Convert PDF to images with high quality
pages = convert_from_path(pdf_path, dpi=dpi)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Histogram equalization for contrast improvement
    equalized = cv2.equalizeHist(gray)
    
    # Adaptive thresholding
    thresholded = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to enhance board edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresholded, kernel, iterations=2)
    
    return dilated

def detect_boards(image):
    return segment_boards(image)

def process_page(image, page_idx):
    # Process different regions of the page
    height, width = image.shape[:2]
    regions = [
        (0, height, 0, width),  # Full page
        (0, height // 2, 0, width),  # Top half
        (height // 2, height, 0, width),  # Bottom half
        (0, height, 0, width // 2),  # Left half
        (0, height, width // 2, width),  # Right half
        (0, height // 3, 0, width // 3),  # Top-left quarter
        (0, height // 3, 2 * width // 3, width),  # Top-right quarter
        (2 * height // 3, height, 0, width // 3),  # Bottom-left quarter
        (2 * height // 3, height, 2 * width // 3, width),  # Bottom-right quarter
        (height // 3, 2 * height // 3, width // 3, 2 * width // 3),  # Center rectangle
        # Additional new regions
        (0, height // 4, width // 4, 3 * width // 4),  # Top-left vertical strip
        (0, height // 4, width - width // 4, width),  # Top-right vertical strip
        (height - height // 4, height, width // 4, 3 * width // 4),  # Bottom-left vertical strip
        (height - height // 4, height, width - width // 4, width),  # Bottom-right vertical strip
        (height // 4, 3 * height // 4, width // 4, width // 2),  # Center-left vertical strip
        (height // 4, 3 * height // 4, width // 2, width - width // 4)  # Center-right vertical strip
    ]
    
    # Process each region
    for region_idx, (y1, y2, x1, x2) in enumerate(regions):
        region_image = image[y1:y2, x1:x2]
        processed_image = preprocess_image(region_image)
        
        # Debugging: Save the processed image of the region
        processed_image_path = os.path.join(output_dir, f'processed_page_{page_idx + 1}_region_{region_idx + 1}.png')
        cv2.imwrite(processed_image_path, processed_image)
        print(f"Saved processed image for page {page_idx + 1}, region {region_idx + 1} as {processed_image_path}")
        
        boards = detect_boards(processed_image)
        print(f"Detected {len(boards)} boards in page {page_idx + 1}, region {region_idx + 1}")
        
        # Extract and save each board region
        for board_idx, board in enumerate(boards):
            board_y1, board_y2, board_x1, board_x2, w, h = board
            
            # Adjust coordinates relative to the region
            adjusted_board_y1 = board_y1 + y1
            adjusted_board_y2 = board_y2 + y1
            adjusted_board_x1 = board_x1 + x1
            adjusted_board_x2 = board_x2 + x1
            
            # Ensure board coordinates are within image bounds
            adjusted_board_y1, adjusted_board_y2 = max(adjusted_board_y1, 0), min(adjusted_board_y2, image.shape[0])
            adjusted_board_x1, adjusted_board_x2 = max(adjusted_board_x1, 0), min(adjusted_board_x2, image.shape[1])
            
            board_image = image[adjusted_board_y1:adjusted_board_y2, adjusted_board_x1:adjusted_board_x2]
            
            output_image_path = os.path.join(output_dir, f'page_{page_idx + 1}_region_{region_idx + 1}_board_{board_idx + 1}.jpg')
            cv2.imwrite(output_image_path, board_image)
            
            print(f"Saved page {page_idx + 1}, region {region_idx + 1} board {board_idx + 1} as {output_image_path}")

# Process each page
for page_idx, page in enumerate(pages):
    page_cv = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    process_page(page_cv, page_idx)

print(f"All board images saved in {output_dir}")
