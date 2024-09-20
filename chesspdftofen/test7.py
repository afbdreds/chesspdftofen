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
    processed_image = preprocess_image(image)
    
    # Debugging: Save the processed image
    processed_image_path = os.path.join(output_dir, f'processed_page_{page_idx + 1}.png')
    cv2.imwrite(processed_image_path, processed_image)
    print(f"Saved processed image for page {page_idx + 1} as {processed_image_path}")
    
    boards = detect_boards(processed_image)
    
    # Extract and save each board region
    for board_idx, board in enumerate(boards):
        y1, y2, x1, x2, w, h = board
        
        # Ensure board coordinates are within image bounds
        y1, y2 = max(y1, 0), min(y2, image.shape[0])
        x1, x2 = max(x1, 0), min(x2, image.shape[1])
        
        board_image = image[y1:y2, x1:x2]
        
        output_image_path = os.path.join(output_dir, f'page_{page_idx + 1}_board_{board_idx + 1}.jpg')
        cv2.imwrite(output_image_path, board_image)
        
        print(f"Saved page {page_idx + 1} board {board_idx + 1} as {output_image_path}")

# Process each page
for page_idx, page in enumerate(pages):
    page_cv = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    process_page(page_cv, page_idx)

print(f"All board images saved in {output_dir}")
