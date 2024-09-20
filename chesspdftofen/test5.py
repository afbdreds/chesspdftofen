import cv2
import numpy as np
import os
from segment_boards import segment_boards
from pdf2image import convert_from_path

# Define paths
pdf_path = r'C:\Users\boldr\Downloads\chesspdftofen-master\chesspdfpage.pdf'
output_dir = r'C:\Users\boldr\Downloads\chesspdftofen-master\cropped_boards'
dpi = 300  # Increased DPI for better quality

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Convert PDF to images with higher quality
pages = convert_from_path(pdf_path, dpi=dpi)

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to enhance contrast
    equalized = cv2.equalizeHist(gray)
    
    # Apply adaptive thresholding to highlight edges
    thresholded = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
    
    # Optional: Apply dilation to fill gaps in edges
    dilated = cv2.dilate(thresholded, None, iterations=2)
    
    return dilated

def detect_boards(image):
    # Apply detection on the processed image
    return segment_boards(image)

# Process each page
for page_idx, page in enumerate(pages):
    # Convert PIL image to OpenCV format
    page_cv = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    
    # Pre-process the image
    processed_image = preprocess_image(page_cv)
    
    # Debugging: Save the processed image to review the preprocessing results
    processed_image_path = os.path.join(output_dir, f'processed_page_{page_idx + 1}.png')
    cv2.imwrite(processed_image_path, processed_image)
    print(f"Saved processed image for page {page_idx + 1} as {processed_image_path}")
    
    # Detect boards
    boards = detect_boards(processed_image)
    
    # Extract and save each board region
    for board_idx, board in enumerate(boards):
        y1, y2, x1, x2, w, h = board
        
        board_image = page_cv[y1:y2, x1:x2]  # Crop the board region
        
        # Save the cropped board image as a JPEG file
        output_image_path = os.path.join(output_dir, f'page_{page_idx + 1}_board_{board_idx + 1}.jpg')
        cv2.imwrite(output_image_path, board_image)
        
        print(f"Saved page {page_idx + 1} board {board_idx + 1} as {output_image_path}")

print(f"All board images saved in {output_dir}")
