import cv2
import os
from segment_boards import segment_boards
from pdf2image import convert_from_path

# Define paths
pdf_path = r'C:\Users\boldr\Downloads\chesspdftofen-master\chesspdfpage.pdf'
output_dir = r'C:\Users\boldr\Downloads\chesspdftofen-master\cropped_boards'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Convert PDF to images
pages = convert_from_path(pdf_path, dpi=300)  # You can adjust the DPI for quality

# Process each page
for page_idx, page in enumerate(pages):
    # Convert PIL image to OpenCV format
    page_cv = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

    # Convert page to grayscale for segmentation
    gray_page = cv2.cvtColor(page_cv, cv2.COLOR_BGR2GRAY)

    # Detect boards
    boards = segment_boards(gray_page)
    
    # Extract and save each board region
    for board_idx, board in enumerate(boards):
        y1, y2, x1, x2, w, h = board
        board_image = page_cv[y1:y2, x1:x2]  # Crop the board region
        
        # Save the cropped board image as a JPEG file
        output_image_path = os.path.join(output_dir, f'page_{page_idx + 1}_board_{board_idx + 1}.jpg')
        cv2.imwrite(output_image_path, board_image)
        
        print(f"Saved page {page_idx + 1} board {board_idx + 1} as {output_image_path}")

print(f"All board images saved in {output_dir}")
