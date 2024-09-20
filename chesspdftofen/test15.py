import cv2
import numpy as np
import os
from segment_boards import segment_boards
from pdf2image import convert_from_path

def process_pdf(pdf_path):
    # Extract base name for the output directory
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(os.path.dirname(pdf_path), base_name)
    dpi = 600  # DPI for good quality

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert PDF to images with high quality
    pages = convert_from_path(pdf_path, dpi=dpi)

    def preprocess_image(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        thresholded = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(thresholded, kernel, iterations=2)
        return dilated

    def detect_boards(image):
        return segment_boards(image)

    def process_page(image, page_idx):
        print(f"Processing page {page_idx + 1}")
        height, width = image.shape[:2]
        regions = [(0, height, 0, width)]  # Add your specific regions here

        seen_boards = set()

        for region_idx, (y1, y2, x1, x2) in enumerate(regions):
            region_image = image[y1:y2, x1:x2]
            processed_image = preprocess_image(region_image)
            region_image_path = os.path.join(output_dir, f'processed_page_{page_idx + 1}_region_{region_idx + 1}.png')
            cv2.imwrite(region_image_path, processed_image)
            print(f"Saved processed image for page {page_idx + 1}, region {region_idx + 1} as {region_image_path}")

            boards = detect_boards(processed_image)

            for board_idx, board in enumerate(boards):
                board_y1, board_y2, board_x1, board_x2, w, h = board
                board_image = image[board_y1:board_y2, board_x1:board_x2]
                output_image_path = os.path.join(output_dir, f'page_{page_idx + 1}_board_{board_idx + 1}.jpg')
                cv2.imwrite(output_image_path, board_image)
                print(f"Saved page {page_idx + 1}, board {board_idx + 1} as {output_image_path}")

    for page_idx, page in enumerate(pages):
        page_cv = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        process_page(page_cv, page_idx)

    print(f"All board images saved in {output_dir}")

# If this file is executed directly, you could call the function here for testing.
if __name__ == "__main__":
    process_pdf(r'C:\Users\boldr\Downloads\chesspdftofen-master\chesspdfpage.pdf')
