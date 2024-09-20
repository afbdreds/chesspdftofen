import cv2
from segment_boards import segment_boards

# Load the image
image_path = r'C:\Users\boldr\Downloads\chesspdftofen-master\chesspdfpage3.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Use cv2.IMREAD_GRAYSCALE if the image is grayscale

# Ensure image was loaded
if image is None:
    print("Error loading image. Please check the path.")
else:
    # Use the segment_boards function
    boards = segment_boards(image)
    
    # Load the image again in color to draw rectangles
    image_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Draw rectangles around detected boards
    for board in boards:
        y1, y2, x1, x2, w, h = board
        top_left = (x1, y1)
        bottom_right = (x2, y2)
        cv2.rectangle(image_color, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangles
        print("this is" , top_left,bottom_right)
    # Save the output image
    output_image_path = r'C:\Users\boldr\Downloads\chesspdftofen-master\output_with_boards.png'
    cv2.imwrite(output_image_path, image_color)
    
    # Optionally, display the image with rectangles
    cv2.imshow('Detected Boards', image_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Output image saved as {output_image_path}")
