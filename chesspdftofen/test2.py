# test.py

import cv2
from segment_boards import segment_boards

# Load the image
image_path = r'C:\Users\boldr\Downloads\chesspdftofen-master\chesspdfpage.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Use cv2.IMREAD_GRAYSCALE if the image is grayscale

# Ensure image was loaded
if image is None:
    print("Error loading image. Please check the path.")
else:
    # Use the segment_boards function
    boards = segment_boards(image)
    
    # Print the results or process further
    print("Detected boards:")
    for board in boards:
        print(board)
    
    # Optionally, display the result
    for board in boards:
        x, y, w, h = board[2], board[0], board[4], board[5]
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the image with rectangles
    cv2.imshow('Detected Boards', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
