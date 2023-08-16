import cv2
import numpy as np

# Load the stereo images
left_img = cv2.imread('im1.jpeg', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('im2.jpeg', cv2.IMREAD_GRAYSCALE)

# Define the number of inner corners of the chessboard (for example, an 8x8 chessboard has 7x7 inner corners)
pattern_size = (7, 5)

# Find the chessboard corners in both images
ret_left, corners_left = cv2.findChessboardCorners(left_img, pattern_size)
ret_right, corners_right = cv2.findChessboardCorners(right_img, pattern_size)

if ret_left and ret_right:
    # Refine the detected corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(left_img, corners_left, (11, 11), (-1, -1), criteria)
    cv2.cornerSubPix(right_img, corners_right, (11, 11), (-1, -1), criteria)

    # These are your 2D points in each image
    points_2d_left = corners_left
    points_2d_right = corners_right
else:
    print("Could not find chessboard corners in one or both images.")
