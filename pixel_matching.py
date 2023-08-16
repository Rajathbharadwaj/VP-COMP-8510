import cv2
import numpy as np

def calculate_fundamental_matrix(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Feature detection and matching using ORB and BFMatcher
    detector = cv2.ORB_create()
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = matcher.match(descriptors1, descriptors2)

    # Extract matching points

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Calculate fundamental matrix
    F, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 3, 0.99)

    return F

# Zero-mean Normalized Cross-Correlation
def zncc(window1, window2):
    mean1, mean2 = np.mean(window1), np.mean(window2)
    window1, window2 = window1 - mean1, window2 - mean2
    numerator = np.sum(window1 * window2)
    denominator = np.sqrt(np.sum(window1**2) * np.sum(window2**2))
    return (numerator / denominator) if denominator != 0 else 0

# Mouse callback function
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw point on left image
        cv2.circle(left_image, (x, y), 5, (0, 255, 0), -1)
        epilines = cv2.computeCorrespondEpilines(np.array([[x, y]]), 1, F)
        epiline = epilines[0][0]

        # Draw epipolar line on right image
        x0, y0 = map(int, [0, -epiline[2] / epiline[1]])
        x1, y1 = map(int, [right_image.shape[1], -(epiline[2] + epiline[0] * right_image.shape[1]) / epiline[1]])
        cv2.line(right_image, (x0, y0), (x1, y1), (0, 255, 0), 1)

        # Find matching point on the epipolar line
        best_zncc, best_pt = -1, (0, 0)
        for x in range(max(0, x0), min(right_image.shape[1], x1)):
            for y in range(max(0, y0), min(right_image.shape[0], y1)):
                zncc_value = zncc(left_image[y-5:y+5, x-5:x+5], right_image[y-5:y+5, x-5:x+5])
                if zncc_value > best_zncc:
                    best_zncc, best_pt = zncc_value, (x, y)

        # Draw matching point on right image
        cv2.circle(right_image, best_pt, 5, (0, 0, 255), -1)

# Load images
left_image = cv2.imread('left1_image.jpeg')
right_image = cv2.imread('right1_image.jpeg')

# Compute fundamental matrix F
F = calculate_fundamental_matrix(left_image, right_image)

# Convert the images to grayscale
gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# Create windows and set the mouse callback function
cv2.namedWindow('Left Image')
cv2.namedWindow('Right Image')

# Set mouse callback
cv2.setMouseCallback('Left Image', on_mouse_click)

# Display images
while True:
    cv2.imshow('Left Image', left_image)
    cv2.imshow('Right Image', right_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()