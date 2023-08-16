import cv2
import numpy as np

# Function to calculate the fundamental matrix
def calculate_fundamental_matrix(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Feature detection and matching using SIFT and FLANN
    detector = cv2.SIFT_create()
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)


    # Extract matching points

    points1 = []

    points2 = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            points2.append(keypoints2[m.trainIdx].pt)
            points1.append(keypoints1[m.queryIdx].pt)

    points1 = np.float32(points1)
    points2 = np.float32(points1)

    # Calculate fundamental matrix
    F, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 3, 0.99)

    return F

# Function to draw epipolar line and corresponding point on the right image
def draw_epipolar_line(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw the epipolar line on the left image
        cv2.circle(left_image, (x, y), 5, (0, 0, 255), -1)
        epipolar_line = np.dot(F, np.array([[x], [y], [1]]))
        epipolar_line /= np.linalg.norm(epipolar_line[:2])
        a = epipolar_line[0]
        b = epipolar_line[1]
        c = epipolar_line[2]
        x0 = 0
        x1 = left_image.shape[1]
        y0 = int((-c - a * x0) / b)
        y1 = int((-c - a * x1) / b)
        cv2.line(left_image, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Search for corresponding point on the right image using ZNCC
        search_range = 20  # Adjust the range as needed
        best_score = -1
        best_match = None

        for dx in range(-search_range, search_range + 1):
            xp = x + dx
            if xp < 0 or xp >= right_image.shape[1]:
                continue
            yp = int((-c - a * xp) / b)
            if yp < 0 or yp >= right_image.shape[0]:
                continue

            patch_left = gray_left[y - 5: y + 5, x - 5: x + 5]
            patch_right = gray_right[yp - 5: yp + 5, xp - 5: xp + 5]

            # Calculate ZNCC score
            mean_left = np.mean(patch_left)
            mean_right = np.mean(patch_right)
            std_left = np.std(patch_left)
            std_right = np.std(patch_right)
            correlation = np.sum((patch_left - mean_left) * (patch_right - mean_right))
            zncc_score = correlation / (10 * std_left * std_right)

            if zncc_score > best_score:
                best_score = zncc_score
                best_match = (xp, yp)
                print(best_match)

        # Draw the corresponding point on the right image
        if best_match is not None:
            cv2.circle(right_image, best_match, 5, (0, 0, 255), -1)

        # Display the updated images
        cv2.imshow('Left Image', left_image)
        cv2.imshow('Right Image', right_image)

# Read the two images
left_image = cv2.imread('left_image.jpeg')
right_image = cv2.imread('right_image.jpeg')

# Calculate the fundamental matrix
F = calculate_fundamental_matrix(left_image, right_image)

# Convert the images to grayscale
gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# Create windows and set the mouse callback function
cv2.namedWindow('Left Image')
cv2.namedWindow('Right Image')
cv2.setMouseCallback('Left Image', draw_epipolar_line)

# Display the images
cv2.imshow('Left Image', left_image)
cv2.imshow('Right Image', right_image)

# Wait for the user to close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
