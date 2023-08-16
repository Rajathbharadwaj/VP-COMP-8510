import cv2
import numpy as np
import sys

# Function to calculate the fundamental matrix
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

# Function to draw epipolar line and corresponding point on the right image
def draw_epipolar_line(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw the epipolar line on the left image
        cv2.circle(param[1], (x, y), 5, (0, 0, 255), -1)
        epipolar_line = np.dot(param[4], np.array([[x], [y], [1]]))
        epipolar_line /= np.linalg.norm(epipolar_line[:2])
        a = epipolar_line[0]
        b = epipolar_line[1]
        c = epipolar_line[2]
        x0 = 0
        x1 = param[0].shape[1]
        y0 = int((-c - a * x0) / b)
        y1 = int((-c - a * x1) / b)
        cv2.line(param[1], (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Search for corresponding point on the right image using ZNCC
        search_range = 120  # Adjust the range as needed
        best_score = -1
        best_match = None

        for dx in range(-search_range, search_range + 1):
            xp = x + dx
            if xp < 0 or xp >= param[1].shape[1]:
                continue
            yp = int((-c - a * xp) / b)
            if yp < 0 or yp >= param[1].shape[0]:
                continue

            patch_left = param[2][y - 5: y + 5, x - 5: x + 5]
            patch_right = param[3][yp - 5: yp + 5, xp - 5: xp + 5]
            print(f'patches {patch_left}, {patch_left}')

            # Calculate ZNCC score
            mean_left = np.mean(patch_left)
            mean_right = np.mean(patch_right)
            std_left = np.std(patch_left)
            std_right = np.std(patch_right)
            correlation = np.sum((patch_left - mean_left) * (patch_right - mean_right))
            print(correlation)
            zncc_score = correlation / (10 * std_left * std_right)
            print(zncc_score)

            if zncc_score > best_score:
                best_score = zncc_score
                best_match = (xp, yp)
                print(best_match)

        # Draw the corresponding point on the right image
        if best_match is not None:
            cv2.circle(param[1], best_match, 5, (0, 0, 255), -1)

        # Display the updated images
        cv2.imshow('Left Image', param[0])
        cv2.imshow('Right Image', param[1])

# Read the two images
def calPM(path1, path2):
    left_image = cv2.imread(path1)
    right_image = cv2.imread(path2)

    # Calculate the fundamental matrix
    F = calculate_fundamental_matrix(left_image, right_image)

    # Convert the images to grayscale
    gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # Create windows and set the mouse callback function
    cv2.namedWindow('Left Image')
    cv2.namedWindow('Right Image')
    cv2.setMouseCallback('Left Image', draw_epipolar_line, [left_image, right_image, gray_left, gray_right, F])

    # Display the images
    cv2.imshow('Left Image', left_image)
    cv2.imshow('Right Image', right_image)

    # Wait for the user to close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

calPM(sys.argv[1], sys.argv[2])