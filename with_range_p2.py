import cv2
import numpy as np

def compute_ZNCC(window1, window2):
    mean1 = np.mean(window1)
    mean2 = np.mean(window2)

    numerator = np.sum((window1 - mean1) * (window2 - mean2))
    denominator = np.sqrt(np.sum((window1 - mean1)**2) * np.sum((window2 - mean2)**2))

    return numerator / denominator

def find_best_match(src_point, F, img1, img2, window_size=5):
    # Compute the epipolar line parameters: ax + by + c = 0
    line = np.dot(F, np.array([src_point[0], src_point[1], 1]))

    a, b, c = line

    # Define range to search around the pixel
    range_search = 30
    best_score = float('-inf')
    best_point = None

    # Walk on the epipolar line
    for x in range(src_point[0] - range_search, src_point[0] + range_search):
        y = int(- (a*x + c) / b)
        if 0 <= y < img2.shape[0]:
            window_img1 = img1[src_point[1]-window_size:src_point[1]+window_size+1, src_point[0]-window_size:src_point[0]+window_size+1]
            window_img2 = img2[y-window_size:y+window_size+1, x-window_size:x+window_size+1]
            
            if window_img1.shape == window_img2.shape: 
                score = compute_ZNCC(window_img1, window_img2)
                
                if score > best_score:
                    best_score = score
                    best_point = (x, y)

    return best_point

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        F, img1, img2 = param

        dst_point = find_best_match((x, y), F, img1, img2)

        cv2.circle(img1, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(img2, dst_point, 5, (0, 0, 255), -1)

        cv2.imshow('Image1', img1)
        cv2.imshow('Image2', img2)

# Load images and compute the fundamental matrix
img1 = cv2.imread('im1.jpeg', 0)
img2 = cv2.imread('im2.jpeg', 0)
keypoints1, descriptors1 = cv2.SIFT_create().detectAndCompute(img1, None)
keypoints2, descriptors2 = cv2.SIFT_create().detectAndCompute(img2, None)

matcher = cv2.BFMatcher()
matches = matcher.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

cv2.imshow('Image1', img1)
cv2.imshow('Image2', img2)
cv2.setMouseCallback('Image1', on_mouse_click, (F, img1, img2))

cv2.waitKey(0)
cv2.destroyAllWindows()
