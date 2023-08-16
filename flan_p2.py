import numpy as np
import cv2

def zncc(window1, window2):
    mean1 = np.mean(window1)
    mean2 = np.mean(window2)
    
    numerator = np.sum((window1 - mean1) * (window2 - mean2))
    denominator = np.sqrt(np.sum((window1 - mean1) ** 2) * np.sum((window2 - mean2) ** 2))
    
    if denominator == 0:
        return 0
    return numerator / denominator

def find_best_match(src_point, F, img1, img2, window_size=5):
    src_point = np.array([src_point], dtype=np.float32)
    line = cv2.computeCorrespondEpilines(src_point, 1, F)[0].ravel()
    
    height, width = img2.shape
    
    best_score = -1
    best_point = None
    for x in range(width):
        y = int(-(line[2] + line[0] * x) / line[1])
        if 0 <= y < height:
            y_int = int(src_point[0][1])
            x_int = int(src_point[0][0])

            window_centered_at_p = img1[y_int-window_size:y_int+window_size+1, 
                            x_int-window_size:x_int+window_size+1]
            window_centered_at_q = img2[y-window_size:y+window_size+1, x-window_size:x+window_size+1]
            
            if window_centered_at_p.shape == window_centered_at_q.shape:
                score = zncc(window_centered_at_p, window_centered_at_q)
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

        # Draw the epipolar line on the right image
        line = cv2.computeCorrespondEpilines(np.array([[x, y]]).reshape(-1, 1, 2), 1, F).reshape(-1,)
        color = (255, 0, 0)  # Blue color for the epipolar line
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [img2.shape[1], -(line[2] + line[0] * img2.shape[1]) / line[1]])
        cv2.line(img2, (x0, y0), (x1, y1), color, 1)
        
        cv2.imshow('Image1', img1)
        cv2.imshow('Image2', img2)


img1 = cv2.imread('im1.jpeg', 0)
img2 = cv2.imread('im2.jpeg', 0)

keypoints1, descriptors1 = cv2.SIFT_create().detectAndCompute(img1, None)
keypoints2, descriptors2 = cv2.SIFT_create().detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

points1 = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)

cv2.namedWindow('Image1')
cv2.setMouseCallback('Image1', on_mouse_click, [F, img1, img2])

cv2.imshow('Image1', img1)
cv2.imshow('Image2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
