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

def three_dimensional_reconstruction(camera_matrix1, camera_matrix2, p1, p2):
    A = np.zeros((4, 4))
    
    A[0] = p1[0] * camera_matrix1[2] - camera_matrix1[0]
    A[1] = p1[1] * camera_matrix1[2] - camera_matrix1[1]
    A[2] = p2[0] * camera_matrix2[2] - camera_matrix2[0]
    A[3] = p2[1] * camera_matrix2[2] - camera_matrix2[1]
    
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1, :3] / Vt[-1, 3]

    return X



import numpy as np

# def camera_calibration(image_points, world_points):
#     A = np.zeros((len(image_points) * 2, 12))
#     for i, (image, object) in enumerate(zip(image_points, world_points)):
#         X, Y, Z = map(float, object) # Convert to float
#         x, y = map(float, image)     # Convert to float
#         A[2 * i, :] = [-X, -Y, -Z, -1, 0, 0, 0, 0, x * X, x * Y, x * Z, x]
#         A[2 * i + 1, :] = [0, 0, 0, 0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y]

#     # Perform SVD decomposition
#     _, _, V = np.linalg.svd(A)
    
#     # Get the last row of V and reshape it into a 3x4 matrix to get the projection matrix P
#     P = V[-1].reshape((3, 4))

#     # Compute the calibration error (reprojection error)
#     projection_error = 0.0
#     for i, (image, object) in enumerate(zip(image_points, world_points)):
#         X_values = [float(point) for point in object]
#         X_values.append(1)
#         X = np.array(X_values, dtype=float) # Ensure consistent data type
#         x = [float(point) for point in image]
#         x.append(1)

#         projected_x = np.dot(P, X)
#         projected_x /= projected_x[-1]  # Normalizing
#         projection_error += np.linalg.norm(projected_x[:-1] - x[:-1])

#         mean_error = projection_error / len(image_points)
#         print("Mean reprojection error: {}".format(mean_error))
#         print(f"Projection Matrix: \n{P}")
    
#         return P




def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        F, img1, img2, M1, M2 = param
        
        dst_point = find_best_match((x, y), F, img1, img2)
        cv2.drawMarker(img1, (x, y), (255, 255, 59), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        cv2.drawMarker(img2, dst_point, (255, 255, 59), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

        # cv2.circle(img1, (x, y), 5, (0, 255, 0), -1)
        # cv2.circle(img2, dst_point, 5, (0, 0, 255), -1)

        # Draw the epipolar line on the right image
        line = cv2.computeCorrespondEpilines(np.array([[x, y]]).reshape(-1, 1, 2), 1, F).reshape(-1,)
        color = (255, 0, 0)  # Blue color for the epipolar line
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [img2.shape[1], -(line[2] + line[0] * img2.shape[1]) / line[1]])
        cv2.line(img2, (x0, y0), (x1, y1), color, 1)
        print(f"The x` y` is -> {dst_point}")
        print('3D Reconstruction: {}'.format(three_dimensional_reconstruction(M1, M2, (x, y), dst_point)))
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

with open('im1_points.txt', 'r') as f:
    points_2d_L = f.readlines()

with open('im2_points.txt', 'r') as f:
    points_2D_R = f.readlines()

with open('3DCoordinates.txt', 'r') as f:
    points_3d = f.readlines()


points_2d_L = [point.split() for point in points_2d_L]  # 2D image points
points_2D_R = [point.split() for point in points_2D_R]  # 2D image points
points_3d = [point.split() for point in points_3d]  # 3D world points

# M1 = camera_calibration(points_2d_L, points_3d)
# M2 = camera_calibration(points_2d_L, points_3d)
ret, M1, dist, rvecs, tvecs = cv2.calibrateCamera(points_3d, points_2d_L, img1.shape[::-1], None, None)
ret, M2, dist, rvecs, tvecs = cv2.calibrateCamera(points_3d, points_2D_R, img2.shape[::-1], None, None)


F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)


cv2.namedWindow('Image1')
cv2.setMouseCallback('Image1', on_mouse_click, [F, img1, img2, M1, M2])

cv2.imshow('Image1', img1)
cv2.imshow('Image2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
