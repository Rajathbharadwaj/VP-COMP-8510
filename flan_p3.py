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
    A = np.zeros((4, 3))

    A[0][0] = p1[0] * camera_matrix1[8] - camera_matrix1[0]
    A[0][1] = p1[0] * camera_matrix1[9] - camera_matrix1[1]
    A[0][2] = p1[0] * camera_matrix1[10] - camera_matrix1[2]

    A[1][0] = p1[1] * camera_matrix1[8] - camera_matrix1[4]
    A[1][1] = p1[1] * camera_matrix1[9] - camera_matrix1[5]
    A[1][2] = p1[1] * camera_matrix1[10] - camera_matrix1[6]

    A[2][0] = p2[0] * camera_matrix2[8] - camera_matrix2[1]
    A[2][1] = p2[0] * camera_matrix2[9] - camera_matrix2[2]
    A[2][2] = p2[0] * camera_matrix2[10] - camera_matrix2[3]

    A[3][0] = p2[0] * camera_matrix2[8] - camera_matrix2[4]
    A[3][1] = p2[0] * camera_matrix2[9] - camera_matrix2[5]
    A[3][2] = p2[0] * camera_matrix2[10] - camera_matrix2[6]

    d = np.zeros((1, 4))
    d[0][0] = p1[0] * camera_matrix1[11] - camera_matrix1[3]
    d[0][1] = p1[1] * camera_matrix1[11] - camera_matrix1[7]
    d[0][2] = p2[0] * camera_matrix2[11] - camera_matrix2[3]
    d[0][3] = p2[1] * camera_matrix2[11] - camera_matrix2[7]

    U, S, Vt = np.linalg.svd(A)
    # Calculate the pseudoinverse of S
    S_pseudo = np.zeros(A.shape).T
    S_pseudo[:S.shape[0], :S.shape[0]] = np.diag(1 / S)
    # Calculate the pseudoinverse of A using SVD
    A_pseudo = np.dot(np.dot(Vt.T, S_pseudo), U.T)

    # Solve for matrix X
    X = np.dot(A_pseudo, d.T)

    return X

def camera_calibration(image_points, world_points):
    A = np.zeros((len(image_points) * 2, 12))
    for i, (image, object) in enumerate(zip(image_points, world_points)):
        X, Y, Z = [float(point) for point in object]
        x, y = [float(point) for point in image]
        A[2 * i, :] = [-X, -Y, -Z, -1, 0, 0, 0, 0, x * X, x * Y, x * Z, x]
        A[2 * i + 1, :] = [0, 0, 0, 0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y]
    # Perform SVD decomposition
    _, _, V = np.linalg.svd(A)
    P = V[-1].reshape((3, 4))

    # Extract camera parameters from V
    camera_params = V[-1, :12]
    camera_params /= camera_params[-1]  # Normalize the parameters

    print(f'Camera parameters: \n {camera_params}')
    # Compute the calibration error
    projection_error = 0.0
    for i, (image, object) in enumerate(zip(image_points, world_points)):
        X = [float(point) for point in object]
        X.append(1)
        x = [float(point) for point in image]
        x.append(1)

        projected_x = np.dot(camera_params.reshape((3, 4)), X)
        projected_x /= projected_x[-1]  # Normalizing
        projection_error += np.linalg.norm(projected_x[:-1] - x[:-1])

    mean_error = projection_error / len(image_points)
    print("Mean reprojection error: {}".format(mean_error))
    print(f"camera_params")
    return camera_params

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

M1 = camera_calibration(points_2d_L, points_3d)
M2 = camera_calibration(points_2D_R, points_3d)
F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)


cv2.namedWindow('Image1')
cv2.setMouseCallback('Image1', on_mouse_click, [F, img1, img2, M1, M2])

cv2.imshow('Image1', img1)
cv2.imshow('Image2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
