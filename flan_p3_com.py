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


def caliCameraAnd3DRecon(x, y, xp, yp):
    # 2D image points
    image_points = np.array([
        [252, 380],
        [84, 352],
        [92, 202],
        [259, 190],
        [407, 356],
        [294, 379],
        [300, 195],
        [411, 219]
    ], dtype=np.float32)

    # 3D object points
    object_points = np.array([
        [0, 1.5, 3.8],
        [0, 22.5, 3.8],
        [0, 22.5, 18.8],
        [0, 1.5, 18.8],
        [24.3, 0, 3.8],
        [3.3, 0, 3.8],
        [3.3, 0, 18.8],
        [24.3, 0, 18.8]
    ], dtype=np.float32)

    image_points2 = np.array([
        [258, 405],
        [105, 364],
        [114, 198],
        [268, 178],
        [474, 376],
        [313, 403],
        [322, 183],
        [480, 213]
    ], dtype=np.float32)
    image = cv2.imread("im1.jpeg")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create initial camera matrix
    camera_matrix = np.array([[500, 0, gray_image.shape[1] / 2],
                                    [0, 500, gray_image.shape[0] / 2],
                                    [0, 0, 1]], dtype=np.float32)
    
    # Distortion coefficients (assuming no distortion)
    distortion_coeffs = np.zeros((5, 1), dtype=np.float32)

    # Calibrate camera
    ret1, mtx1, distortion_coeffs1, R1, T1 = cv2.calibrateCamera([object_points], [image_points], (gray_image.shape[1], gray_image.shape[0]), camera_matrix, distortion_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_K3)
    ret2, mtx2, distortion_coeffs2, R2, T2 = cv2.calibrateCamera([object_points], [image_points2], (gray_image.shape[1], gray_image.shape[0]), camera_matrix, distortion_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_K3)
    print("Camera matrix 1:")
    print(mtx1)
    print("Camera matrix 2:")
    print(mtx2)
    print("\nDistortion coefficients:")
    print(distortion_coeffs)

    R1_t = cv2.Rodrigues(R1[0])[0]
    T1_t = T1[0]
    Rt_1 = np.concatenate([R1_t,T1_t], axis=-1) # [R|t]
    P1 = np.matmul(mtx1,Rt_1) # A[R|t]

    R2_t = cv2.Rodrigues(R2[0])[0]
    T2_t = T2[0]
    Rt_2 = np.concatenate([R2_t,T2_t], axis=-1) # [R|t]
    P2 = np.matmul(mtx2,Rt_2) # A[R|t]
    # Perform triangulation
    points1 = np.array([x, y])
    points2 = np.array([xp, yp])
    points_3d_homogeneous = cv2.triangulatePoints(P1, P2, points1, points2)
    print(f"3D homo -> {points_3d_homogeneous}")
    def DLT(P1, P2, point1, point2):
 
        A = [point1[1]*P1[2,:] - P1[1,:],
            P1[0,:] - point1[0]*P1[2,:],
            point2[1]*P2[2,:] - P2[1,:],
            P2[0,:] - point2[0]*P2[2,:]
            ]
        A = np.array(A).reshape((4,4))
        #print('A: ')
        #print(A)
    
        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices = False)
    
        print('Triangulated point: ')
        print(Vh[3,0:3]/Vh[3,3])
        return Vh[3,0:3]/Vh[3,3]
    p3d = DLT(P1, P2, points1, points2)
    print(f"DLT -> {p3d}")
    # Convert homogeneous coordinates to 3D Cartesian coordinates
    # points_3d_cartesian = cv2.convertPointsFromHomogeneous(points_3d_homogeneous.T)
    # print(f"Points 3D Cartesian -> {points_3d_cartesian}")
    # # Extract the 3D coordinates
    # x_3d = points_3d_cartesian[0, 0, 0]
    # y_3d = points_3d_cartesian[0, 0, 1]
    # z_3d = points_3d_cartesian[0, 0, 2]

    # print("3D Coordinates (X, Y, Z):")
    # print(x_3d, y_3d, z_3d)




def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        F, img1, img2, points1, points2 = param
        
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
        print(caliCameraAnd3DRecon(x, y, dst_point[0], dst_point[1]))
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
cv2.setMouseCallback('Image1', on_mouse_click, [F, img1, img2, points1, points2])

cv2.imshow('Image1', img1)
cv2.imshow('Image2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
