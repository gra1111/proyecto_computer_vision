import os
import glob
import copy
import imageio
import numpy as np
from typing import List
import cv2
print("OpenCV should be 4.8.0.76 Current version:", cv2.__version__)


def show_image(img: np.array, img_name: str = "Image"):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_image(output_folder: str, img_name: str, img: np.array):
    os.makedirs(output_folder, exist_ok=True)
    img_path = os.path.join(output_folder, img_name)
    cv2.imwrite(img_path, img)


def load_images(filenames: List) -> List:
    return [imageio.imread(filename) for filename in filenames]


def get_chessboard_points(chessboard_shape, dx, dy):
    points = []
    for j in range(chessboard_shape[0]):
        for i in range(chessboard_shape[1]):
            x = i * dx
            y = j * dy
            z = 0.0
            points.append([x, y, z])

    # Convertimos la lista a array de float32
    return np.array(points, dtype=np.float32)
# Build a list containing the paths of all images for calibration


imgs_path = []
path = "../data/"
imgs_path = []
for j in range(10):
    imgs_path.append(path + 'calibration_0' + str(j) + '.jpg')
print(imgs_path)
imgs = load_images(imgs_path)
imgs_copy = [im.copy() for im in imgs]
# Find corners with cv2.findChessboardCorners()
corners = [cv2.findChessboardCorners(img, (7, 9)) for img in imgs]
corners_copy = copy.deepcopy(corners)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

# To refine corner detections with cv2.cornerSubPix() you need to input grayscale images. Build a list containing grayscale images.
imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

corners_refined = [cv2.cornerSubPix(i, cor[1], (9, 7), (-1, -1), criteria)
                   if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]

# Use cv2.drawChessboardCorners() to draw the cornes
for i in range(len(imgs_copy)):
    cv2.drawChessboardCorners(
        imgs_copy[i], (7, 9), corners[i][1], corners[i][0])

# save the images
output_folder = "../imagenes_con_marca"

for i in range(len(imgs_copy)):
    nombre_base = os.path.splitext(os.path.basename(imgs_path[i]))[0]
    nuevo_nombre = f"{nombre_base}_marked.jpg"

    write_image(output_folder, nuevo_nombre, imgs_copy[i])
    show_image(imgs_copy[i], nuevo_nombre)

# get chessboards points using our own function
chessboard_points = [get_chessboard_points((9, 7), 20, 20) for img in imgs]

# calibrate the camera now
# Filter data and get only those with adequate detections
valid_corners = [cor[1] for cor in corners if cor[0]]
# Convert list to numpy array
valid_corners = np.asarray(valid_corners, dtype=np.float32)
chesboard_points_valid = [get_chessboard_points(
    (9, 7), 20, 20) for _ in range(len(valid_corners))]
image_size = imgs_copy[0].shape[:2]
rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    chesboard_points_valid, valid_corners, image_size, None, None)

# Obtain extrinsics
extrinsics = list(map(lambda rvec, tvec: np.hstack(
    (cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

# Print outputs
print("Intrinsics:\n", intrinsics)
print("Distortion coefficients:\n", dist_coeffs)
print("Root mean squared reprojection error:\n", rms)

print("\nExtrinsics :")
for i, ext in enumerate(extrinsics):
    print(f"Vista {i:02d}:\n{ext}\n")
