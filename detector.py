import os
import pickle
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from descriptor import Descriptor
from sklearn.cluster import MeanShift
from scipy.spatial.distance import euclidean


class VehicleDetector:
    def __init__(self, window_size, x_overlap, y_step, x_range, y_range, scale):
        self.window_size = window_size
        self.x_overlap = x_overlap
        self.y_step = y_step
        self.x_range = x_range
        self.y_range = y_range
        self.scale = scale
        self.windows = None

    def load_model(self, file_path):
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError("File " + file_path + " not found.")

        model_data = pickle.load(open(file_path, "rb"))

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.color_const = model_data["color_const"]
        self.channels = model_data["channels"]
        self.descriptor = Descriptor(
            hog=model_data["hog"],
            histogram=model_data["histogram"],
            spatial=model_data["spatial"],
            hog_size=model_data["hog_size"],
            hog_bins=model_data["hog_bins"],
            cell_size=model_data["cell_size"],
            cells_per_block=model_data["cells_per_block"],
            histogram_bins=model_data["histogram_bins"],
            spatial_size=model_data["spatial_size"]
        )

        return self

    def classify(self, image):
        self.windows = sliding_window((image.shape[1], image.shape[0]), window_size=self.window_size,
                                      x_overlap=self.x_overlap, y_step=self.y_step, x_range=self.x_range,
                                      y_range=self.y_range, scale=self.scale)

        if self.color_const > -1:
            image = cv2.cvtColor(image, self.color_const)

        feature_matrix = [self.descriptor.get_features(image[y_upper:y_lower, x_upper:x_lower, :])
                           for (x_upper, y_upper, x_lower, y_lower) in self.windows]
        feature_matrix = self.scaler.transform(feature_matrix)

        predictions = self.model.predict(feature_matrix)

        result_windows = []
        for count, item in enumerate(predictions):
            if item == 1:
                result_windows.append(self.windows[count])

        return result_windows


def sliding_window(img_size, window_size, x_overlap, y_step, x_range, y_range, scale):
    windows = []
    height, width = img_size[1], img_size[0]
    for y in range(int(y_range[0] * height), int(y_range[1] * height), int(y_step * height)):
        window_width = int(window_size[0] + (scale * (y - (y_range[0] * height))))
        window_height = int(window_size[1] + (scale * (y - (y_range[0] * height))))
        if y + window_height > int(y_range[1] * height) or window_width > width:
            break
        x_step = int((1 - x_overlap) * window_width)
        for x in range(int(x_range[0] * width), int(x_range[1] * width), x_step):
            windows.append((x, y, x + window_width, y + window_height))

    return windows


def mean_shift(windows):
    centers = np.array([[int((window[0] + window[2]) / 2), int((window[1] + window[3]) / 2)] for window in windows])
    ms = MeanShift()
    ms.fit(centers)

    return ms.cluster_centers_


def get_box(windows, cluster_center, threshold=100, minimum_nn=3):
    centers = [((int((window[0] + window[2]) / 2), int((window[1] + window[3]) / 2)), window) for window in windows]
    distances = [(euclidean(center[0], cluster_center), center[1]) for center in centers]

    x_min = []
    y_min = []
    x_max = []
    y_max = []

    for distance in distances:
        if distance[0] < float(threshold):
            x_min.append(distance[1][0])
            x_max.append(distance[1][2])
            y_min.append(distance[1][1])
            y_max.append(distance[1][3])

    if len(x_min) < minimum_nn:
        return None

    return min(x_min), min(y_min), max(x_max), max(y_max)


def get_boxes(windows, cluster_centers, threshold=100, minimum_nn=3):
    boxes = []
    for cluster_center in cluster_centers:
        box = get_box(windows, cluster_center, threshold, minimum_nn)
        if box is not None:
            boxes.append(box)

    return boxes


def draw_centers(image, windows):
    centers = [(int((window[0] + window[2]) / 2), int((window[1] + window[3]) / 2)) for window in windows]
    for center in centers:
        cv2.circle(image, (int(center[0]), int(center[1])), 3, (255, 0, 0), 3)
        plt.imshow(np.array(image).astype('uint8'))


def draw_windows(image, windows):
    red = 255
    for window in windows:
        cv2.rectangle(image, (window[0], window[1]), (window[2], window[3]), (red, 255-red, 0), 2)
        plt.imshow(np.array(image).astype('uint8'))
        red -= 2
