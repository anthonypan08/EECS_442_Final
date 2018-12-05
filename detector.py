from datetime import datetime
import os
import pickle
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from descriptor import Descriptor
from sklearn.cluster import MeanShift
from scipy.spatial.distance import euclidean


class Detector:
    def __init__(self, init_size, x_overlap, y_step, x_range, y_range, scale):
        self.init_size = init_size
        self.x_overlap = x_overlap
        self.y_step = y_step
        self.x_range = x_range
        self.y_range = y_range
        self.scale = scale
        self.windows = None

    def load_classifier(self, file_path):
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError("File " + file_path + " does not exist.")
        classifier_data = pickle.load(open(file_path, "rb"))

        self.classifier = classifier_data["classifier"]
        self.scaler = classifier_data["scaler"]
        self.cv_color_const = classifier_data["cv_color_const"]
        self.channels = classifier_data["channels"]
        self.descriptor = Descriptor(
            hog_features=classifier_data["hog_features"],
            hist_features=classifier_data["hist_features"],
            spatial_features=classifier_data["spatial_features"],
            window_size=classifier_data["size"],
            hog_bins=classifier_data["hog_bins"],
            cell_size=classifier_data["cell_size"],
            cells_per_block=classifier_data["cells_per_block"],
            hist_bins=classifier_data["hist_bins"],
            spatial_size=classifier_data["spatial_size"]
        )

        return self

    def classify(self, image):
        self.windows = sliding_window((image.shape[1], image.shape[0]), init_size=self.init_size,
                                      x_overlap=self.x_overlap, y_step=self.y_step, x_range=self.x_range,
                                      y_range=self.y_range, scale=self.scale)

        if self.cv_color_const > -1:
            image = cv2.cvtColor(image, self.cv_color_const)

        feature_vectors = [self.descriptor.get_features(
                image[y_upper:y_lower, x_upper:x_lower, :])
            for (x_upper, y_upper, x_lower, y_lower) in self.windows]

        # Scale feature vectors, predict, and return predictions.
        feature_vectors = self.scaler.transform(feature_vectors)

        predictions = self.classifier.predict(feature_vectors)
        return [self.windows[ind] for ind in np.argwhere(predictions == 1)[:, 0]]


def sliding_window(image_size, init_size, x_overlap, y_step, x_range, y_range, scale):
    windows = []
    height, width = image_size[1], image_size[0]
    for y in range(int(y_range[0] * height), int(y_range[1] * height), int(y_step * height)):
        win_width = int(init_size[0] + (scale * (y - (y_range[0] * height))))
        win_height = int(init_size[1] + (scale * (y - (y_range[0] * height))))
        if y + win_height > int(y_range[1] * height) or win_width > width:
            break
        x_step = int((1 - x_overlap) * win_width)
        for x in range(int(x_range[0] * width), int(x_range[1] * width), x_step):
            windows.append((x, y, x + win_width, y + win_height))

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
