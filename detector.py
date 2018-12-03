import cv2
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import AdaBoostClassifier
from descriptor import Descriptor


class Detector:
    def __init__(self, model_path, file_path):
        self.model = pickle.load(open(model_path, 'rb'))

        if not os.path.isfile(file_path):
            raise FileNotFoundError("File " + file_path + " not found.")
        data = pickle.load(open(file_path, "rb"))
        self.scaler = data["scaler"]
        self.descriptor = Descriptor(
            hog_features=data["hog_features"],
            size=data["size"],
            hog_bins=data["hog_bins"],
            pix_per_cell=data["pix_per_cell"],
            cells_per_block=data["cells_per_block"],
            block_stride=data["block_stride"],
            block_norm=data["block_norm"],
            transform_sqrt=data["transform_sqrt"],
            signed_gradient=data["signed_gradient"]
        )

    def detect_windows(self, img):
        windows = sliding_window((img.shape[1], img.shape[0]))
        feature_vectors = [self.descriptor.getFeatureVector(img[y_upper:y_lower, x_upper:x_lower, :])
                           for (x_upper, y_upper, x_lower, y_lower) in windows]

        feature_vectors = self.scaler.transform(feature_vectors)
        predictions = self.model.predict(feature_vectors)

        return [windows[i[0]] for i in np.argwhere(predictions)]


def draw_windows(img, windows):
    for window in windows:
        cv2.rectangle(img, (window[0], window[1]), (window[2], window[3]), (255, 0, 0), 2)
        plt.imshow(np.array(img).astype('uint8'))
    plt.show()


def draw_centers(img, windows):
    for window in windows:
        cv2.circle(img, (int(window[0] + window[2] / 2), int(window[1] + window[3] / 2)), 3, (255, 0, 0), 3)
        plt.imshow(np.array(img).astype('uint8'))
    plt.show()


def sliding_window(img_size, init_size=(256, 256), x_step=0.05, y_step=0.05, scale=1.1):
    windows = []
    width, height = img_size[0], img_size[1]
    window_width, window_height = init_size[0], init_size[1]
    for y in range(0, height, int(y_step * height)):
        init_size = (init_size[0] * scale, init_size[1] * scale)
        if y + window_height > height:
            break
        for x in range(0, width, int(x_step * width)):
            if x + window_width > width:
                break
            windows.append((x, y, x + window_width, y + window_height))

    return windows


def main():
    detector = Detector("temp_classifier.pkl", "temp_data.pkl")
    img = cv2.imread("data/MIT_no bike/SSDB03523.JPG")
    windows = detector.detect_windows(img)
    draw_centers(img, windows)


if __name__ == '__main__':
    main()
