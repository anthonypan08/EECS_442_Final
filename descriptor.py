import cv2
import numpy as np
from skimage import feature


class Descriptor:
    def __init__(self, hog_features=False, hist_features=False, spatial_features=False, window_size=(64, 64),
                 hog_bins=9, cell_size=(8, 8), cells_per_block=(2, 2), hist_bins=16, spatial_size=(16, 16)):
        self.window_size = window_size
        self.hog_features = hog_features
        self.hist_features = hist_features
        self.spatial_features = spatial_features
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins

        block_size = (cells_per_block[0] * cell_size[0], cells_per_block[1] * cell_size[1])
        block_stride = (int(block_size[0] / 2), int(block_size[1] / 2))

        self.HOGDescriptor = cv2.HOGDescriptor(window_size, block_size, block_stride, cell_size, hog_bins)

    def get_features(self, image):
        if image.shape[:2] != self.window_size:
            image = cv2.resize(image, self.window_size, interpolation=cv2.INTER_AREA)

        feature_vector = np.array([])

        if self.hog_features:
            feature_vector = np.hstack((feature_vector, self.HOGDescriptor.compute(image)[:, 0]))

        if self.hist_features:
            hist_vector = np.array([])
            for channel in range(image.shape[2]):
                channel_hist = np.histogram(image[:, :, channel], bins=self.hist_bins, range=(0, 255))[0]
                hist_vector = np.hstack((hist_vector, channel_hist))
            feature_vector = np.hstack((feature_vector, hist_vector))

        if self.spatial_features:
            spatial_image = cv2.resize(image, self.spatial_size, interpolation=cv2.INTER_AREA)
            spatial_vector = spatial_image.ravel()
            feature_vector = np.hstack((feature_vector, spatial_vector))
        
        return feature_vector
