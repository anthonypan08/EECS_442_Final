import cv2
import numpy as np
from skimage import feature


class Descriptor:
    def __init__(self, hog=False, histogram=False, spatial=False, hog_size=(64, 64),
                 hog_bins=9, cell_size=(8, 8), cells_per_block=(2, 2), histogram_bins=16, spatial_size=(16, 16)):
        self.hog_size = hog_size
        self.hog = hog
        self.histogram = histogram
        self.spatial = spatial
        self.spatial_size = spatial_size
        self.histogram_bins = histogram_bins

        block_size = (cells_per_block[0] * cell_size[0], cells_per_block[1] * cell_size[1])
        block_stride = (int(block_size[0] / 2), int(block_size[1] / 2))

        self.HOGDescriptor = cv2.HOGDescriptor(hog_size, block_size, block_stride, cell_size, hog_bins)

    def get_features(self, image):
        if image.shape[:2] != self.hog_size:
            image = cv2.resize(image, self.hog_size, interpolation=cv2.INTER_AREA)

        feature_matrix = np.array([])

        if self.hog:
            feature_matrix = np.hstack((feature_matrix, self.HOGDescriptor.compute(image)[:, 0]))

        if self.histogram:
            hist_vector = np.array([])
            for channel in range(image.shape[2]):
                channel_hist = np.histogram(image[:, :, channel], bins=self.histogram_bins, range=(0, 255))[0]
                hist_vector = np.hstack((hist_vector, channel_hist))
            feature_matrix = np.hstack((feature_matrix, hist_vector))

        if self.spatial:
            spatial_image = cv2.resize(image, self.spatial_size, interpolation=cv2.INTER_AREA)
            spatial_vector = spatial_image.ravel()
            feature_matrix = np.hstack((feature_matrix, spatial_vector))
        
        return feature_matrix
