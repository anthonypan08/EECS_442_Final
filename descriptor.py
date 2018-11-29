import cv2
import numpy as np
from skimage import feature

'''
Any time we want to obtain a feature vector from an image, 
we can simply pass it to the object’s getFeatureVector() method, 
which returns the feature vector.
'''

class Descriptor:
    '''
    This is a class that produce a feature vector for an input image.
    '''

    # input parameters for cv2
    def __init__(self, hog_features=False, color_hist_features=False,
                 spatial_features=False, hog_lib="cv", size=(64, 64),
                 hog_bins=9, pix_per_cell=(8, 8), cells_per_block=(2, 2),
                 block_stride=None, block_norm="L1", transform_sqrt=True,
                 signed_gradient=False, hist_bins=16, spatial_size=(16, 16)):

        """
        Set feature parameters. For HOG features, either the OpenCV
        implementation (cv2.HOGDescriptor) or scikit-image implementation
        (skimage.feature.hog) may be selected via @param hog_lib. Some
        parameters apply to only one implementation (indicated below).
        @param hog_features (bool): Include HOG features in feature vector.
        @param hist_features (bool): Include color channel histogram features
            in feature vector.
        @param spatial_features (bool): Include spatial features in feature vector.
        @param size (int, int): Resize images to this (width, height) before
            computing features.
        @param hog_lib ["cv", "sk"]: Select the library to be used for HOG
            implementation. "cv" selects OpenCV (@see cv2.HOGDescriptor).
            "sk" selects scikit-image (@see skimage.feature.hog).
        @param pix_per_cell (int, int): HOG pixels per cell.
        @param cells_per_block (int, int): HOG cells per block.
        @param block_stride (int, int): [OpenCV only] Number of pixels by which
            to shift block during HOG block normalization. Defaults to half of
            cells_per_block.
        @param block_norm: [scikit-image only] Block normalization method for
            HOG. OpenCV uses L2-Hys.
        @param transform_sqrt (bool): [scikit-image only].
            @see skimage.feature.hog
        @param hog_bins (int): Number of HOG gradient histogram bins.
        @param signed_gradient (bool): [OpenCV only] Use signed gradient (True)
            or unsigned gradient (False) for HOG. Currently, scikit-image HOG
            only supports unsigned gradients.
        @param hist_bins (int): Number of color histogram bins per color channel.
        @param spatial_size (int, int): Resize images to (width, height) for
            spatial binning.
        """


        '''
        block size is in pixels (not cells), 
        and the block stride for block normalization must be also in pixels.
        '''
        self.hog_features = hog_features
        self.color_hist_features = color_hist_features
        self.spatial_features = spatial_features
        self.size = size
        self.hog_lib = hog_lib
        self.pix_per_cell = pix_per_cell
        self.cells_per_block = cells_per_block

        window_size = size
        cell_size = pix_per_cell
        block_size = (cells_per_block[0] * cell_size[0],
                         cells_per_block[1] * cell_size[1])

        if block_stride is not None:
            stride_block = self.block_stride
        else:
            stride_block = (int(block_size[0] / 2), int(block_size[1] / 2))

        # parameters to be tuned
        nbins = hog_bins
        derivAperture = 1
        win_sigma = -1.
        histogram_norm = 0  # L2Hys (currently the only available option)
        L2HysThreshold = 0.2
        gamma_correct = 1
        levels = 64
        gradients_signed = signed_gradient

        self.HOGDescriptor = cv2.HOGDescriptor(window_size, block_size,
                                                   stride_block, cell_size, nbins, derivAperture, win_sigma,
                                                   histogram_norm, L2HysThreshold, gamma_correct,
                                                   levels, signed_gradient)

        self.hist_bins = hist_bins
        self.spatial_size = spatial_size

    def getFeatureVector(self, image):

        '''Return the feature vector for an image.'''

        if image.shape[:2] != self.size:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)

        feature_vector = np.array([])

        if self.hog_features:
            feature_vector = np.hstack(
                (feature_vector, self.HOGDescriptor.compute(image)[:, 0]))

        if self.color_hist_features:
            # np.histogram() returns a tuple if given a 2D array and an array
            # if given a 3D array. To maintain compatibility with other
            # functions in the object detection pipeline, check that the input
            # array has three dimensions. Add axis if necessary.
            # Note that histogram bin range assumes uint8 array.
            if len(image.shape) < 3:
                image = image[:, :, np.newaxis]

            '''
            added color features and spital features
            obtain color channel histogram features
            '''
            hist_vector = np.array([])
            for channel in range(image.shape[2]):
                channel_hist = np.histogram(image[:, :, channel],
                                            bins=self.hist_bins, range=(0, 255))[0]
                hist_vector = np.hstack((hist_vector, channel_hist))
            feature_vector = np.hstack((feature_vector, hist_vector))

        if self.spatial_features:
            spatial_image = cv2.resize(image, self.spatial_size,
                                       interpolation=cv2.INTER_AREA)
            # flattened
            spatial_vector = spatial_image.ravel()
            feature_vector = np.hstack((feature_vector, spatial_vector))

        return feature_vector