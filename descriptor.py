import cv2
import numpy as np
from skimage import feature


class Descriptor:

    """
    Class that combines feature descriptors into a single descriptor
    to produce a feature vector for an input image.
    """

    def __init__(self, hog_features=False, size=(64, 64),
                 hog_bins=9, pix_per_cell=(8, 8), cells_per_block=(2, 2),
                 block_stride=None, block_norm="L1", transform_sqrt=True,
                 signed_gradient=False):
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

        self.hog_features = hog_features
        self.size = size
        self.pix_per_cell = pix_per_cell
        self.cells_per_block = cells_per_block

        winSize = size
        cellSize = pix_per_cell
        blockSize = (cells_per_block[0] * cellSize[0],
                        cells_per_block[1] * cellSize[1])

        if block_stride is not None:
            blockStride = self.block_stride
        else:
            blockStride = (int(blockSize[0] / 2), int(blockSize[1] / 2))

            nbins = hog_bins
            derivAperture = 1
            winSigma = -1.
            # L2Hys (currently the only available option)
            histogramNormType = 0
            L2HysThreshold = 0.2
            gammaCorrection = 1
            nlevels = 64
            signedGradients = signed_gradient

            self.HOGDescriptor = cv2.HOGDescriptor(winSize, blockSize,
                                                   blockStride, cellSize, nbins, derivAperture, winSigma,
                                                   histogramNormType, L2HysThreshold, gammaCorrection,
                                                   nlevels, signedGradients)

    def getFeatureVector(self, image):
        """Return the feature vector for an image."""

        if image.shape[:2] != self.size:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)

        feature_vector = np.array([])

        if self.hog_features:
            feature_vector = np.hstack(
                (feature_vector, self.HOGDescriptor.compute(image)[:, 0]))
        return feature_vector
