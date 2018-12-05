import cv2
from train import process_files, train_svm
from detector import Detector, mean_shift, get_boxes, draw_windows, draw_centers
import matplotlib.pyplot as plt
import numpy as np

pos_dir = "GTI_positive"
neg_dir = "GTI_negative"
video_file = "videos/test_video.mp4"


def train():
    feature_data = process_files(pos_dir, neg_dir, color_space="YCrCb", channels=[0, 1, 2], hog_features=True,
                                 hist_features=True, spatial_features=True,
                                 size=(64, 64), pix_per_cell=(8, 8), cells_per_block=(2, 2),
                                 hog_bins=20, hist_bins=16, spatial_size=(20, 20))

    train_svm(feature_data=feature_data, C=1000, output_file=True, output_filename="test_classifier.pkl")


def test():
    draw = True
    filename = "4.jpg"
    thresholds = [100]
    minimum_nn = 5

    detector = Detector(init_size=(50, 50), x_overlap=0.8, y_step=0.01,
                        x_range=(0.3, 0.75), y_range=(0.4, 0.8), scale=1.1)
    detector.load_classifier(file_path="test_classifier.pkl")
    image = cv2.imread("data/" + filename)
    windows = detector.classify(image)
    
    if draw:
        draw_windows(image, windows)
        draw_centers(image, windows)
        plt.show()
        plt.savefig('output/' + 'before_merge.jpg')

    for threshold in thresholds:
        boxes = get_boxes(windows=windows, cluster_centers=mean_shift(windows),
                          threshold=threshold, minimum_nn=minimum_nn)
        draw_windows(image, boxes)
        plt.savefig('output/' + "min=" + str(minimum_nn) + "_threshold=" + str(threshold) + '_' + filename)
        plt.gcf().clear()


if __name__ == "__main__":
    train()
    test()
