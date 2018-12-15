import cv2
from train import process_files, train_svm
from detector import VehicleDetector, mean_shift, get_boxes, draw_windows, draw_centers
import matplotlib.pyplot as plt
import numpy as np

positive = "GTI_positive"
negative = "GTI_negative"


def train():
    feature_data = process_files(positive, negative, color_space="YCrCb", channels=[0, 1, 2], hog=True,
                                 histogram=True, spatial=True,
                                 hog_size=(64, 64), cell_size=(8, 8), cells_per_block=(2, 2),
                                 hog_bins=20, histogram_bins=16, spatial_size=(20, 20))
    train_svm(feature_data=feature_data, C=1000, output_file=True, output_filename="GTI_model.pkl")


def test():
    draw = True
    file_name = "1.jpg"
    thresholds = [100]
    minimum_nn = 5

    detector = VehicleDetector(window_size=(60, 60), x_overlap=0.8, y_step=0.01,
                        x_range=(0.1, 0.9), y_range=(0.4, 0.8), scale=1.1)
    detector.load_model(file_path="GTI_model.pkl")
    image = cv2.imread("data/" + file_name)
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
        plt.savefig('output/' + "min=" + str(minimum_nn) + "_threshold=" + str(threshold) + '_' + file_name)
        plt.gcf().clear()


if __name__ == "__main__":
    train()
    test()
