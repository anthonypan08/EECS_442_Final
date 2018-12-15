from datetime import datetime
import os
import pickle
import random
import time
import warnings
import cv2
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from descriptor import Descriptor


def process_files(positive_dir, negative_dir, color_space="bgr", channels=[0, 1, 2], hog=False, histogram=False,
                  spatial=False, hog_size=(64, 64), hog_bins=9, cell_size=(8, 8), cells_per_block=(2, 2),
                  histogram_bins=16, spatial_size=(16, 16)):
    # take care of training files
    positive_dir = os.path.abspath(positive_dir)
    negative_dir = os.path.abspath(negative_dir)
    if not os.path.isdir(positive_dir):
        raise FileNotFoundError("Directory " + positive_dir + " not found.")
    if not os.path.isdir(negative_dir):
        raise FileNotFoundError("Directory " + negative_dir + " not found.")
    positive_files = [os.path.join(positive_dir, file) for file in os.listdir(positive_dir)
                      if os.path.isfile(os.path.join(positive_dir, file))]
    negative_files = [os.path.join(negative_dir, file) for file in os.listdir(negative_dir)
                      if os.path.isfile(os.path.join(negative_dir, file))]
    print("{} positive files and {} negative files found.\n".format(len(positive_files), len(negative_files)))

    # color space info
    color_space = color_space.lower()
    if color_space == "hls":
        color_const = cv2.COLOR_BGR2HLS
    elif color_space == "hsv":
        color_const = cv2.COLOR_BGR2HSV
    elif color_space == "luv":
        color_const = cv2.COLOR_BGR2Luv
    elif color_space == "ycrcb" or color_space == "ycc":
        color_const = cv2.COLOR_BGR2YCrCb
    elif color_space == "yuv":
        color_const = cv2.COLOR_BGR2YUV
    else:
        color_const = -1

    # store feature vectors for both positive and negative files
    positive_features = []
    negative_features = []
    time_begin = time.time()

    # create feature descriptor object
    descriptor = Descriptor(hog=hog, histogram=histogram, spatial=spatial, hog_size=hog_size, hog_bins=hog_bins,
                            cell_size=cell_size, cells_per_block=cells_per_block, histogram_bins=histogram_bins,
                            spatial_size=spatial_size)

    # extract features from each file
    for i, file_path in enumerate(positive_files + negative_files):
        image = cv2.imread(file_path)
        if image is None:
            continue

        if color_const > -1:
            image = cv2.cvtColor(image, color_const)

        feature_vector = descriptor.get_features(image)

        if i < len(positive_files):
            positive_features.append(feature_vector)
        else:
            negative_features.append(feature_vector)

    print("Features extraction completed in {:.1f} seconds\n".format(time.time() - time_begin))

    num_features = len(positive_features[0])

    # scale features
    scaler = StandardScaler().fit(positive_features + negative_features)
    positive_features = scaler.transform(positive_features)
    negative_features = scaler.transform(negative_features)
    
    # randomize lists of feature vectors by splitting them into training, cross-validation, and test sets
    # the ratio is 75/20/5
    random.shuffle(positive_features)
    random.shuffle(negative_features)

    num_positive_train = int(round(0.75 * len(positive_features)))
    num_negative_train = int(round(0.75 * len(negative_features)))
    num_positive_val = int(round(0.2 * len(positive_features)))
    num_negative_val = int(round(0.2 * len(negative_features)))

    positive_train = positive_features[0:num_positive_train]
    negative_train = negative_features[0:num_negative_train]

    positive_val = positive_features[num_positive_train:(num_positive_train + num_positive_val)]
    negative_val = negative_features[num_negative_train:(num_negative_train + num_negative_val)]

    positive_test = positive_features[(num_positive_train + num_positive_val):]
    negative_test = negative_features[(num_negative_train + num_negative_val):]

    print("Randomized images into training, cross-validation, and test sets.\n")
    print("{} images in positive training set.".format(len(positive_train)))
    print("{} images in positive cross-validation set.".format(len(positive_val)))
    print("{} images in positive test set.".format(len(positive_test)))
    print("{} total positive images.\n".format(len(positive_train) + len(positive_val) + len(positive_test)))
    print("{} images in negative training set.".format(len(negative_train)))
    print("{} images in negative cross-validation set.".format(len(negative_val)))
    print("{} images in negative test set.".format(len(negative_test)))
    print("{} total negative images.\n".format(len(negative_train) + len(negative_val) + len(negative_test)))

    # store data and parameters in a dictionary
    feature_data = {
        "positive_train": positive_train,
        "negative_train": negative_train,
        "positive_val": positive_val,
        "negative_val": negative_val,
        "positive_test": positive_test,
        "negative_test": negative_test,
        "scaler": scaler,
        "hog": hog,
        "histogram": histogram,
        "spatial": spatial,
        "color_space": color_space,
        "color_const": color_const,
        "channels": channels,
        "hog_size": hog_size,
        "hog_bins": hog_bins,
        "cell_size": cell_size,
        "cells_per_block": cells_per_block,
        "histogram_bins": histogram_bins,
        "spatial_size": spatial_size,
        "num_features": num_features
    }

    return feature_data


def train_svm(file_path=None, feature_data=None, C=1, dual=False, fit_intercept=False,
              output_file=False, output_filename=None):
    if file_path is not None:
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError("File " + file_path + " not found.")
        feature_data = pickle.load(open(file_path, "rb"))
    elif feature_data is None:
        raise ValueError("Invalid feature data.")

    # train model on training set
    positive_train = np.asarray(feature_data["positive_train"])
    negative_train = np.asarray(feature_data["negative_train"])
    positive_val = np.asarray(feature_data["positive_val"])
    negative_val = np.asarray(feature_data["negative_val"])
    positive_test = np.asarray(feature_data["positive_test"])
    negative_test = np.asarray(feature_data["negative_test"])

    train_set = np.vstack((positive_train, negative_train))
    train_labels = np.concatenate((np.ones(positive_train.shape[0],), np.zeros(negative_train.shape[0],)))

    time_begin = time.time()
    model = svm.LinearSVC(C=C, dual=dual, fit_intercept=fit_intercept)
    model.fit(train_set, train_labels)
    print("Model train completed in {:.1f} s.\n".format(time.time() - time_begin))

    # cross-validation set
    positive_val_predicted = model.predict(positive_val)
    negative_val_predicted = model.predict(negative_val)

    false_negative_val = np.sum(positive_val_predicted != 1)
    false_positive_val = np.sum(negative_val_predicted == 1)
    positive_predict_accuracy = 1 - (false_negative_val / float(positive_val.shape[0]))
    negative_predict_accuracy = 1 - (false_positive_val / float(negative_val.shape[0]))
    total_accuracy = 1 - ((false_negative_val + false_positive_val) / float(positive_val.shape[0] +
                                                                            negative_val.shape[0]))

    print("Validation set false negatives: {} / {} ({:.3}% accuracy)".format(
        false_negative_val, positive_val.shape[0], 100 * positive_predict_accuracy))
    print("Validation set false positives: {} / {} ({:.3f}% accuracy)".format(
        false_positive_val, negative_val.shape[0], 100 * negative_predict_accuracy))
    print("Validation set total wrong classifications: {} / {} ({:.3f}% accuracy)\n".format(
        false_negative_val + false_positive_val, positive_val.shape[0] + negative_val.shape[0],
        100 * total_accuracy))

    # retrain model
    positive_train = np.vstack((positive_train, positive_val[positive_val_predicted != 1, :]))
    negative_train = np.vstack((negative_train, negative_val[negative_val_predicted == 1, :]))
    train_set = np.vstack((positive_train, negative_train))
    train_labels = np.concatenate((np.ones(positive_train.shape[0],), np.zeros(negative_train.shape[0],)))

    model.fit(train_set, train_labels)

    positive_test_predicted = model.predict(positive_test)
    negative_test_predicted = model.predict(negative_test)

    false_negative_test = np.sum(positive_test_predicted != 1)
    false_positive_test = np.sum(negative_test_predicted == 1)
    positive_predict_accuracy = 1 - (false_negative_test / float(positive_test.shape[0]))
    negative_predict_accuracy = 1 - (false_positive_test / float(negative_test.shape[0]))
    total_accuracy = 1 - ((false_negative_test + false_positive_test) / float(positive_test.shape[0] +
                                                                              negative_test.shape[0]))

    print("Model retrained.\n")
    print("Test set false negatives: {} / {} ({:.3}% accuracy)".format(
        false_negative_test, positive_test.shape[0], 100 * positive_predict_accuracy))
    print("Test set false positives: {} / {} ({:.3f}% accuracy)".format(
        false_positive_test, negative_test.shape[0], 100 * negative_predict_accuracy))
    print("Test set total misclassifications: {} / {} ({:.3f}% accuracy)".format(
        false_negative_test + false_positive_test, positive_test.shape[0] + negative_test.shape[0],
        100 * total_accuracy))

    # create a new dict that excludes image data from feature_data dict
    exclude_keys = ("positive_train", "negative_train", "positive_val", "negative_val", "positive_test",
                    "negative_test")
    model_data = {key: val for key, val in feature_data.items() if key not in exclude_keys}
    model_data["model"] = model

    if output_file:
        if output_filename is None:
            output_filename = "temp_model.pkl"

        pickle.dump(model_data, open(output_filename, "wb"))

    return model_data
