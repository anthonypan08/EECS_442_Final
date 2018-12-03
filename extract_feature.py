from datetime import datetime
import os
import pickle
import random
import cv2
import numpy as np
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from descriptor import Descriptor


def process_files(pos_dir, neg_dir, hog_features=True, size=(64, 64),
                  hog_bins=9, pix_per_cell=(8, 8), cells_per_block=(2, 2), block_stride=None, block_norm="L1",
                  transform_sqrt=True, signed_gradient=False):
    """
    Extract features from positive and negative directories and store feature info to a dict.

    :param pos_dir: path to directory containing positive samples.
    :param neg_dir: path to directory containing negative samples.
    :param hog_features: boolean value for hog features in descriptor
    :return: dictionary containing training, cross-validation, test sets
    """

    pos_dir = os.path.abspath(pos_dir)
    neg_dir = os.path.abspath(neg_dir)

    if not os.path.isdir(pos_dir):
        raise FileNotFoundError("Directory " + pos_dir + " not found.")
    if not os.path.isdir(neg_dir):
        raise FileNotFoundError("Directory " + neg_dir + " not found.")

    pos_files = [os.path.join(pos_dir, file) for file in
                 os.listdir(pos_dir) if os.path.isfile(os.path.join(pos_dir, file))]
    neg_files = [os.path.join(neg_dir, file) for file in
                 os.listdir(neg_dir) if os.path.isfile(os.path.join(neg_dir, file))]
    print("{} positive files and {} negative files found.\n".format(len(pos_files), len(neg_files)))

    pos_features = []
    neg_features = []

    # get feature descriptor object
    descriptor = Descriptor(hog_features=hog_features, size=size, hog_bins=hog_bins, pix_per_cell=pix_per_cell,
                            cells_per_block=cells_per_block, block_stride=block_stride, block_norm=block_norm,
                            transform_sqrt=transform_sqrt, signed_gradient=signed_gradient)

    # iterate through files and extract features
    channels = [0, 1, 2]
    for i, filepath in enumerate(pos_files + neg_files):
        image = cv2.imread(filepath)
        if image is None:
            continue

        if len(image.shape) > 2:
            image = image[:, :, channels]

        feature_vector = descriptor.getFeatureVector(image)

        if i < len(pos_files):
            pos_features.append(feature_vector)
        else:
            neg_features.append(feature_vector)
    print("Features extracted from {} files\n".format(len(pos_features) + len(neg_features)))

    # instantiate scaler and scale features.
    scaler = StandardScaler().fit(pos_features + neg_features)
    pos_features = scaler.transform(pos_features)
    neg_features = scaler.transform(neg_features)

    # randomize lists of feature vectors and split 75/20/5 into training, cross-validation, and test sets respectively
    random.shuffle(pos_features)
    random.shuffle(neg_features)

    num_pos_val = int(round(0.2 * len(pos_features)))
    num_neg_val = int(round(0.2 * len(neg_features)))
    num_pos_train = int(round(0.75 * len(pos_features)))
    num_neg_train = int(round(0.75 * len(neg_features)))

    pos_train = pos_features[0:num_pos_train]
    neg_train = neg_features[0:num_neg_train]

    pos_val = pos_features[num_pos_train: (num_pos_train + num_pos_val)]
    neg_val = neg_features[num_neg_train: (num_neg_train + num_neg_val)]

    pos_test = pos_features[(num_pos_train + num_pos_val):]
    neg_test = neg_features[(num_neg_train + num_neg_val):]

    print("{} samples in positive training set.".format(len(pos_train)))
    print("{} samples in positive cross-validation set.".format(len(pos_val)))
    print("{} samples in positive test set.".format(len(pos_test)))
    print("{} total positive samples.\n".format(len(pos_train) + len(pos_val) + len(pos_test)))

    print("{} samples in negative training set.".format(len(neg_train)))
    print("{} samples in negative cross-validation set.".format(len(neg_val)))
    print("{} samples in negative test set.".format(len(neg_test)))
    print("{} total negative samples.\n".format(len(neg_train) + len(neg_val) + len(neg_test)))

    # Store sample data and parameters in dict.
    # Descriptor class object seems to produce errors when unpickling and
    # has been commented out below. The descriptor will be re-instantiated
    # by the Detector object later.
    feature_data = {
        "pos_train": pos_train,
        "neg_train": neg_train,
        "pos_val": pos_val,
        "neg_val": neg_val,
        "pos_test": pos_test,
        "neg_test": neg_test,
    }

    return feature_data


def train_adaboost(feature_data=None, n_estimators=50, base_estimator=None,
                   learning_rate=1.0, output_model=False, output_filename=None):
    """
    Train adaboost from feature data extracted by process_files() and save model to filename.

    :param feature_data: feature data dictionary from process_files().
    :param n_estimators: number of estimators for adaboost classifier.
    :param base_estimator: base_estimator for adaboost classifier.
    :param learning_rate: integer learning rate for adaboost classifier.
    :param output_model: boolean value for saving model to file.
    :param output_filename: name of output file.
    :return: None.
    """

    # train classifier on training set
    pos_train = np.asarray(feature_data["pos_train"])
    neg_train = np.asarray(feature_data["neg_train"])
    pos_val = np.asarray(feature_data["pos_val"])
    neg_val = np.asarray(feature_data["neg_val"])
    pos_test = np.asarray(feature_data["pos_test"])
    neg_test = np.asarray(feature_data["neg_test"])

    # train classifier
    train_set = np.vstack((pos_train, neg_train))
    train_labels = np.concatenate((np.ones(pos_train.shape[0],), np.zeros(neg_train.shape[0],)))
    classifier = AdaBoostClassifier(base_estimator=base_estimator, learning_rate=learning_rate,
                                    n_estimators=n_estimators)
    classifier.fit(train_set, train_labels)
    print("Classifier trained!")

    # run classifier on cross-validation set
    pos_val_predict = classifier.predict(pos_val)
    neg_val_predict = classifier.predict(neg_val)
    false_neg_val = np.sum(pos_val_predict != 1)
    false_pos_val = np.sum(neg_val_predict == 1)
    pos_predict_accuracy = 1 - (false_neg_val / float(pos_val.shape[0]))
    neg_predict_accuracy = 1 - (false_pos_val / float(neg_val.shape[0]))
    total_accuracy = 1 - ((false_neg_val + false_pos_val) / float(pos_val.shape[0] + neg_val.shape[0]))

    print("Validation set false negatives: {} / {} ({:.3}% accuracy)".format(
        false_neg_val, pos_val.shape[0], 100 * pos_predict_accuracy))
    print("Validation set false positives: {} / {} ({:.3f}% accuracy)".format(
        false_pos_val, neg_val.shape[0], 100 * neg_predict_accuracy))
    print("Validation set total misclassifications: {} / {} ({:.3f}% accuracy)\n".format(
        false_neg_val + false_pos_val, pos_val.shape[0] + neg_val.shape[0],
        100 * total_accuracy))

    # retrain classifier with misses from validation set and run on test set
    pos_train = np.vstack((pos_train, pos_val[pos_val_predict != 1, :]))
    neg_train = np.vstack((neg_train, neg_val[neg_val_predict == 1, :]))
    train_set = np.vstack((pos_train, neg_train))
    train_labels = np.concatenate((np.ones(pos_train.shape[0],), np.zeros(neg_train.shape[0],)))

    classifier.fit(train_set, train_labels)
    print("Classifier retrained!\n")

    pos_test_predicted = classifier.predict(pos_test)
    neg_test_predicted = classifier.predict(neg_test)

    false_neg_test = np.sum(pos_test_predicted != 1)
    false_pos_test = np.sum(neg_test_predicted == 1)
    pos_predict_accuracy = 1 - (false_neg_test / float(pos_test.shape[0]))
    neg_predict_accuracy = 1 - (false_pos_test / float(neg_test.shape[0]))
    total_accuracy = 1 - ((false_neg_test + false_pos_test) / float(pos_test.shape[0] + neg_test.shape[0]))

    print("Test set false negatives: {} / {} ({:.3}% accuracy)".format(
        false_neg_test, pos_test.shape[0], 100 * pos_predict_accuracy))
    print("Test set false positives: {} / {} ({:.3f}% accuracy)".format(
        false_pos_test, neg_test.shape[0], 100 * neg_predict_accuracy))
    print("Test set total misclassifications: {} / {} ({:.3f}% accuracy)".format(
        false_neg_test + false_pos_test, pos_test.shape[0] + neg_test.shape[0],
        100 * total_accuracy))

    # output model to pickle file
    if output_model:
        if output_filename is None:
            output_filename = (datetime.now().strftime("%Y%m%d%H%M") + "_classifier.pkl")
        pickle.dump(classifier, open(output_filename, "wb"))
        print("\nAdaboost classifier data saved to {}".format(output_filename))

    return None


def main():
    pos_dir = "data/positive"
    neg_dir = "data/negative"
    feature = process_files(pos_dir, neg_dir)
    train_adaboost(feature_data=feature, n_estimators=100, output_model=True, output_filename="temp_classifier.pkl")


if __name__ == "__main__":
    main()
