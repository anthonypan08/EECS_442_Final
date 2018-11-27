import numpy as np
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering

"""
def get_kernel(x, constant = 1):
    norm_x = np.linalg.norm(x)
    if  norm_x < 1:
        return constant * (1 - norm_x)
    return 0

"""

def mean_shift(rect_list):
    band = 1
    clustering = MeanShift(bandwidth=band).fit(rect_list)
    labels = clustering.labels_
    tally = np.zeros(clustering.cluster_centers_.length)
    to_merge_center_idx = []

    for i in labels:
        tally[i] += 1
    clustering_tally = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit(tally)

    max_ind = np.argmax(tally)
    max_ind = clustering_tally.labels_[max_ind]

    for idx, i in enumerate(clustering_tally.labels_):
        if i == max_ind:
            to_merge_center_idx.append(tally[idx])
    to_merge_points [] * to_merge_center_idx.length

    for idx, i in enumerate(labels):
        if i in to_merge_center_idx:
            to_merge_points[i].append(rect_list[idx])

    to_merge_points = np.asarray(to_merge_points)

    return to_merge_points


    """
    iter = 0
    x = rect_list[0]
    mean = 0
    h = 1
    while iter = 0 or (abs(mean - x) > 0.000001 and iter < 10000):
        numerator = 0
        denominator = 0
        for i in rect_list:
            if get_kernel(x - i)
            numerator += i * get_kernel((x - i / h))
            denomiator += get_kernel((x - i / h))


        iter += 1
    """
