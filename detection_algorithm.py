import numpy as np
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering


def get_kernel(x, constant = 1):
    norm_x = np.linalg.norm(x)
    if  norm_x < 1:
        return constant * (1 - norm_x)
    return 0



def mean_shift(rect_list):
    band = 1
    clustering = MeanShift(bandwidth=band).fit(X)
    labels = clustering.labels_
    tally = np.zeros(clustering.cluster_centers_.length)
    for i in labels:
        tally[i] += 1
    clustering = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit(tally)

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
