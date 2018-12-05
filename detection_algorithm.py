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

    tally = np.zeros(clustering.cluster_centers_.shape[0])

    print (clustering.cluster_centers_)
    print (clustering.labels_)
    for i in labels:
        tally[i] += 1

    tally = tally.reshape(-1,1)
    clustering_tally = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit(tally)

    max_ind = np.argwhere(tally.flatten() == np.max(tally)).flatten()

    
    print("@@@",max_ind)
    to_merge_center = [clustering.cluster_centers_[x] for x in max_ind ]

    print (clustering_tally.labels_)
    print (tally)

    to_merge_points = [[] for i in range(len(clustering.cluster_centers_))]

    print (to_merge_points)
    ##work on the indexing issues
    for idx, i in enumerate(clustering.labels_):

        if i in max_ind:
            print ("###",i)
            print(labels[idx])

            to_merge_points[labels[idx]].append(rect_list[idx])

    to_merge_points = [x for x in to_merge_points if x]
    to_merge_points = np.asarray(to_merge_points)

    print(to_merge_center)
    print(to_merge_points)
    return (to_merge_center, to_merge_points)


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
print (mean_shift(np.array([[1, 1], [2, 1], [1, 0],[5,6],[6,6],[40, 7], [40, 6], [39, 6]]))[0])
