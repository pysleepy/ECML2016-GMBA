import numpy as np
import sklearn.neighbors as nn
import matplotlib.pyplot as plt

import LoadingData as ld
import BuildGraph as bg

__author__ = 'PengYan'
# modified @ 30th, Jan., 2016


def gmba(feature_set, label_set, nb=3, sample_size=1, similar_threshold=1.0, beta=0.9, stype='WeJaccard', var=1):
    # feature_set and label_set shall ba matrix, each sample a row
    # nb is the number of neighbors
    # sample_size is the percentage of samples used to calculate
    # similar_threshold is the threshold to divide samples into hit and miss
    # beta is the step length parameter in gmba
    # stype and var are the same as those in get_similarity()
    # return a 2 1-dimension narray, index is the stored_index for labels in decreasing order,
    # feature_score is the sorted scores


    print('GMBA')

    n_row, n_col = feature_set.shape
    n_label = label_set.shape[1]

    n_sample = int(n_row * sample_size)
    sample_list = np.random.choice(range(n_row), n_sample, replace=False)

    feature_score = np.zeros(n_col) + 1

    sample_same = dict()
    sample_diff = dict()
    near_hit = dict()
    near_hit_distance = dict()
    margin = dict()

    # divide hit and miss
    adjacent_matrix = bg.build_adjacent_matrix(label_set, stype, var)
    for row in range(n_row):
        sample_same[row] = []
        sample_diff[row] = []
        for col in range(n_row):
            if row != col:
                if adjacent_matrix[row, col] >= similar_threshold:
                    sample_same[row].append(col)
                else:
                    sample_diff[row].append(col)

    print('Searching near_hit, near_miss')
    # find nearest hit, nearest miss and margin
    for sample in range(n_row):
        if sample_same[sample].__len__() >= nb and sample_diff[sample].__len__() >= nb:
            # near hit
            neighbor_finder = nn.NearestNeighbors(n_neighbors=nb)
            neighbor_finder.fit(feature_set[sample_same[sample], :])
            score, ind = neighbor_finder.kneighbors(feature_set[sample, :])
            near_hit[sample] = ind[0]
            near_hit_distance[sample] = score[0]
            # near miss
            neighbor_finder = nn.NearestNeighbors(n_neighbors=1)
            neighbor_finder.fit(feature_set[sample_diff[sample], :])
            score, ind = neighbor_finder.kneighbors(feature_set[sample, :])
            # margin
            margin[sample] = abs(near_hit_distance[sample][0] ** 2 - score[0][0] ** 2)

    print('Ranking Features')
    for ind in range(n_sample):
        sample = sample_list[ind]
        current_sample = feature_set[sample, :]
        diff = np.zeros(n_col)
        if sample_same[sample].__len__() >= nb and sample_diff[sample].__len__() >= nb:
            temp = np.zeros(n_col)
            for hit in range(nb):
                temp += (2.0 * feature_score * adjacent_matrix[sample, near_hit[sample][hit]] *
                         ((np.asarray(current_sample)[0]) - (np.asarray(feature_set)[sample_same[sample][near_hit[sample][hit]]])) ** 2)
                for miss in range(sample_diff[sample].__len__()):
                    d = current_sample - feature_set[sample_diff[sample][miss], :]
                    if (np.linalg.norm(d) ** 2) < (near_hit_distance[sample][hit] ** 2 + margin[sample]):
                        temp += (2.0 * feature_score * (adjacent_matrix[sample, sample_same[sample][near_hit[sample][hit]]] - adjacent_matrix[sample, sample_diff[sample][miss]]) *
                                 (((np.asarray(current_sample)[0] - np.asarray(feature_set)[sample_same[sample][near_hit[sample][hit]]]) ** 2) -
                                  ((np.asarray(current_sample)[0] - np.asarray(feature_set)[sample_diff[sample][miss]]) ** 2)))
                diff += temp
            # print(diff / np.linalg.norm(diff))
            if np.linalg.norm(diff) != 0:
                feature_score -= (beta * diff / np.linalg.norm(diff))

    sorted_index = feature_score.argsort().tolist()
    sorted_index.reverse()
    sorted_index = np.array(sorted_index)
    feature_score = feature_score[sorted_index]
    return sorted_index, feature_score

if __name__ == '__main__':
    data_name = 'emotions'
    features = ld.import_matrix('CrossValidation/' + data_name + '/' + data_name + '_feature_train_0.csv')
    labels = ld.import_matrix('CrossValidation/' + data_name + '/' + data_name + '_label_train_0.csv')
    feature_names = np.array(ld.import_data('CrossValidation/' + data_name + '/' + data_name + '_feature_names.csv')[0])

    index, scores = gmba(features, labels)
    print('Sorted Index:')
    print(index)
    print('Sorted Scores:')
    print(scores)
    print('Sorted Feature Names:')
    print(feature_names[index])
    plt.plot(scores)
    plt.show()
