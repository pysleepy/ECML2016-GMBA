import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as nn
import LoadingData as ld

__author__ = 'PengYan'
# modified @ 22th, Jan., 2016


def relief(feature_set, label_set, nb=3, sample_size=0.3, distance_p=2):
    # relief for single label(binary(0-1)) problems
    # feature_set and label_set shall ba matrix, each sample a row
    # nb is the number of neighbors
    # sample_size is the percentage of samples used to calculate
    # distance_p is the parameter in minkov distance
    # return a 2 1-dimension narray, index is the stored_index for labels in decreasing order,
    # feature_score is the sorted scores

    print('Relief')
    feature_set = np.mat(feature_set)
    label_set = np.mat(label_set)

    nrow, ncol = feature_set.shape
    sample_list = np.random.choice(range(nrow), int(nrow * sample_size), replace=False)

    feature_score = np.zeros(ncol)

    # divide data into positive set and negative set
    p_index = []
    n_index = []
    for row in sample_list:
        if label_set[row, 0] == 0:
            n_index.append(row)
        else:
            p_index.append(row)

    if p_index.__len__() <= nb or n_index.__len__() <= nb:
        print('Not Enough Positive or Negative Samples')
        sorted_index = feature_score.argsort().tolist()
        sorted_index.reverse()
        sorted_index = np.array(sorted_index)
        feature_score = feature_score[sorted_index]
        return sorted_index, feature_score

    p_feature = feature_set[p_index, :]
    n_feature = feature_set[n_index, :]

    p_neighbor_finder = nn.NearestNeighbors(n_neighbors=nb, p=distance_p)
    p_neighbor_finder.fit(np.asarray(p_feature))

    n_neighbor_finder = nn.NearestNeighbors(n_neighbors=nb, p=distance_p)
    n_neighbor_finder.fit(np.asarray(n_feature))

    print('Searching for Neighbors')
    p_neighbors = p_neighbor_finder.kneighbors(np.asarray(feature_set), return_distance=False)
    n_neighbors = n_neighbor_finder.kneighbors(np.asarray(feature_set), return_distance=False)

    print('Ranking Features')
    for row in sample_list:
        for col in range(ncol):
            diff_pn = 0
            diff_nn = 0
            for n in range(nb):
                diff_pn += np.linalg.norm(feature_set[row, col] - p_feature[p_neighbors[row, n], col])
                diff_nn += np.linalg.norm(feature_set[row, col] - n_feature[n_neighbors[row, n], col])
            if label_set[row, 0] == 1:
                feature_score[col] += (diff_nn / nb - diff_pn / nb)
            else:
                feature_score[col] += (diff_pn / nb - diff_nn / nb)
    feature_score /= (nrow * sample_size)

    sorted_index = feature_score.argsort().tolist()
    sorted_index.reverse()
    sorted_index = np.array(sorted_index)
    feature_score = feature_score[sorted_index]
    return sorted_index, feature_score

if __name__ == '__main__':
    features = ld.import_matrix('CrossValidation/emotions/emotions_feature_train_0.csv')
    labels = ld.import_matrix('CrossValidation/emotions/emotions_label_train_0.csv')
    feature_names = np.array(ld.import_data('CrossValidation/emotions/emotions_feature_names.csv')[0])
    index, scores = relief(features, labels[:, 1])
    print('Sorted Index:')
    print(index)
    print('Sorted Scores:')
    print(scores)
    print('Sorted Feature Names:')
    print(feature_names[index])
    plt.plot(scores)
    plt.show()
