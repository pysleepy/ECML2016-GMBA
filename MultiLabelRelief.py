import numpy as np
import sklearn.neighbors as nn
import matplotlib.pyplot as plt
import LoadingData as ld

__author__ = 'PengYan'
# modified @ 25th, Jan., 2016
# modified @ 24th, Jan., 2016


def multi_label_relief(feature_set, label_set, nb=3, sample_size=1, distance_p=2):
    # rank features using Multi Label F Statistic
    # Reference:
    # Ding, C.H., Huang, H., Kong, D., & Zhao, H.. (2012). Multi-label ReliefF and
    # F-statistic feature selections for image annotation. CVPR.

    # feature_set and label_set shall ba matrix, each sample a row
    # nb is the number of neighbors
    # sample_size is the percentage of samples used to calculate
    # distance_p is the parameter in minkov distance
    # return a 2 1-dimension narray, index is the stored_index for labels in decreasing order,
    # feature_score is the sorted scores


    print('MLRF')
    nrow, ncol = feature_set.shape
    nlabel = label_set.shape[1]
    sample_list = np.random.choice(range(nrow), int(nrow * sample_size), replace=False)

    feature_score = np.zeros((1, ncol))

    prob = label_set.mean(axis=0)
    n_sample = dict()
    pair_score = dict()

    # find positive and negative samples for each label
    for label in range(nlabel):
        n_sample[label, 0] = []
        n_sample[label, 1] = []
        for row in sample_list:
            if label_set[row, label] == 0:
                n_sample[label, 0].append(row)
            else:
                n_sample[label, 1].append(row)

    for label1 in range(nlabel - 1):
        for label2 in range(label1 + 1, nlabel):
            pair_score[label1, label2] = np.zeros((1, ncol))
        if n_sample[label1, 0].__len__() >= nb and n_sample[label1, 1].__len__() >= nb:
            # find near miss for label1
            n_neighbor_finder = nn.NearestNeighbors(n_neighbors=nb, p=distance_p)
            n_neighbor_finder.fit(np.asarray(feature_set[n_sample[label1, 0], :]))
            near_miss = n_neighbor_finder.kneighbors(np.asarray(feature_set[n_sample[label1, 0], :]), return_distance=False)
            # find near hit for label1
            p_neighbor_finder = nn.NearestNeighbors(n_neighbors=nb, p=distance_p)
            p_neighbor_finder.fit(np.asarray(feature_set[n_sample[label1, 1], :]))
            near_hit = p_neighbor_finder.kneighbors(np.asarray(feature_set[n_sample[label1, 1], :]), return_distance=False)

            for label2 in range(label1 + 1, nlabel):
                for r in range(near_miss.__len__()):
                    for c in range(nb):
                        if label_set[near_miss[r, c], label2] == 1:
                            pair_score[label1, label2] += 1.0 * (prob[0, label2] / (1 - prob[0, label1])) *\
                                                          np.abs(feature_set[n_sample[label1, 0][r], :] -
                                                                 feature_set[near_miss[r, c], :]) / nb
                for r in range(n_sample[label1, 1].__len__()):
                    for c in range(nb):
                        if label_set[near_hit[r, c], label2] == 1:
                            pair_score[label1, label2] -= (prob[0, label2] /(1 - prob[0, label1])) *\
                                                          np.abs(feature_set[n_sample[label1, 1][r], :] -
                                                                 feature_set[near_hit[r, c], :]) / nb

    for label1 in range(nlabel - 1):
        for label2 in range(label1 + 1, nlabel):
            feature_score += pair_score[label1, label2]

    feature_score = feature_score[0]
    sorted_index = feature_score.argsort().tolist()
    sorted_index.reverse()
    sorted_index = np.array(sorted_index)
    feature_score = feature_score[sorted_index]
    return sorted_index, feature_score

if __name__ == '__main__':
    features = ld.import_matrix('CrossValidation/emotions/emotions_feature_train_0.csv')
    labels = ld.import_matrix('CrossValidation/emotions/emotions_label_train_0.csv')
    feature_names = np.array(ld.import_data('CrossValidation/emotions/emotions_feature_names.csv')[0])
    index, scores = multi_label_relief(features, labels)
    print('Sorted Index:')
    print(index)
    print('Sorted Scores:')
    print(scores)
    print('Sorted Feature Names:')
    print(feature_names[index])
    plt.plot(scores)
    plt.show()
