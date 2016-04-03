import numpy as np
import matplotlib.pyplot as plt
import LoadingData as ld

__author__ = 'PengYan'
# modified @ 23th, Jan., 2016
# modified @ 22th, Jan., 2016


def fisher_score(feature_set, label_set):
    # fisher score for single label(0-1) problems
    # feature_set and label_set shall ba matrix, each sample a row
    # return a 2 dimension narray, index is the stored_index for labels in decreasing order,
    # feature_score is the sorted scores

    print('FisherScore')
    feature_set = np.mat(feature_set)
    label_set = np.mat(label_set)

    nrow, ncol = feature_set.shape
    feature_score = np.zeros(ncol)
    all_mean = feature_set.mean(axis=0)

    # divide data into positive set and negative set
    p_index = []
    n_index = []
    for row in range(nrow):
        if label_set[row, 0] == 0:
            n_index.append(row)
        else:
            p_index.append(row)

    if n_index.__len__() == 0 or p_index.__len__() == 0:
        print('Not Enough Positive or Negative Samples')
        sorted_index = feature_score.argsort().tolist()
        sorted_index.reverse()
        sorted_index = np.array(sorted_index)
        feature_score = feature_score[sorted_index]
        return sorted_index, feature_score

    n_positive = p_index.__len__()
    n_negative = n_index.__len__()

    p_mean = feature_set[p_index, :].mean(axis=0)
    n_mean = feature_set[n_index, :].mean(axis=0)

    p_var = feature_set[p_index, :].var(axis=0)
    n_var = feature_set[n_index, :].var(axis=0)

    print('Ranking Features')
    for col in range(ncol):
        if (n_positive * p_var[0, col] + n_negative * n_var[0, col]) == 0:
            feature_score[col] = 0
        else:
            '''
            feature_score[col] += (n_positive * (p_mean[0, col] - all_mean[0, col]) ** 2 +
                                   n_negative * (n_mean[0, col] - all_mean[0, col]) ** 2) /\
                                  (n_positive * p_var[0, col] + n_negative * n_var[0, col])
            '''
            feature_score[col] += (n_positive * (p_mean[0, col] - all_mean[0, col]) ** 2 +
                                   n_negative * (n_mean[0, col] - all_mean[0, col]) ** 2) /\
                                  (((n_positive - 1) * p_var[0, col] + (n_negative - 1) * n_var[0, col]) / (nrow - 2))

    sorted_index = feature_score.argsort().tolist()
    sorted_index.reverse()
    sorted_index = np.array(sorted_index)
    feature_score = feature_score[sorted_index]
    return sorted_index, feature_score

if __name__ == '__main__':
    features = ld.import_matrix('CrossValidation/emotions/emotions_feature_train_0.csv')
    labels = ld.import_matrix('CrossValidation/emotions/emotions_label_train_0.csv')
    feature_names = np.array(ld.import_data('CrossValidation/emotions/emotions_feature_names.csv')[0])
    index, scores = fisher_score(features, labels[:, 0])
    print('Sorted Index:')
    print(index)
    print('Sorted Scores:')
    print(scores)
    print('Sorted Feature Names:')
    print(feature_names[index])
    plt.plot(scores)
    plt.show()
