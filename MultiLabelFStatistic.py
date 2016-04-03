import numpy as np
import LoadingData as ld
import matplotlib.pyplot as plt

__author__ = 'PengYan'
# modified @ 25th, Jan., 2016
# modified @ 24th, Jan., 2016


def multi_label_f_statistic(feature_set, label_set):
    # rank features using Multi Label F Statistic
    # Reference:
    # Ding, C.H., Huang, H., Kong, D., & Zhao, H.. (2012). Multi-label ReliefF and
    # F-statistic feature selections for image annotation. CVPR.

    # feature_set and label_set shall ba matrix, each sample a row
    # return a 2 dimension narray, index is the stored_index for labels in decreasing order,
    # feature_score is the sorted scores

    print('MLFS')
    nrow, ncol = feature_set.shape
    nlabel = label_set.shape[1]

    feature_score = np.zeros(ncol)

    numerator = np.mat(np.zeros((1, ncol)))
    denominator = 0
    for label in range(nlabel):
        for row in range(nrow):
            numerator += (label_set[row, label] * feature_set[row, :])
            denominator += (label_set[row, label])
    sample_mean_all = numerator * 1.0 / denominator

    sample_mean_label = dict()
    sample_list = dict()
    for label in range(nlabel):
        sample_list[label] = []
        for row in range(nrow):
            if label_set[row, label] == 1:
                sample_list[label].append(row)
        if feature_set[sample_list[label], :].__len__() > 0:
            sample_mean_label[label] = feature_set[sample_list[label], :].mean(axis=0)
        else:
            sample_mean_label[label] = np.mat(np.zeros((1, ncol)))

    '''
    scatter_between = np.mat(np.zeros((ncol, ncol)))
    for label in range(nlabel):
        for row in range(nrow):
            scatter_between += (label_set[row, label] * (sample_mean_label[label] - sample_mean_all).transpose() *
                                (sample_mean_label[label] - sample_mean_all))

    scatter_within = np.mat(np.zeros((ncol, ncol)))
    for label in range(nlabel):
        for row in range(nrow):
            scatter_within += (label_set[row, label] * (feature_set[row, :] - sample_mean_label[label]).transpose() *
                               (feature_set[row, :] - sample_mean_label[label]))

    print('Ranking Features:')
    for col in range(ncol):
        if (nlabel - 1) != 0 and scatter_within[col, col] != 0:
            feature_score[col] = (nrow - nlabel) * 1.0 / (nlabel - 1) * scatter_between[col, col] / scatter_within[col, col]
        elif (nlabel - 1) == 0 and scatter_within[col, col] != 0:
            feature_score[col] = (nrow - nlabel) * 1.0 / 1.0 * scatter_between[col, col] / scatter_within[col, col]
        elif (nlabel - 1) != 0 and scatter_within[col, col] == 0:
            feature_score[col] = (nrow - nlabel) * 1.0 / (nlabel - 1) * scatter_between[col, col] / 1.0
        else:
            feature_score[col] = (nrow - nlabel) * 1.0 / 1.0 * scatter_between[col, col] / 1.0
        print('Feature_' + str(col) + '(' + str(ncol) + '):' + str(feature_score[col]))
    '''

    print('Ranking Features:')

    for col in range(ncol):
        sw = 0.0
        for label in range(nlabel):
            if sample_list[label].__len__() != 0:
                sw += feature_set[sample_list[label], :][:, col].var(axis=0)
            else:
                sw += 0
        sb = 0.0
        for label in range(nlabel):
            sb += (sample_list[label].__len__() * (sample_mean_label[label][0, col] - sample_mean_all[0, col]) ** 2)

        if sw != 0:
            feature_score[col] = 1.0 * (nrow - nlabel) * sb / (nlabel - 1) / sw
        else:
            feature_score[col] = 0

    sorted_index = feature_score.argsort().tolist()
    sorted_index.reverse()
    sorted_index = np.array(sorted_index)
    feature_score = feature_score[sorted_index]
    return sorted_index, feature_score

if __name__ == '__main__':
    features = ld.import_matrix('CrossValidation/emotions/emotions_feature_train_0.csv')
    labels = ld.import_matrix('CrossValidation/emotions/emotions_label_train_0.csv')
    feature_names = np.array(ld.import_data('CrossValidation/emotions/emotions_feature_names.csv')[0])
    index, scores = multi_label_f_statistic(features, labels)
    print('Sorted Index:')
    print(index)
    print('Sorted Scores:')
    print(scores)
    print('Sorted Feature Names:')
    print(feature_names[index])
    plt.plot(scores)
    plt.show()
