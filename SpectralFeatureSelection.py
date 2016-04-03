import numpy as np
import LaplacianMatrix as lm
import matplotlib.pyplot as plt
import BuildGraph as bg
import LoadingData as ld
import CrossValidationWrapper as cv


__author__ = 'PengYan'
# modified @ 22th, Jan., 2016
# modified @ 18th, Jan., 2016
# modified @ 15th, Jan., 2016
# modified @ 14th, Jan., 2016


def spectral_feature_selection(feature_set, label_set, rtype=1, stype='WeJaccard', var=1):
    # rank features using spectral methods
    # Reference:
    # Zhao Z., Liu H. Spectral feature selection for supervised and unsupervised learning[C].\
    # Proceedings of the 24th international conference on Machine learning. ACM, 2007: 1151-1157.

    # feature_set and label_set shall ba matrix, each sample a row
    # rtype = 1 for the first kind of feature score, 2 for the second kind
    # stype and var are the same as those in get_similarity()
    # return a 2 1-dimension narray, index is the stored_index for labels in increasing order,
    # feature_score is the sorted scores


    print('SPEC')
    nrow, ncol = feature_set.shape
    feature_score = np.zeros(ncol)

    adjacent_matrix = bg.build_adjacent_matrix(label_set, stype, var)
    deg_matrix = lm.get_deg_matrix(adjacent_matrix)
    sqrt_deg_matrix = np.mat(np.diag(np.diag(deg_matrix) ** 0.5))
    laplacian_matrix = lm.get_laplacian_matrix(adjacent_matrix)

    # laplacian_matrix = laplacian_matrix * laplacian_matrix * laplacian_matrix * laplacian_matrix

    eigen_values, eigen_vectors = lm.get_spectrum(adjacent_matrix)

    print('Ranking Features:')
    for col in range(ncol):

        normalized_feature = sqrt_deg_matrix * feature_set[:, col]
        if np.linalg.norm(normalized_feature) != 0:
            normalized_feature /= np.linalg.norm(normalized_feature)
        else:
            normalized_feature += 1
            normalized_feature /= np.linalg.norm(normalized_feature)
        feature_score[col] = normalized_feature.transpose() * laplacian_matrix * normalized_feature
        if rtype == 2:
            if (1 - normalized_feature.transpose() * eigen_vectors[:, 0]) != 0:
                feature_score[col] /= (1 - normalized_feature.transpose() * eigen_vectors[:, 0])
            else:
                feature_score[col] = 2.0

        print('Feature_' + str(col) + '(' + str(ncol) + '):' + str(feature_score[col]))
    sorted_index = feature_score.argsort()
    feature_score.sort()

    return sorted_index, feature_score


if __name__ == '__main__':
    data_name = 'emotions'
    nfold = 10
    need_normalize = True
    sampling = False
    nsample = 2500

    ld.generate_validate_data(data_name, nfold)
    for fold in range(nfold):
        print('fold:' + str(fold))
        features = ld.import_matrix('CrossValidation/' + data_name + '/' + data_name +
                                    '_feature_train_' + str(fold) + '.csv')
        labels = ld.import_matrix('CrossValidation/' + data_name + '/' + data_name +
                                  '_label_train_' + str(fold) + '.csv')
        feature_names = np.array(ld.import_data('CrossValidation/' + data_name + '/' + data_name +
                                                '_feature_names.csv')[0])

        if sampling:
            features = features[0:nsample, :]
            labels = labels[0:nsample, :]

        # temp:normalization
        if need_normalize:
            features = cv.normalization(features)

        index, scores = spectral_feature_selection(features, labels)
        print('Sorted Index:')
        print(index)
        print('Sorted Scores:')
        print(scores)
        print('Sorted Feature Names:')
        print(feature_names[index])
        plt.hold(True)
        plt.plot(scores)
    plt.show()
