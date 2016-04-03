import time as time
import numpy as np
import LoadingData as ld
import MultiLabelClassification as mlc

__author__ = 'PengYan'
# modified @ 25th, Jan., 2016
# modified @ 24th, Jan., 2016
# modified @ 23th, Jan., 2016
# modified @ 22th, Jan., 2016
# modified @ 21th, Jan., 2016
# modified @ 20th, Jan., 2016
# modified @ 19th, Jan., 2016


def normalization(origin_data, need_normalize):
    # eliminate features with variance 0
    # if need_normalize is True, normalize the data(matrix) by z-score and return it
    data_matrix = np.mat(origin_data).copy()
    nrow, ncol = data_matrix.shape
    delete_list = []
    print('Normalizing')
    for col in range(ncol):
        # if var is zero, delete it
        if abs(data_matrix[:, col].std() - 0) < 0.0001:
            delete_list.append(col)
        elif need_normalize:
            # z-score
            data_matrix[:, col] -= data_matrix[:, col].mean()
            data_matrix[:, col] /= data_matrix[:, col].std()
    print('feature: ' + str(delete_list) + ' deleted')
    data_matrix = np.delete(data_matrix, delete_list, axis=1)
    return data_matrix


def contaminate(origin_data, threshold=0.05, rseed=0):
    # whether to contaminate the label set(generate noise)
    # threshold denotes the probability of noise and rseed is the random seed
    nrow, ncol = origin_data.shape
    contaminated_size = int(nrow * threshold)
    for col in range(ncol):
        index = np.random.RandomState(rseed+col).choice(range(nrow), size=contaminated_size, replace=False)
        for ind in range(contaminated_size):
            temp = bool(origin_data[index[ind], col])
            origin_data[index[ind], col] = (temp ^ 1)
    return origin_data


def cross_validation_wrapper(data_name, nfold=10, need_shuffling=False, title_line=True, clf='kNN', mlc_type='BR',
                             classifier_threshold=3, compare_with='all_features', feature_threshold=0.2,
                             need_normalize=True, need_sampling=False, nsample=1800,
                             need_contaminate=False, random_seed=1231231):
    # similar to the cross_validation_filter, except for using wrapper methods for feature selection here, including
    # SPCE, Fisher_Score and Relief, or all_features

    # measurements
    hamming_loss = np.zeros((9, nfold))
    f1_macro = np.zeros((9, nfold))
    f1_micro = np.zeros((9, nfold))
    ranking_loss = np.zeros((9, nfold))
    average_precision = np.zeros((9, nfold))

    train_time = np.zeros((9, nfold))
    test_time = np.zeros((9, nfold))

    # generate random_seed
    if random_seed == 0.1:
        random_seed = int(time.time())

    # shuffling data if needed
    if need_shuffling:
        ld.generate_validate_data(data_name, nfold, title_line)

    for fold in range(nfold):
        print('fold:' + str(fold))
        # loading data
        feature_train = ld.import_matrix('CrossValidation/' + data_name + '/' + data_name +
                                         '_feature_train_' + str(fold) + '.csv')
        label_train = ld.import_matrix('CrossValidation/' + data_name + '/' + data_name +
                                       '_label_train_' + str(fold) + '.csv')
        feature_test = ld.import_matrix('CrossValidation/' + data_name + '/' + data_name +
                                        '_feature_test_' + str(fold) + '.csv')
        label_test = ld.import_matrix('CrossValidation/' + data_name + '/' + data_name +
                                      '_label_test_' + str(fold) + '.csv')

        # sampling if needed
        if need_sampling:
            feature_train = feature_train[0:nsample, :]
            label_train = label_train[0:nsample, :]

        # normalization data if needed
        nrow_train = feature_train.shape[0]
        data = np.append(feature_train, feature_test, axis=0)
        data = normalization(data, need_normalize)
        feature_train = data[0:nrow_train, :]
        feature_test = data[nrow_train:, :]

        # contaminate if needed
        if need_contaminate:
            label_train = contaminate(label_train, rseed=random_seed)

        if compare_with == 'all_features':
            # select classifier
            if mlc_type == 'BR':
                c = mlc.BinaryRelevance(label_train.shape[1])
            elif mlc_type == 'CLR':
                c = mlc.CalibratedLabelRank(label_train.shape[1])
            else:
                c = mlc.ClassifierChains(label_train.shape[1], rseed=random_seed)

            # training classifiers with all features
            c.train(feature_train, label_train, clf, classifier_threshold)
            # testing
            c.test(feature_test)
            # get evaluations
            if mlc_type != 'CLR':
                hamming_loss[:, fold], f1_macro[:, fold], f1_micro[:, fold], train_time[:, fold], test_time[:, fold] =\
                    c.evaluation(label_test)
            else:
                ranking_loss[:, fold], average_precision[:, fold], train_time[:, fold], test_time[:, fold] =\
                    c.evaluation(label_test)
        else:
            # test with feature selection
            if mlc_type == 'BR':
                    c = mlc.BinaryRelevance(label_train.shape[1], select_features=True, fs_method=compare_with,
                                            feature_threshold=feature_threshold)
            elif mlc_type == 'CLR':
                c = mlc.CalibratedLabelRank(label_train.shape[1], select_features=True, fs_method=compare_with,
                                            feature_threshold=feature_threshold)
            else:
                c = mlc.ClassifierChains(label_train.shape[1], select_features=True, fs_method=compare_with,
                                         feature_threshold=feature_threshold, rseed=random_seed)

            # training with fs methods
            c.train(feature_train, label_train, clf, classifier_threshold)
            # testing
            c.test(feature_test)

            # get evaluations
            if mlc_type != 'CLR':
                hamming_loss[:, fold], f1_macro[:, fold], f1_micro[:, fold], train_time[:, fold], test_time[:, fold] =\
                    c.evaluation(label_test)
            else:
                ranking_loss[:, fold], average_precision[:, fold], train_time[:, fold], test_time[:, fold] =\
                    c.evaluation(label_test)

    # results
    print('==========================================')
    print(data_name + '_' + str(feature_threshold) + '_' + mlc_type + '_' + compare_with + '_Wrapper:')

    if compare_with == 'all_features' or feature_threshold != 0:
        print('Elapsed Time for Training:')
        print(str(train_time[0].mean()))
        print('Elapsed Time for Testing:')
        print(str(test_time[0].mean()))
        if mlc_type != 'CLR':
            print('Average Hamming Loss:')
            print(str(hamming_loss[0].mean()) + '(' + str(hamming_loss[0].std()) + '):')
            print('Average F1_Macro:')
            print(str(f1_macro[0].mean()) + '(' + str(f1_macro[0].std()) + '):')
            print('Average F1_Micro:')
            print(str(f1_micro[0].mean()) + '(' + str(f1_micro[0].std()) + '):')
        else:
            print('Average Ranking Loss:')
            print(str(ranking_loss[0].mean()) + '(' + str(ranking_loss[0].std()) + '):')
            print('Average Average Precision:')
            print(str(average_precision[0].mean()) + '(' + str(average_precision[0].std()) + '):')
    if mlc_type != 'CLR':
        return hamming_loss, f1_macro, f1_micro, train_time, test_time
    else:
        return ranking_loss, average_precision, train_time, test_time

if __name__ == '__main__':
    cross_validation_wrapper(data_name='emotions', nfold=10, need_shuffling=False, feature_threshold=0.2, clf='kNN',
                             mlc_type='CLR', classifier_threshold=3, need_normalize=True, compare_with='SPEC',
                             need_sampling=False, nsample=1800, need_contaminate=False, random_seed=0)
