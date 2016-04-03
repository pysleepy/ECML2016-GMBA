import numpy as np
import time as time
import LoadingData as ld
import MultiLabelClassification as mlc
import MultiLabelFStatistic as mlfs
import MultiLabelRelief as mlrf
import SpectralFeatureSelection as sfs
import GMBA as gmb

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


def cross_validation_filter(data_name, nfold=10, need_shuffling=False, title_line=True, clf='kNN', mlc_type='BR',
                            classifier_threshold=3, compare_with='all_features', feature_threshold=0.2,
                            need_normalize=True, need_sampling=False, nsample=1800,
                            need_contaminate=False, random_seed=1231231):
    # cross validation with filter model(MLRF, MLFS, SPEC and GMBA, or all_features
    # data_name is the name of data set. first time run cross_validation, you need shuffling the data
    # by set need_shuffling=True. nfold denotes the fold of cross validation. title_line ara parameter in import_data(
    # see LoadingData for more information). clf denotes the base classifier, classifier_threshold is the parameter in
    # classifier, mlc_type denotes the transform strategy. compare_with denotes what feature selection method is used,
    # all_features denotes no feature selection. feature_threshold denote how many(percentage) features are used to
    # train and classify. need_contaminate denotes whether add noise into label set

    # initialize measurements
    hamming_loss = np.zeros((9, nfold))
    f1_macro = np.zeros((9, nfold))
    f1_micro = np.zeros((9, nfold))
    ranking_loss = np.zeros((9, nfold))
    average_precision = np.zeros((9, nfold))

    feature_selection_time = np.zeros((1, nfold))
    train_time = np.zeros((9, nfold))
    test_time = np.zeros((9, nfold))

    # generate random_seed
    if random_seed == 0.1:
        random_seed = int(time.time())

    # shuffling data
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
            # select features
            start_time = time.time()
            if compare_with == 'MLFS':
                index, scores = mlfs.multi_label_f_statistic(feature_train, label_train)
            elif compare_with == 'MLRF':
                index, scores = mlrf.multi_label_relief(feature_train, label_train)
            elif compare_with == 'SPEC':
                index, scores = sfs.spectral_feature_selection(feature_train, label_train)
            elif compare_with == 'GMBA':
                index, scores = gmb.gmba(feature_train, label_train)

            feature_selection_time[:, fold] += (time.time() - start_time)

            # test for specified number of features
            if feature_threshold != 0:
                # get feature list
                feature_list = index[range(int(feature_threshold * feature_train.shape[1]))]
                # select classifier
                if mlc_type == 'BR':
                    c = mlc.BinaryRelevance(label_train.shape[1])
                elif mlc_type == 'CLR':
                    c = mlc.CalibratedLabelRank(label_train.shape[1])
                else:
                    c = mlc.ClassifierChains(label_train.shape[1], rseed=random_seed)
                # training classifiers with selected features
                c.train(feature_train[:, feature_list], label_train, clf, classifier_threshold)
                # testing
                c.test(feature_test[:, feature_list])
                # get evaluations
                if mlc_type != 'CLR':
                    hamming_loss[:, fold], f1_macro[:, fold], f1_micro[:, fold], train_time[:, fold],\
                     test_time[:, fold] = c.evaluation(label_test)
                else:
                    ranking_loss[:, fold], average_precision[:, fold], train_time[:, fold], test_time[:, fold] =\
                        c.evaluation(label_test)
            else:
                for percent in range(1, 10):
                    # get feature list
                    feature_list = index[range(int(percent * 0.1 * feature_train.shape[1]))]
                    if mlc_type == 'BR':
                        c = mlc.BinaryRelevance(label_train.shape[1])
                    elif mlc_type == 'CLR':
                        c = mlc.CalibratedLabelRank(label_train.shape[1])
                    else:
                        c = mlc.ClassifierChains(label_train.shape[1], rseed=random_seed)
                    # training classifiers with selected features
                    c.train(feature_train[:, feature_list], label_train, clf, classifier_threshold)
                    # testing
                    c.test(feature_test[:, feature_list])
                    # get evaluations
                    if mlc_type != 'CLR':
                        hamming_loss[percent-1, fold], f1_macro[percent-1, fold], f1_micro[percent-1, fold],\
                            train_time[percent-1, fold], test_time[percent-1, fold] = c.evaluation(label_test)
                    else:
                        ranking_loss[percent-1, fold], average_precision[percent-1, fold],\
                            train_time[percent-1, fold], test_time[percent-1, fold] = c.evaluation(label_test)

    # results
    print('==========================================')
    print(data_name + '_' + str(feature_threshold) + '_' + mlc_type + '_' + compare_with + '_Filter:')

    if compare_with != 'all_features':
        print('Elapsed Time for Feature Selection:')
        print(str(feature_selection_time[0].mean()))

    if compare_with == 'all_features' or feature_threshold != 0:
        print('Elapsed Time for Training:')
        print(str(train_time[0].mean()) + '(' + str(train_time[0].mean() + feature_selection_time.mean()) + ')')
        print('Elapsed Time for Testing:')
        print(str(test_time[0].mean()))
        if mlc_type != 'CLR':
            print('Average Hamming Loss:')
            print(str(hamming_loss[0].mean()) + '(' + str(hamming_loss[0].std()) + '):')
            print('Average F1_Macro:')
            print(str(f1_macro[0].mean()) + '(' + str(f1_macro[0].std()) + '):')
            print('Average F1_Micro:')
            print(str(f1_micro[0].mean()) + '(' + str(f1_micro[0].std()) + '):')
            # return hamming_loss, f1_micro, f1_macro, feature_selection_time + train_time, test_time
        else:
            print('Average Ranking Loss:')
            print(str(ranking_loss[0].mean()) + '(' + str(ranking_loss[0].std()) + '):')
            print('Average Average Precision:')
            print(str(average_precision[0].mean()) + '(' + str(average_precision[0].std()) + '):')
            # return ranking_loss, average_precision, feature_selection_time + train_time, test_time
    if mlc_type != 'CLR':
        return hamming_loss, f1_macro, f1_micro, feature_selection_time + train_time, test_time
    else:
        return ranking_loss, average_precision, feature_selection_time + train_time, test_time

if __name__ == '__main__':
    cross_validation_filter(data_name='emotions', nfold=10, need_shuffling=False, feature_threshold=0, clf='kNN',
                            mlc_type='BR', classifier_threshold=3, need_normalize=True, compare_with='MLFS',
                            need_sampling=False, nsample=1800, need_contaminate=False, random_seed=0)
