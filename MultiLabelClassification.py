import sklearn.svm as svm
import sklearn.linear_model as lm
import sklearn.neighbors as nn
import sklearn.naive_bayes as nby
import numpy as np
import time as time
import LoadingData as ld
import Relief as rf
import FisherScore as fsc
import SpectralFeatureSelection as sfs



__author__ = 'PengYan'
# modified @ 23th, Jan., 2016
# modified @ 23th, Jan., 2016
# modified @ 22th, Jan., 2016
# modified @ 19th, Jan., 2016
# modified @ 16th, Jan., 2016
# modified @ 15th, Jan., 2016


class Evaluation(object):
    # evaluating results get by multi label classifiers
    # label_test is a matrix(one sample per row), is the true labels for every sample
    # label_predict is the predicted results(matrix, one sample per row)
    # label_ranking is the results from ranking algorithms, one sample per row

    # evaluating return hamming_loss, f1_macro and f1_micro for classifying results.(they are 1-dimension narray)
    # evaluating_rank return ranking_loss and average_precision for ranking algorithms(they are 1-dimension narray)

    def __init__(self, label_test, label_predict=[], label_ranking=[]):
        self.label_test = label_test
        self.label_predict = label_predict
        self.label_ranking = label_ranking
        self.nrow, self.nlabel = label_test.shape
        self.hamming_loss = 0
        self.f1_macro = 0
        self.f1_micro = 0
        self.tp = np.zeros(self.nlabel)
        self.fp = np.zeros(self.nlabel)
        self.tn = np.zeros(self.nlabel)
        self.fn = np.zeros(self.nlabel)
        self.ranking_loss = 0
        self.average_precision = 0

    def get_hamming_loss(self):
        for row in range(self.nrow):
            for label in range(self.nlabel):
                if self.label_test[row, label] != self.label_predict[row, label]:
                    self.hamming_loss += (1.0 / self.nlabel)
        self.hamming_loss = self.hamming_loss * 1.0 / self.nrow
        print('Hamming Loss:')
        print(self.hamming_loss)
        return self.hamming_loss

    def get_f1_measure(self):
        for row in range(self.nrow):
            for label in range(self.nlabel):
                if self.label_test[row, label] == self.label_predict[row, label]:
                    if self.label_test[row, label] == 0:
                        self.tn[label] += 1
                    else:
                        self.tp[label] += 1
                else:
                    if self.label_test[row, label] == 1:
                        self.fn[label] += 1
                    else:
                        self.fp[label] += 1
        # f1_macro
        for label in range(self.nlabel):
            if self.tp[label] + self.fp[label] + self.fn[label] != 0:
                self.f1_macro += 2.0 * self.tp[label] / (2.0 * self.tp[label] + self.fp[label] + self.fn[label]) /\
                                 self.nlabel
            else:
                self.f1_macro += (1 / self.nlabel)
        # f1_micro
        self.f1_micro = 2.0 * self.tp.sum() / (2.0 * self.tp.sum() + self.fn.sum() + self.fn.sum())
        print('F1_Macro:')
        print(self.f1_macro)
        print('F1_Micro:')
        print(self.f1_micro)
        return self.f1_macro, self.f1_micro

    def get_ranking_loss(self):
        for row in range(self.nrow):
            temp = self.label_test[row, :]
            n1 = temp[temp == 1].size
            n2 = temp[temp == 0].size
            total_combine = n1 * n2
            error_combine = 0
            if total_combine != 0:
                for label1 in range(self.nlabel - 1):
                    for label2 in range(label1 + 1, self.nlabel):
                        if (self.label_test[row, label1] > self.label_test[row, label2]) and\
                                (self.label_ranking[row, label1] <= self.label_ranking[row, label2]):
                            error_combine += 1
                        elif (self.label_test[row, label1] < self.label_test[row, label2]) and\
                                (self.label_ranking[row, label1] >= self.label_ranking[row, label2]):
                            error_combine += 1
                self.ranking_loss += (error_combine * 1.0 / total_combine)
        self.ranking_loss /= self.nrow
        print('Ranking Loss:')
        print(self.ranking_loss)
        return self.ranking_loss

    def get_average_precision(self):
        for row in range(self.nrow):
            temp_test = self.label_test[row, :]
            temp_rank = self.label_ranking[row, :]
            n1 = temp_test[temp_test == 1].size
            sorted_index = temp_rank.argsort().getA1().tolist()
            sorted_index.reverse()
            for label in range(self.nlabel):
                if temp_test[0, label] == 1:
                    current_sum = 0
                    for ind in range(self.nlabel):
                        if sorted_index[ind] != label:
                            if temp_test[0, sorted_index[ind]] == 1:
                                current_sum += 1.0
                        else:
                            current_sum += 1.0
                            current_sum /= (ind + 1)
                            break
                    self.average_precision += (current_sum / n1)
        self.average_precision /= (1.0 * self.nrow)
        print('Average Precision:')
        print(self.average_precision)
        return self.average_precision

    def evaluating(self):
        print('Evaluating:')
        self.get_hamming_loss()
        self.get_f1_measure()
        return self.hamming_loss, self.f1_macro, self.f1_micro

    def evaluating_rank(self):
        # Ranking Loss
        print('Evaluating:')
        self.get_ranking_loss()
        self.get_average_precision()
        return self.ranking_loss, self.average_precision


class BinaryRelevance(object):
    # classify with BR strategy. can select feature for transformed binary-classification problems
    # with relief, fisher score and spec. classifier ar ridge, kNN, lasso and SVM. classifier_threshold is
    # the parameter for classifiers. all input data shall be matrix, one sample per row

    # In BR algorithm, a label no instances associated with will not be predicted for an unseen instance, and we will
    # set 0 by default. (such kind of labels are not processed for CLR and CC, so there may be some errors when training
    # data has only 0s for the label)

    def __init__(self, nlabel, select_features=False, fs_method='Relief', feature_threshold=0.2):
        self.nlabel = nlabel
        self.select_features = select_features
        self.fs_method = fs_method
        self.feature_threshold = feature_threshold

        self.feature_list = dict()
        self.classifiers = dict()
        self.label_predict = dict()

        self.hamming_loss = np.zeros((9, 1))
        self.f1_macro = np.zeros((9, 1))
        self.f1_micro = np.zeros((9, 1))
        self.train_time = np.zeros((9, 1))
        self.test_time = np.zeros((9, 1))

    def train(self, feature_train, label_train, clf='Ridge', classifier_threshold=1):
        ncol = feature_train.shape[1]

        # training all features
        if not self.select_features:
            start_time = time.time()
            for label in range(self.nlabel):
                # select classifiers
                if clf == 'kNN':
                    self.classifiers[label] = nn.KNeighborsClassifier(classifier_threshold)
                elif clf == 'LASSO':
                    self.classifiers[label] = lm.Lasso(classifier_threshold)
                elif clf == 'SVM':
                    self.classifiers[label] = svm.SVC()
                elif clf == 'Bayes':
                    self.classifiers[label] = nby.GaussianNB()
                else:
                    self.classifiers[label] = lm.Ridge(classifier_threshold)
                # training
                print('Training Classifier for Label:' + str(label))
                if np.asarray(label_train)[:, label].var(axis=0) == 0:
                    self.classifiers[label] = []
                else:
                    self.classifiers[label].fit(np.asarray(feature_train), np.asarray(label_train)[:, label])
            self.train_time[0, 0] += (time.time() - start_time)
        else:
            # select features
            start_time = time.time()
            for label in range(self.nlabel):
                print('Select Features for Label:' + str(label))
                if self.fs_method == 'FisherScore':
                    self.feature_list[label], scores = fsc.fisher_score(feature_train, label_train[:, label])
                elif self.fs_method == 'Relief':
                    self.feature_list[label], scores = rf.relief(feature_train, label_train[:, label])
                elif self.fs_method == 'SPEC':
                    self.feature_list[label], scores = sfs.spectral_feature_selection(
                        feature_train, label_train[:, label], stype='SingleLabel')
            self.train_time[:, 0] += (time.time() - start_time)

            if self.feature_threshold != 0:
                start_time = time.time()
                for label in range(self.nlabel):
                    # select classifiers
                    if clf == 'kNN':
                        self.classifiers[label] = nn.KNeighborsClassifier(classifier_threshold)
                    elif clf == 'LASSO':
                        self.classifiers[label] = lm.Lasso(classifier_threshold)
                    elif clf == 'SVM':
                        self.classifiers[label] = svm.SVC()
                    elif clf == 'Bayes':
                        self.classifiers[label] = nby.GaussianNB()
                    else:
                        self.classifiers[label] = lm.Ridge(classifier_threshold)
                    # training
                    print('Training Classifier for Label:' + str(label))
                    current_list = self.feature_list[label][range(int(self.feature_threshold * ncol))]
                    self.classifiers[label].fit(np.asarray(feature_train)[:, current_list],
                                                np.asarray(label_train)[:, label])
                self.train_time[:, 0] += (time.time() - start_time)
            else:
                for label in range(self.nlabel):
                    for percent in range(1, 10):
                        start_time = time.time()
                        # select classifiers
                        if clf == 'kNN':
                            self.classifiers[label, percent] = nn.KNeighborsClassifier(classifier_threshold)
                        elif clf == 'LASSO':
                            self.classifiers[label, percent] = lm.Lasso(classifier_threshold)
                        elif clf == 'SVM':
                            self.classifiers[label, percent] = svm.SVC()
                        elif clf == 'Bayes':
                            self.classifiers[label, percent] = nby.GaussianNB()
                        else:
                            self.classifiers[label, percent] = lm.Ridge(classifier_threshold)
                        # training
                        print('Training Classifier for Label:' + str(label) + ' ' + str(percent * 0.1))
                        current_list = self.feature_list[label][range(int(percent * 0.1 * ncol))]
                        self.classifiers[label, percent].fit(np.asarray(feature_train)[:, current_list],
                                                             np.asarray(label_train)[:, label])
                        self.train_time[percent-1, 0] += (time.time() - start_time)
        return self.classifiers

    def test(self, feature_test):
        nrow, ncol = feature_test.shape

        # test with all features
        if not self.select_features:
            for label in range(self.nlabel):
                if self.classifiers[label] != []:
                    print('Testing Label:' + str(label))
                    start_time = time.time()
                    self.label_predict[label] =\
                        np.mat(self.classifiers[label].predict(np.asarray(feature_test))).transpose()
                    self.label_predict[label][self.label_predict[label] >= 0.5] = 1
                    self.label_predict[label][self.label_predict[label] < 0.5] = 0
                    self.test_time[:, 0] += (time.time() - start_time)
                else:
                    self.label_predict[label] = np.mat(np.zeros((nrow, 1)))

        elif self.feature_threshold != 0:
            # test with selected features
            for label in range(self.nlabel):
                print('Testing Label:' + str(label))
                start_time = time.time()
                current_list = self.feature_list[label][range(int(self.feature_threshold * ncol))]
                self.label_predict[label] =\
                    np.mat(self.classifiers[label].predict(np.asarray(feature_test)[:, current_list])).transpose()
                self.label_predict[label][self.label_predict[label] >= 0.5] = 1
                self.label_predict[label][self.label_predict[label] < 0.5] = 0
                self.test_time[:, 0] += (time.time() - start_time)
        else:
            for label in range(self.nlabel):
                for percent in range(1, 10):
                    print('Testing for Label:' + str(label) + ' ' + str(percent * 0.1))
                    start_time = time.time()
                    current_list = self.feature_list[label][range(int(percent * 0.1 * ncol))]
                    self.label_predict[label, percent] = np.mat(self.classifiers[label, percent].predict(
                        np.asarray(feature_test)[:, current_list])).transpose()
                    self.label_predict[label, percent][self.label_predict[label, percent] >= 0.5] = 1
                    self.label_predict[label, percent][self.label_predict[label, percent] < 0.5] = 0
                    self.test_time[percent-1, 0] += (time.time() - start_time)

        return self.label_predict

    def evaluation(self, label_test):
        if not self.select_features or (self.feature_threshold != 0):
            results = self.label_predict[0]
            for label in range(1, self.nlabel):
                results = np.append(results, self.label_predict[label], axis=1)
            self.hamming_loss[0, 0], self.f1_macro[0, 0], self.f1_micro[0, 0] =\
                Evaluation(label_test, results).evaluating()
            return self.hamming_loss[0, 0], self.f1_macro[0, 0], self.f1_micro[0, 0],\
                self.train_time[0, 0], self.test_time[0, 0]
        else:
            for percent in range(1, 10):
                print(str(percent * 10) + '% Features')
                results = self.label_predict[0, percent]
                for label in range(1, self.nlabel):
                    results = np.append(results, self.label_predict[label, percent], axis=1)
                self.hamming_loss[percent-1, 0], self.f1_macro[percent-1, 0], self.f1_micro[percent-1, 0] =\
                    Evaluation(label_test, results).evaluating()
            return self.hamming_loss[:, 0], self.f1_macro[:, 0], self.f1_micro[:, 0],\
                self.train_time[:, 0], self.test_time[:, 0]


class CalibratedLabelRank(object):
    # classify with CLR strategy(This is a label ranking methods),
    # can select feature for transformed binary-classification problems
    # with relief, fisher score and spec. classifier ar ridge, kNN, lasso and SVM. classifier_threshold is
    # the parameter for classifiers. all input data shall be matrix, one sample per row
    def __init__(self, nlabel, select_features=False, fs_method='Relief', feature_threshold=0.2):
        self.nlabel = nlabel
        self.select_features = select_features
        self.fs_method = fs_method
        self.feature_threshold = feature_threshold

        self.feature_list = dict()
        self.classifiers = dict()
        self.pairs = dict()
        self.construct_labels = dict()
        self.label_ranking = dict()

        self.ranking_loss = np.zeros((9, 1))
        self.average_precision = np.zeros((9, 1))
        self.train_time = np.zeros((9, 1))
        self.test_time = np.zeros((9, 1))

    def train(self, feature_train, label_train, clf='Ridge', threshold=1):
        nrow, ncol = feature_train.shape
        # transform problems
        print('Problem Transformation')
        for label1 in range(self.nlabel - 1):
            for label2 in range(label1 + 1, self.nlabel):
                self.pairs[label1, label2] = []
                self.construct_labels[label1, label2] = []
                for row in range(nrow):
                    if (label_train[row, label1] == 1) and (label_train[row, label2] == 0):
                        self.pairs[label1, label2].append(row)
                        self.construct_labels[label1, label2].append(1)
                    elif (label_train[row, label1 == 0]) and (label_train[row, label2] == 1):
                        self.pairs[label1, label2].append(row)
                        self.construct_labels[label1, label2].append(0)
                # at least 10 samples in the new data set and the new data set could not every elements ara 1 or 0
                if (self.pairs[label1, label2].__len__() < 10) or\
                        (np.array(self.construct_labels[label1, label2]).var() == 0):
                    self.pairs[label1, label2] = []
                    self.construct_labels[label1, label2] = []
        # training pairs
        if not self.select_features:
            for label1 in range(self.nlabel - 1):
                for label2 in range(label1 + 1, self.nlabel):
                    if not self.pairs[label1, label2] == []:
                        if clf == 'kNN':
                            self.classifiers[label1, label2] = nn.KNeighborsClassifier(threshold)
                        elif clf == 'LASSO':
                            self.classifiers[label1, label2] = lm.Lasso(threshold)
                        elif clf == 'SVM':
                            self.classifiers[label1, label2] = svm.SVC()
                        else:
                            self.classifiers[label1, label2] = lm.Ridge(threshold)

                        start_time = time.time()
                        # training
                        print('Training Classifier for Label Pair:' + str((label1, label2)))
                        self.classifiers[label1, label2].fit(np.asarray(feature_train[self.pairs[label1, label2], :]),
                                                             np.asarray(self.construct_labels[label1, label2]))
                        self.train_time[:, 0] += (time.time() - start_time)
        else:
            # select features
            start_time = time.time()
            for label1 in range(self.nlabel - 1):
                for label2 in range(label1 + 1, self.nlabel):
                    if not self.pairs[label1, label2] == []:
                        print('Select Features for Label:' + str((label1, label2)))
                        if self.fs_method == 'Relief':
                            self.feature_list[label1, label2], score =\
                                rf.relief(feature_train[self.pairs[label1, label2], :],
                                          np.mat(self.construct_labels[label1, label2]).transpose())
                        elif self.fs_method == 'FisherScore':
                            self.feature_list[label1, label2], score =\
                                fsc.fisher_score(feature_train[self.pairs[label1, label2], :],
                                                 np.mat(self.construct_labels[label1, label2]).transpose())
                        elif self.fs_method == 'SPEC':
                            self.feature_list[label1, label2], score =\
                                sfs.spectral_feature_selection(feature_train[self.pairs[label1, label2], :],
                                                               np.mat(self.construct_labels[label1, label2]).transpose(),
                                                               stype='SingleLabel')
            self.train_time[:, 0] += (time.time() - start_time)
            if self.feature_threshold != 0:
                start_time = time.time()
                for label1 in range(self.nlabel - 1):
                    for label2 in range(label1 + 1, self.nlabel):
                        if not self.pairs[label1, label2] == []:
                            # select classifiers
                            if clf == 'kNN':
                                self.classifiers[label1, label2] = nn.KNeighborsClassifier(threshold)
                            elif clf == 'LASSO':
                                self.classifiers[label1, label2] = lm.Lasso(threshold)
                            elif clf == 'SVM':
                                self.classifiers[label1, label2] = svm.SVC()
                            else:
                                self.classifiers[label1, label2] = lm.Ridge(threshold)

                            # training
                            print('Training Classifier for Label:' + str((label1, label2)))
                            current_list = self.feature_list[label1, label2][range(int(self.feature_threshold * ncol))]
                            self.classifiers[label1, label2].fit(
                                np.asarray(feature_train[self.pairs[label1, label2], :][:, current_list]),
                                np.asarray(self.construct_labels[label1, label2]))
                self.train_time[:, 0] += (time.time() - start_time)
            else:
                for label1 in range(self.nlabel - 1):
                    for label2 in range(label1 + 1, self.nlabel):
                        if not self.pairs[label1, label2] == []:
                            for percent in range(1, 10):
                                # select classifiers
                                if clf == 'kNN':
                                    self.classifiers[label1, label2, percent] = nn.KNeighborsClassifier(threshold)
                                elif clf == 'LASSO':
                                    self.classifiers[label1, label2, percent] = lm.Lasso(threshold)
                                elif clf == 'SVM':
                                    self.classifiers[label1, label2, percent] = svm.SVC()
                                else:
                                    self.classifiers[label1, label2, percent] = lm.Ridge(threshold)
                                # training
                                print('Training Classifier for Label:' + str((label1, label2)) + ' ' + str(percent * 0.1))
                                start_time = time.time()
                                current_list = self.feature_list[label1, label2][range(int(percent * 0.1 * ncol))]
                                self.classifiers[label1, label2, percent].fit(
                                    np.asarray(feature_train[self.pairs[label1, label2], :][:, current_list]),
                                    np.asarray(self.construct_labels[label1, label2]))
                                self.train_time[percent - 1, 0] += (time.time() - start_time)
        return self.classifiers

    def test(self, feature_test):
        nrow, ncol = feature_test.shape
        # predict with all features
        if not self.select_features:
            self.label_ranking[0] = np.mat(np.zeros((nrow, self.nlabel)))
            # testing pairs
            for label1 in range(self.nlabel - 1):
                for label2 in range(label1 + 1, self.nlabel):
                    if self.classifiers.get((label1, label2), -1) != -1:
                        # testing
                        start_time = time.time()
                        print('Testing Label Pair:' + str((label1, label2)))
                        pair_results = self.classifiers[label1, label2].predict(np.asarray(feature_test))
                        self.test_time[:, 0] += (time.time() - start_time)
                        pair_results = np.array(pair_results)
                        pair_results[pair_results >= 0.5] = 1
                        pair_results[pair_results < 0.5] = 0
                        # voting
                        for row in range(nrow):
                            if pair_results[row] == 1:
                                self.label_ranking[0][row, label1] += 1
                            else:
                                self.label_ranking[0][row, label2] += 1
        elif self.feature_threshold != 0:
            # predict with specified features
            self.label_ranking[0] = np.mat(np.zeros((nrow, self.nlabel)))
            # testing pairs
            for label1 in range(self.nlabel - 1):
                for label2 in range(label1 + 1, self.nlabel):
                    if self.classifiers.get((label1, label2), -1) != -1:
                        current_list = self.feature_list[label1, label2][range(int(self.feature_threshold * ncol))]
                        # testing
                        start_time = time.time()
                        print('Testing Label Pair:' + str((label1, label2)))
                        pair_results = self.classifiers[label1, label2].predict(
                            np.asarray(feature_test[:, current_list]))
                        self.test_time[:, 0] += (time.time() - start_time)
                        pair_results = np.array(pair_results)
                        pair_results[pair_results >= 0.5] = 1
                        pair_results[pair_results < 0.5] = 0
                        # voting
                        for row in range(nrow):
                            if pair_results[row] == 1:
                                self.label_ranking[0][row, label1] += 1
                            else:
                                self.label_ranking[0][row, label2] += 1
        else:
            for percent in range(1, 10):
                self.label_ranking[percent] = np.mat(np.zeros((nrow, self.nlabel)))
                # testing pairs
                for label1 in range(self.nlabel - 1):
                    for label2 in range(label1 + 1, self.nlabel):
                        if self.classifiers.get((label1, label2, percent), -1) != -1:
                            current_list = self.feature_list[label1, label2][range(int(percent * 0.1 * ncol))]
                            # testing
                            start_time = time.time()
                            print('Testing Label Pair:' + str((label1, label2)) + ' ' + str(percent * 0.1))
                            pair_results = self.classifiers[label1, label2, percent].predict(
                                 np.asarray(feature_test[:, current_list]))
                            self.test_time[percent - 1, 0] += (time.time() - start_time)
                            pair_results = np.array(pair_results)
                            pair_results[pair_results >= 0.5] = 1
                            pair_results[pair_results < 0.5] = 0
                            # voting
                            for row in range(nrow):
                                if pair_results[row] == 1:
                                    self.label_ranking[percent][row, label1] += 1
                                else:
                                    self.label_ranking[percent][row, label2] += 1
        return self.label_ranking

    def evaluation(self, label_test):
        if not self.select_features or (self.feature_threshold != 0):

            results = self.label_ranking[0]
            self.ranking_loss[0, 0], self.average_precision[0, 0] =\
                Evaluation(label_test, label_ranking=results).evaluating_rank()
            return self.ranking_loss[0, 0], self.average_precision[0, 0], self.train_time[0, 0], self.test_time[0, 0]
        else:
            for percent in range(1, 10):
                print(str(percent * 10) + '% Features')
                results = self.label_ranking[percent]
                self.ranking_loss[percent - 1, 0], self.average_precision[percent - 1, 0] =\
                    Evaluation(label_test, label_ranking=results).evaluating_rank()
            return self.ranking_loss[:, 0], self.average_precision[:, 0], self.train_time[:, 0], self.test_time[:, 0]


class ClassifierChains(object):
    # classify with CC strategy, can select feature for transformed binary-classification problems
    # with relief, fisher score and spec. classifier ar ridge, kNN, lasso and SVM. classifier_threshold is
    # the parameter for classifiers. rseed is the random seed for shuffling the labels
    # all input data shall be matrix, one sample per row
    def __init__(self, nlabel, select_features=False, fs_method='Relief', feature_threshold=0.2, rseed=1):
        self.nlabel = nlabel
        self.select_features = select_features
        self.fs_method = fs_method
        self.feature_threshold = feature_threshold

        self.sequence = range(nlabel)
        np.random.RandomState(rseed).shuffle(self.sequence)
        self.feature_list = dict()
        self.classifiers = dict()
        self.label_predict = dict()

        self.hamming_loss = np.zeros((9, 1))
        self.f1_macro = np.zeros((9, 1))
        self.f1_micro = np.zeros((9, 1))
        self.train_time = np.zeros((9, 1))
        self.test_time = np.zeros((9, 1))
        self.train_time = np.zeros((9, 1))
        self.test_time = np.zeros((9, 1))

    def train(self, feature_train, label_train, clf='Ridge', classifier_threshold=1):
        ncol = feature_train.shape[1]
        extended_feature = feature_train.copy()

        if not self.select_features:
            for index in range(self.nlabel):
                label = self.sequence[index]
                # select classifier
                if clf == 'kNN':
                    self.classifiers[label] = nn.KNeighborsClassifier(classifier_threshold)
                elif clf == 'LASSO':
                    self.classifiers[label] = lm.Lasso(classifier_threshold)
                elif clf == 'SVM':
                    self.classifiers[label] = svm.SVC()
                else:
                    self.classifiers[label] = lm.Ridge(classifier_threshold)
                start_time = time.time()
                print('Training Classifier for Label:' + str(label))
                self.classifiers[label].fit(np.asarray(extended_feature), np.asarray(label_train)[:, label])
                self.train_time[0, 0] += (time.time() - start_time)
                extended_feature = np.append(extended_feature, label_train[:, label], axis=1)
        else:
            # select features
            start_time = time.time()
            for index in range(self.nlabel):
                label = self.sequence[index]
                print('Select Features for Label:' + str(label))
                if self.fs_method == 'Relief':
                    self.feature_list[label], scores = rf.relief(extended_feature, label_train[:, label])
                elif self.fs_method == 'FisherScore':
                    self.feature_list[label], scores = fsc.fisher_score(extended_feature, label_train[:, label])
                elif self.fs_method == 'SPEC':
                    self.feature_list[label], scores = sfs.spectral_feature_selection(
                        extended_feature, label_train[:, label], stype='SingleLabel')
                extended_feature = np.append(extended_feature, label_train[:, label], axis=1)
            self.train_time[:, 0] += (time.time() - start_time)

            if self.feature_threshold != 0:
                start_time = time.time()
                for index in range(self.nlabel):
                    label = self.sequence[index]
                    # select classifiers
                    if clf == 'kNN':
                        self.classifiers[label] = nn.KNeighborsClassifier(classifier_threshold)
                    elif clf == 'LASSO':
                        self.classifiers[label] = lm.Lasso(classifier_threshold)
                    elif clf == 'SVM':
                        self.classifiers[label] = svm.SVC()
                    else:
                        self.classifiers[label] = lm.Ridge(classifier_threshold)
                    # training
                    print('Training Classifier for Label:' + str(label))
                    current_list = self.feature_list[label][range(int(self.feature_threshold * (ncol + index)))]
                    self.classifiers[label].fit(np.asarray(extended_feature)[:, current_list],
                                                np.asarray(label_train)[:, label])
                self.train_time[:, 0] += (time.time() - start_time)
            else:
                for percent in range(1, 10):
                    for index in range(self.nlabel):
                        label = self.sequence[index]
                        start_time = time.time()
                        # select classifiers
                        if clf == 'kNN':
                            self.classifiers[label, percent] = nn.KNeighborsClassifier(classifier_threshold)
                        elif clf == 'LASSO':
                            self.classifiers[label, percent] = lm.Lasso(classifier_threshold)
                        elif clf == 'SVM':
                            self.classifiers[label, percent] = svm.SVC()
                        else:
                            self.classifiers[label, percent] = lm.Ridge(classifier_threshold)
                        # training
                        print('Training Classifier for Label:' + str(label) + ' ' + str(percent * 0.1))
                        current_list = self.feature_list[label][range(int(percent * 0.1 * (ncol + index)))]
                        self.classifiers[label, percent].fit(np.asarray(extended_feature)[:, current_list],
                                                             np.asarray(label_train)[:, label])
                        self.train_time[percent-1, 0] += (time.time() - start_time)
        return self.classifiers

    def test(self, feature_test):
        ncol = feature_test.shape[1]
        extended_feature = feature_test.copy()

        # test with all features
        if not self.select_features:
            for index in range(self.nlabel):
                label = self.sequence[index]
                print('Testing Label:' + str(label))
                start_time = time.time()
                self.label_predict[label] =\
                    np.mat(self.classifiers[label].predict(np.asarray(extended_feature))).transpose()
                self.label_predict[label][self.label_predict[label] >= 0.5] = 1
                self.label_predict[label][self.label_predict[label] < 0.5] = 0
                self.test_time[:, 0] += (time.time() - start_time)
                extended_feature = np.append(extended_feature, self.label_predict[label], axis=1)
        elif self.feature_threshold != 0:
            # test with selected features
            for index in range(self.nlabel):
                label = self.sequence[index]
                print('Testing Label:' + str(label))
                start_time = time.time()
                current_list = self.feature_list[label][range(int(self.feature_threshold * (ncol + index)))]
                self.label_predict[label] =\
                    np.mat(self.classifiers[label].predict(np.asarray(extended_feature)[:, current_list])).transpose()
                self.label_predict[label][self.label_predict[label] >= 0.5] = 1
                self.label_predict[label][self.label_predict[label] < 0.5] = 0
                self.test_time[:, 0] += (time.time() - start_time)
                extended_feature = np.append(extended_feature, self.label_predict[label], axis=1)
        else:
            for percent in range(1, 10):
                extended_feature = feature_test.copy()
                for index in range(self.nlabel):
                    label = self.sequence[index]
                    print('Testing for Label:' + str(label) + ' ' + str(percent * 0.1))
                    start_time = time.time()
                    current_list = self.feature_list[label][range(int(percent * 0.1 * (ncol + index)))]
                    self.label_predict[label, percent] = np.mat(self.classifiers[label, percent].predict(
                        np.asarray(extended_feature)[:, current_list])).transpose()
                    self.label_predict[label, percent][self.label_predict[label, percent] >= 0.5] = 1
                    self.label_predict[label, percent][self.label_predict[label, percent] < 0.5] = 0
                    self.test_time[percent-1, 0] += (time.time() - start_time)
                    extended_feature = np.append(extended_feature, self.label_predict[label, percent], axis=1)
        return self.label_predict

    def evaluation(self, label_test):
        if not self.select_features or (self.feature_threshold != 0):
            results = self.label_predict[0]
            for label in range(1, self.nlabel):
                results = np.append(results, self.label_predict[label], axis=1)
            self.hamming_loss[0, 0], self.f1_macro[0, 0], self.f1_micro[0, 0] =\
                Evaluation(label_test, label_predict=results).evaluating()
            return self.hamming_loss[0, 0], self.f1_macro[0, 0], self.f1_micro[0, 0],\
                self.train_time[0, 0], self.test_time[0, 0]
        else:
            for percent in range(1, 10):
                print(str(percent * 10) + '% Features')
                results = self.label_predict[0, percent]
                for label in range(1, self.nlabel):
                    results = np.append(results, self.label_predict[label, percent], axis=1)
                self.hamming_loss[percent-1, 0], self.f1_macro[percent-1, 0], self.f1_micro[percent-1, 0] =\
                    Evaluation(label_test, label_predict=results).evaluating()
            return self.hamming_loss[:, 0], self.f1_macro[:, 0], self.f1_micro[:, 0],\
                self.train_time[:, 0], self.test_time[:, 0]


if __name__ == '__main__':
    print('Loading Data:')
    feature_train = ld.import_matrix('CrossValidation/emotions/emotions_feature_train_0.csv')
    feature_test = ld.import_matrix('CrossValidation/emotions/emotions_feature_test_0.csv')
    label_train = ld.import_matrix('CrossValidation/emotions/emotions_label_train_0.csv')
    label_test = ld.import_matrix('CrossValidation/emotions/emotions_label_test_0.csv')
    c = BinaryRelevance(label_train.shape[1], select_features=True, fs_method='FisherScore')
    c.train(feature_train, label_train, clf='kNN', classifier_threshold=3)
    c.test(feature_test)
    c.evaluation(label_test)
    c = CalibratedLabelRank(label_train.shape[1], select_features=True, fs_method='FisherScore')
    c.train(feature_train, label_train, clf='kNN', threshold=3)
    c.test(feature_test)
    c.evaluation(label_test)
    c = ClassifierChains(label_train.shape[1], select_features=True, fs_method='FisherScore', feature_threshold=0)
    c.train(feature_train, label_train, clf='kNN', classifier_threshold=3)
    c.test(feature_test)
    c.evaluation(label_test)
