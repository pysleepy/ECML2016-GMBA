import os as os
import time as time
import numpy as np
import matplotlib.pyplot as plt
import LoadingData as ld
import CrossValidationFilter as filter
import CrossValidationWrapper as wrapper


__author__ = 'PengYan'


def test(data_set_name, mlc, test_name, need_shuffling=False, need_sampling=False, need_normalize=True,
         need_contaminate=False, algorithm_list_filter=[], algorithm_list_wrapper=[]):
    # run cross validations for specified methods(list in algorithm_list_filter and algorithm_list_wrapper).
    # mlc denotes transform strategy, test_name specified which directory to save the results. need_shuffling for first
    # time running the test

    hamming_loss = dict()
    f1_macro = dict()
    f1_micro = dict()

    ranking_loss = dict()
    average_precision = dict()

    train_time = dict()
    test_time = dict()

    if algorithm_list_filter == [] and algorithm_list_wrapper == []:
        algorithm_list_filter = ['MLFS', 'MLRF', 'SPEC', 'ML_LMBA', 'all_features']
        algorithm_list_wrapper = ['FisherScore', 'Relief']
    # algorithm_list_wrapper = ['FisherScore', 'Relief', 'SPEC']

    dir_name = test_name + '/' + data_set_name + '/' + mlc

    # check directory
    if not os.path.exists(test_name):
        print('Directory: ' + test_name + ' Does Not Exist')
        try:
            os.mkdir(test_name)
            print('Directory Made')
        except Exception, e:
            print(e)

    if not os.path.exists(test_name + '/' + data_set_name):
        print('Directory: ' + data_set_name + ' Does Not Exist')
        try:
            os.mkdir(test_name + '/' + data_set_name)
            print('Directory Made')
        except Exception, e:
            print(e)

    if not os.path.exists(test_name + '/' + data_set_name + '/' + mlc):
        print('Directory: ' + data_set_name + '/' + mlc + ' Does Not Exist')
        try:
            os.mkdir( test_name + '/' + data_set_name + '/' + mlc)
            print('Directory Made')
        except Exception, e:
            print(e)

    if need_shuffling:
        ld.generate_validate_data(data_name=data_set_name, nflod=10, title_line=True)

    for algorithm in algorithm_list_filter:
        if mlc != 'CLR':
            hamming_loss[algorithm], f1_macro[algorithm], f1_micro[algorithm], train_time[algorithm], test_time[algorithm] =\
                filter.cross_validation_filter(
                    data_name=data_set_name, mlc_type=mlc, compare_with=algorithm, feature_threshold=0,
                    need_normalize=need_normalize, need_sampling=need_sampling, need_contaminate=need_contaminate)
            np.save(dir_name + '/hamming_loss_' + algorithm, hamming_loss[algorithm])
            np.save(dir_name + '/f1_macro_' + algorithm, f1_macro[algorithm])
            np.save(dir_name + '/f1_micro_' + algorithm, f1_micro[algorithm])
            np.save(dir_name + '/train_time_' + algorithm, train_time[algorithm])
            np.save(dir_name + '/test_time_' + algorithm, test_time[algorithm])
        else:
            ranking_loss[algorithm], average_precision[algorithm], train_time[algorithm], test_time[algorithm] =\
                filter.cross_validation_filter(
                    data_name=data_set_name, mlc_type=mlc, compare_with=algorithm, feature_threshold=0,
                    need_normalize=need_normalize, need_sampling=need_sampling, need_contaminate=need_contaminate)
            np.save(dir_name + '/ranking_loss_' + algorithm, ranking_loss[algorithm])
            np.save(dir_name + '/average_precision_' + algorithm, average_precision[algorithm])
            np.save(dir_name + '/train_time_' + algorithm, train_time[algorithm])
            np.save(dir_name + '/test_time_' + algorithm, test_time[algorithm])

    for algorithm in algorithm_list_wrapper:
        if mlc != 'CLR':
            hamming_loss[algorithm + '_wrapper'] = np.zeros((9, 10))
            f1_macro[algorithm + '_wrapper'] = np.zeros((9, 10))
            f1_micro[algorithm + '_wrapper'] = np.zeros((9, 10))
            train_time[algorithm + '_wrapper'] = np.zeros((9, 10))
            test_time[algorithm + '_wrapper'] = np.zeros((9, 10))
            for percent in range(1, 10):
                a, b, c, d, e = wrapper.cross_validation_wrapper(
                    data_name=data_set_name, mlc_type=mlc, compare_with=algorithm, feature_threshold=(percent * 0.1),
                    need_normalize=need_normalize, need_sampling=need_sampling, need_contaminate=need_contaminate)
                hamming_loss[algorithm + '_wrapper'][percent-1, :], f1_macro[algorithm + '_wrapper'][percent-1, :],\
                f1_micro[algorithm + '_wrapper'][percent-1, :], train_time[algorithm + '_wrapper'][percent-1, :],\
                test_time[algorithm + '_wrapper'][percent-1, :] = a[percent-1, :], b[percent-1, :], c[percent-1, :],\
                                                                  d[percent-1, :], e[percent-1, :]

            np.save(dir_name + '/hamming_loss_' + algorithm + '_wrapper', hamming_loss[algorithm + '_wrapper'])
            np.save(dir_name + '/f1_macro_' + algorithm + '_wrapper', f1_macro[algorithm + '_wrapper'])
            np.save(dir_name + '/f1_micro_' + algorithm + '_wrapper', f1_micro[algorithm + '_wrapper'])
            np.save(dir_name + '/train_time_' + algorithm + '_wrapper', train_time[algorithm + '_wrapper'])
            np.save(dir_name + '/test_time_' + algorithm + '_wrapper', test_time[algorithm + '_wrapper'])
        else:
            ranking_loss[algorithm + '_wrapper'] = np.zeros((9, 10))
            average_precision[algorithm + '_wrapper'] = np.zeros((9, 10))
            train_time[algorithm + '_wrapper'] = np.zeros((9, 10))
            test_time[algorithm + '_wrapper'] = np.zeros((9, 10))
            for percent in range(1, 10):
                a, b, c, d = wrapper.cross_validation_wrapper(
                    data_name=data_set_name, mlc_type=mlc, compare_with=algorithm, feature_threshold=(percent * 0.1),
                    need_normalize=need_normalize, need_sampling=need_sampling, need_contaminate=need_contaminate)
                ranking_loss[algorithm + '_wrapper'][percent-1, :],\
                average_precision[algorithm + '_wrapper'][percent-1, :],\
                train_time[algorithm + '_wrapper'][percent-1, :],\
                test_time[algorithm + '_wrapper'][percent-1, :] = a[percent-1, :], b[percent-1, :], c[percent-1, :],\
                                                                  d[percent-1, :]

            np.save(dir_name + '/ranking_loss_' + algorithm + '_wrapper', ranking_loss[algorithm + '_wrapper'])
            np.save(dir_name + '/average_precision_' + algorithm + '_wrapper', average_precision[algorithm + '_wrapper'])
            np.save(dir_name + '/train_time_' + algorithm + '_wrapper', train_time[algorithm + '_wrapper'])
            np.save(dir_name + '/test_time_' + algorithm + '_wrapper', test_time[algorithm + '_wrapper'])

    return load_results(data_set_name, mlc, test_name, algorithm_list_filter, algorithm_list_wrapper)
    #  return hamming_loss, f1_macro, f1_micro, ranking_loss, average_precision, train_time, test_time


def load_results(data_set_name, mlc, test_name, algorithm_list_filter=[], algorithm_list_wrapper=[]):
    # loading results from test() and show the plot
    hamming_loss = dict()
    f1_macro = dict()
    f1_micro = dict()

    ranking_loss = dict()
    average_precision = dict()

    train_time = dict()
    test_time = dict()

    leg = dict()
    leg['MLFS'] = ['>', 'orange']
    leg['MLRF'] = ['v', 'green']
    leg['SPEC'] = ['s', 'blue']
    leg['ML_LMBA'] = ['o', 'red']
    leg['FisherScore'] = ['<', 'brown']
    leg['Relief'] = ['*', 'purple']
    leg['all_features'] = ['.', 'black']
    x = range(10, 100, 10)

    dir_name = test_name + '/' + data_set_name + '/' + mlc

    # algorithm_list_filter = ['ML_LMBA_tuned']
    # loading results before 2016-02-03, use this.
    # To load results later than the date, replace 'Base' with 'all_features'
    if algorithm_list_filter == [] and algorithm_list_wrapper == []:
        # algorithm_list_filter = []
        algorithm_list_filter = ['MLFS', 'MLRF', 'SPEC', 'ML_LMBA', 'all_features']
        algorithm_list_wrapper = ['FisherScore', 'Relief']
        # algorithm_list_wrapper = ['FisherScore', 'Relief', 'SPEC']

    for algorithm in algorithm_list_filter:
        if mlc != 'CLR':
            hamming_loss[algorithm] = np.load(dir_name + '/hamming_loss_' + algorithm + '.npy')
            f1_macro[algorithm] = np.load(dir_name + '/f1_macro_' + algorithm + '.npy')
            f1_micro[algorithm] = np.load(dir_name + '/f1_micro_' + algorithm + '.npy')
            train_time[algorithm] = np.load(dir_name + '/train_time_' + algorithm + '.npy')
            test_time[algorithm] = np.load(dir_name + '/test_time_' + algorithm + '.npy')
        else:
            ranking_loss[algorithm] = np.load(dir_name + '/ranking_loss_' + algorithm + '.npy')
            average_precision[algorithm] = np.load(dir_name + '/average_precision_' + algorithm + '.npy')
            train_time[algorithm] = np.load(dir_name + '/train_time_' + algorithm + '.npy')
            test_time[algorithm] = np.load(dir_name + '/test_time_' + algorithm + '.npy')

    for algorithm in algorithm_list_wrapper:
        if mlc != 'CLR':
            hamming_loss[algorithm + '_wrapper'] = np.load(dir_name + '/hamming_loss_' + algorithm + '_wrapper' + '.npy')
            f1_macro[algorithm + '_wrapper'] = np.load(dir_name + '/f1_macro_' + algorithm + '_wrapper' + '.npy')
            f1_micro[algorithm + '_wrapper'] = np.load(dir_name + '/f1_micro_' + algorithm + '_wrapper' + '.npy')
            train_time[algorithm + '_wrapper'] = np.load(dir_name + '/train_time_' + algorithm + '_wrapper' + '.npy')
            test_time[algorithm + '_wrapper'] = np.load(dir_name + '/test_time_' + algorithm + '_wrapper' + '.npy')
        else:
            ranking_loss[algorithm + '_wrapper'] = np.load(dir_name + '/ranking_loss_' + algorithm + '_wrapper' + '.npy')
            average_precision[algorithm + '_wrapper'] = np.load(dir_name + '/average_precision_' +
                                                                algorithm + '_wrapper' + '.npy')
            train_time[algorithm + '_wrapper'] = np.load(dir_name + '/train_time_' + algorithm + '_wrapper' + '.npy')
            test_time[algorithm + '_wrapper'] = np.load(dir_name + '/test_time_' + algorithm + '_wrapper' + '.npy')

    if mlc != 'CLR':
        figure_1 = plt.figure(1)
        figure_1.suptitle('Hamming Loss: ' + data_set_name)
        figure_1.hold(True)
        plt.xlim(5, 95)
        plt.xticks(x)
        for algorithm in algorithm_list_filter:
            plt.plot(x, hamming_loss[algorithm].mean(axis=1), linewidth=2.0, linestyle="-", c=leg[algorithm][1])
            plt.scatter(x, hamming_loss[algorithm].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm)
        for algorithm in algorithm_list_wrapper:
            plt.plot(x, hamming_loss[algorithm + '_wrapper'].mean(axis=1), linewidth=2.0, linestyle="-", c=leg[algorithm][1])
            plt.scatter(x, hamming_loss[algorithm + '_wrapper'].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm + '_wrapper')
        handles, labels = plt.axes().get_legend_handles_labels()
        plt.legend(handles, labels, loc=0)

        figure_2 = plt.figure(2)
        figure_2.suptitle('F1_Macro: ' + data_set_name)
        figure_2.hold(True)
        plt.xlim(5, 95)
        plt.xticks(x)
        for algorithm in algorithm_list_filter:
            plt.plot(x, f1_macro[algorithm].mean(axis=1), linewidth=2.0, linestyle="-", c=leg[algorithm][1])
            plt.scatter(x, f1_macro[algorithm].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm)
        for algorithm in algorithm_list_wrapper:
            plt.plot(x, f1_macro[algorithm + '_wrapper'].mean(axis=1), linewidth=2.0, linestyle="-", c=leg[algorithm][1])
            plt.scatter(x, f1_macro[algorithm + '_wrapper'].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm + '_wrapper')
        handles, labels = plt.axes().get_legend_handles_labels()
        plt.legend(handles, labels, loc=0)

        figure_3 = plt.figure(3)
        figure_3.suptitle('F1_Micro: ' + data_set_name )
        figure_3.hold(True)
        plt.xlim(5, 95)
        plt.xticks(x)
        for algorithm in algorithm_list_filter:
            plt.plot(x, f1_micro[algorithm].mean(axis=1), linewidth=2.0, linestyle="-", c=leg[algorithm][1])
            plt.scatter(x, f1_micro[algorithm].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm)
        for algorithm in algorithm_list_wrapper:
            plt.plot(x, f1_micro[algorithm + '_wrapper'].mean(axis=1), linewidth=2.0, linestyle="-", c=leg[algorithm][1])
            plt.scatter(x, f1_micro[algorithm + '_wrapper'].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm + '_wrapper')
        handles, labels = plt.axes().get_legend_handles_labels()
        plt.legend(handles, labels, loc=0)

        plt.xlim(0, 100)
        figure_4 = plt.figure(4)
        figure_4.suptitle('training time: ' + data_set_name )
        figure_4.hold(True)
        plt.xlim(5, 95)
        plt.xticks(x)
        for algorithm in algorithm_list_filter:
            plt.plot(x, train_time[algorithm].mean(axis=1), linewidth=2.0, linestyle="-", c=leg[algorithm][1])
            plt.scatter(x, train_time[algorithm].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm)
        for algorithm in algorithm_list_wrapper:
            plt.plot(x, train_time[algorithm + '_wrapper'].mean(axis=1), linewidth=2.0, linestyle="-", c=leg[algorithm][1])
            plt.scatter(x, train_time[algorithm + '_wrapper'].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm + '_wrapper')
        handles, labels = plt.axes().get_legend_handles_labels()
        plt.legend(handles, labels, loc=0)

        plt.xlim(0, 100)
        figure_5 = plt.figure(5)
        figure_5.suptitle('testing time: ' + data_set_name )
        figure_5.hold(True)
        plt.xlim(5, 95)
        plt.xticks(x)
        for algorithm in algorithm_list_filter:
            plt.plot(x, test_time[algorithm].mean(axis=1), linewidth=2.0, linestyle="-", c=leg[algorithm][1])
            plt.scatter(x, test_time[algorithm].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm)
        for algorithm in algorithm_list_wrapper:
            plt.plot(x, test_time[algorithm + '_wrapper'].mean(axis=1), linewidth=2.0, linestyle="-", c=leg[algorithm][1])
            plt.scatter(x, test_time[algorithm + '_wrapper'].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm + '_wrapper')
        handles, labels = plt.axes().get_legend_handles_labels()
        plt.legend(handles, labels, loc=0)

        plt.figure(1).savefig(dir_name + '/hamming_loss')
        plt.figure(2).savefig(dir_name + '/f1_macro')
        plt.figure(3).savefig(dir_name + '/f1_micro')
        plt.figure(4).savefig(dir_name + '/train_time')
        plt.figure(5).savefig(dir_name + '/test_time')

        plt.show()
        return hamming_loss, f1_macro, f1_micro, train_time, test_name

    else:

        figure_1 = plt.figure(1)
        figure_1.suptitle('Ranking Loss: ' + data_set_name)
        figure_1.hold(True)
        plt.xlim(5, 95)
        plt.xticks(x)
        for algorithm in algorithm_list_filter:
            plt.plot(x, ranking_loss[algorithm].mean(axis=1), linewidth=2.0, c=leg[algorithm][1], linestyle="-")
            plt.scatter(x, ranking_loss[algorithm].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm)
        for algorithm in algorithm_list_wrapper:
            plt.plot(x, ranking_loss[algorithm + '_wrapper'].mean(axis=1), linewidth=2.0, linestyle="-", c=leg[algorithm][1])
            plt.scatter(x, ranking_loss[algorithm + '_wrapper'].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm + '_wrapper')
        handles, labels = plt.axes().get_legend_handles_labels()
        plt.legend(handles, labels, loc=0)

        figure_2 = plt.figure(2)
        figure_2.suptitle('Average Precision: ' + data_set_name)
        figure_2.hold(True)
        plt.xlim(5, 95)
        plt.xticks(x)
        for algorithm in algorithm_list_filter:
            plt.plot(x, average_precision[algorithm].mean(axis=1), linewidth=2.0, c=leg[algorithm][1], linestyle="-")
            plt.scatter(x, average_precision[algorithm].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm)
        for algorithm in algorithm_list_wrapper:
            plt.plot(x, average_precision[algorithm + '_wrapper'].mean(axis=1), linewidth=2.0, linestyle="-", c=leg[algorithm][1])
            plt.scatter(x, average_precision[algorithm + '_wrapper'].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm + '_wrapper')
        handles, labels = plt.axes().get_legend_handles_labels()
        plt.legend(handles, labels, loc=0)

        plt.xlim(0, 100)
        figure_3 = plt.figure(3)
        figure_3.suptitle('training time: ' + data_set_name )
        figure_3.hold(True)
        plt.xlim(5, 95)
        plt.xticks(x)
        for algorithm in algorithm_list_filter:
            plt.plot(x, train_time[algorithm].mean(axis=1), linewidth=2.0, linestyle="-", c=leg[algorithm][1])
            plt.scatter(x, train_time[algorithm].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm)
        for algorithm in algorithm_list_wrapper:
            plt.plot(x, train_time[algorithm + '_wrapper'].mean(axis=1), linewidth=2.0, linestyle="-", c=leg[algorithm][1])
            plt.scatter(x, train_time[algorithm + '_wrapper'].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm + '_wrapper')
        handles, labels = plt.axes().get_legend_handles_labels()
        plt.legend(handles, labels, loc=0)

        plt.xlim(0, 100)
        figure_4 = plt.figure(4)
        figure_4.suptitle('testing time: ' + data_set_name )
        figure_4.hold(True)
        plt.xlim(5, 95)
        plt.xticks(x)
        for algorithm in algorithm_list_filter:
            plt.plot(x, test_time[algorithm].mean(axis=1), linewidth=2.0, linestyle="-", c=leg[algorithm][1])
            plt.scatter(x, test_time[algorithm].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm)
        for algorithm in algorithm_list_wrapper:
            plt.plot(x, test_time[algorithm + '_wrapper'].mean(axis=1), linewidth=2.0, linestyle="-", c=leg[algorithm][1])
            plt.scatter(x, test_time[algorithm + '_wrapper'].mean(axis=1), marker=leg[algorithm][0], c=leg[algorithm][1], s=50, label=algorithm + '_wrapper')
        handles, labels = plt.axes().get_legend_handles_labels()
        plt.legend(handles, labels, loc=0)

        plt.figure(1).savefig(dir_name + '/ranking_loss')
        plt.figure(2).savefig(dir_name + '/average_precision')
        plt.figure(3).savefig(dir_name + '/train_time')
        plt.figure(4).savefig(dir_name + '/test_time')

        plt.show()
        return ranking_loss, average_precision, train_time, test_time

if __name__ == '__main__':
    data_name = 'emotions'
    test_name = 'temp'
    need_normalize = True
    need_sampling = False
    need_contaminate = False
    algorithm_list_filter = []
    algorithm_list_filter = ['MLFS', 'MLRF', 'SPEC', 'ML_LMBA', 'all_features']
    algorithm_list_wrapper = []
    algorithm_list_wrapper = ['FisherScore', 'Relief']

    if need_normalize:
        test_name += '_normalized'
    else:
        test_name += '_unnormalized'

    test(data_name, 'BR', test_name, need_normalize=need_normalize, need_sampling=need_sampling,
         algorithm_list_filter=algorithm_list_filter, algorithm_list_wrapper=algorithm_list_wrapper,
         need_contaminate=need_contaminate)
    # test(data_name, 'CLR', test_name, need_normalize=need_normalize, need_sampling=need_sampling,
    #      algorithm_list_filter=algorithm_list_filter, algorithm_list_wrapper=algorithm_list_wrapper)
    # test(data_name, 'CC', test_name, need_normalize=need_normalize, need_sampling=need_sampling,
    #      algorithm_list_filter=algorithm_list_filter, algorithm_list_wrapper=algorithm_list_wrapper)
