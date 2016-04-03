import csv as csv
import numpy as np
import time as t
import os as os


__author__ = 'PengYan'
# modified @ 15th, Jan., 2016
# modified @ 14th, Jan., 2016
# modified @ 13th, Jan., 2016


def import_data(file_name):
    # open file given the file name and
    # return a 2-dimension list
    file_data = []
    row_count = 0
    print('Loading file: ' + file_name)
    try:
        csv_file = file(file_name, 'rb')
        lines = csv.reader(csv_file)
        for line in lines:
            file_data.append(line)
            row_count += 1
        csv_file.close()
    except Exception as e:
        print(e)
        if not file_data == []:
            print("Current Data: row " + str(row_count))
            print(file_data[-1])
    else:
        col_count = file_data[0].__len__()
        print(file_name + ' Loaded Successfully')
        print('Data Size: ' + str(row_count) + ' rows, ' + str(col_count) + ' cols')
        return file_data


def generate_validate_data(data_name, nflod=10, title_line=True):
    # loading the data, shuffling and dividing into nflod subsets and save them
    # data shall be two parts, feature set named data_name_feature and label set named data_name_label
    # data shall locate in './datasets/'. shuffled data will be saved in './CrossValidation/dataname/'
    # title_line denotes whether the first line in feature set and label set are the name of features or labels. if it
    # is True, then the title line will be save in data_name_feature_names and data_name_label_names separately

    # divide data set into nflod subsets
    print('Loading Feature Set')
    feature = import_data('datasets/' + data_name + '_feature.csv')
    print('Loading Label Set')
    label = import_data('datasets/' + data_name + '_label.csv')

    # check directory
    if not os.path.exists('CrossValidation/' + data_name):
        print('Directory: ' + data_name + ' Does Not Exist')
        try:
            os.mkdir('CrossValidation/' + data_name)
            print('Directory Made')
        except Exception, e:
            print(e)
            return

    # save the names in feature_names.csv and label_names.csv
    if title_line:
        try:
            feature_names = file('CrossValidation/' + data_name + '/' + data_name + '_feature_names.csv', 'wb')
            label_names = file('CrossValidation/' + data_name + '/' + data_name + '_label_names.csv', 'wb')
            feature_names_writer = csv.writer(feature_names)
            label_names_writer = csv.writer(label_names)
            feature_names_writer.writerow(feature[0])
            label_names_writer.writerow(label[0])
            feature = feature[1:]
            label = label[1:]
            feature_names.close()
            label_names.close()
            print('Title Saved')
        except Exception, e:
            print(e)
            return

    print('Shuffling')
    random_seed = int(t.time())  # create the seed
    # shuffle
    np.random.RandomState(random_seed).shuffle(feature)
    np.random.RandomState(random_seed).shuffle(label)

    # save file
    nrow = feature.__len__()
    subset_size = nrow / nflod
    for i in range(nflod):
        try:
            feature_train_file = file('CrossValidation/' + data_name + '/' +data_name +
                                      '_feature_train_' + str(i) + '.csv', 'wb')
            label_train_file = file('CrossValidation/' + data_name + '/' + data_name +
                                    '_label_train_' + str(i) + '.csv', 'wb')
            feature_test_file = file('CrossValidation/' + data_name + '/' + data_name +
                                     '_feature_test_' + str(i) + '.csv', 'wb')
            label_test_file = file('CrossValidation/' + data_name + '/' + data_name +
                                   '_label_test_' + str(i) + '.csv', 'wb')

            feature_train_writer = csv.writer(feature_train_file)
            label_train_writer = csv.writer(label_train_file)
            feature_test_writer = csv.writer(feature_test_file)
            label_test_writer = csv.writer(label_test_file)

            for j in range(nrow):
                if j in range(i * subset_size, (i + 1) * subset_size):
                    feature_test_writer.writerow(feature[j])
                    label_test_writer.writerow(label[j])
                else:
                    feature_train_writer.writerow(feature[j])
                    label_train_writer.writerow(label[j])

            feature_train_file.close()
            label_train_file.close()
            feature_test_file.close()
            label_test_file.close()

            print('Subset_' + str(i) + ' Saved')
        except Exception, e:
            print(e)
            break
    print('Subsets Prepared')


def import_matrix(file_name, has_title_line=False):
    # import the shuffled data as matrix
    # attention, feature_set and label_set shall imported separately

    data = import_data(file_name)
    if has_title_line:
        field_names = np.array(data[0])
        data_matrix = np.mat(data[1:], dtype=np.float64)
        return field_names, data_matrix
    else:
        data_matrix = np.mat(data[0:], dtype=np.float64)
        return data_matrix


if __name__ == '__main__':
    generate_validate_data('emotions')
    feature_names = import_data('CrossValidation/emotions/emotions_feature_names.csv')
    print('Number of Features:')
    print(feature_names[0].__len__())
    label_names = import_data('CrossValidation/emotions/emotions_label_names.csv')
    print('Number of Labels:')
    print(label_names[0].__len__())
    feature_train = import_matrix('CrossValidation/emotions/emotions_feature_train_0.csv')
    print('Number of Training Samples:')
    print(feature_train.shape[0])
    feature_test = import_matrix('CrossValidation/emotions/emotions_feature_test_0.csv')
    print('Number of Testing Samples:')
    print(feature_test.shape[0])




"""
def file2matrix(filename, num_of_attri, num_of_label=1):
    fr = open(filename)
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)
    return_mat = np.zeros((number_of_lines, num_of_attri))
    class_label_vector = np.zeros((number_of_lines, 1))
    index = 0
    for line in array_of_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:num_of_attri]
        class_label_vector[index, :] = list_from_line[-1]
        index += 1
    return return_mat, class_label_vector
"""