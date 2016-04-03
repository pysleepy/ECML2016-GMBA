import numpy as np


__author__ = 'PengYan'
# modified @ 18th, Jan., 2016
# modified @ 14th, Jan., 2016
# modified @ 13th, Jan., 2016


def get_similarity(a, b, stype='WeJaccard', var=1, weight=[]):
    # calculate similarity between a and b (a and b shall be 1*d matrix)
    # stype denotes what kind of similarity is used
    # var denotes variance for RBF kernel
    # weight is needed when use weighted Jaccard similarity
    # return a number(double)
    a = np.mat(a)
    b = np.mat(b)
    if stype == 'WeJaccard':
        inte = 0;
        unio = 0;
        for label in range(a.size):
            if a[0, label] == 1 and b[0, label] == 1:
                inte += weight[0, label]
                unio += weight[0, label]
            elif a[0, label] == 1 or b[0, label] == 1:
                unio += weight[0, label]
        if not unio == 0:
            return inte * 1.0 / unio
        else:
            return 0

    if stype == 'Jaccard':
        num_intersection = (a * b.transpose())[0, 0]  # transform it into a number
        num_union = a.sum() + b.sum() - num_intersection
        if num_union != 0:
            return num_intersection * 1.0 / num_union
        else:
            return 0.0

    if stype == 'SingleLabel':
        if a[0, 0] == b[0, 0]:
            return 1.0
        else:
            return 0.0

    if stype == 'cosine':
        if np.linalg.norm(a) != 0 and np.linalg.norm(b) != 0:
            return ((a * b.transpose() * 1.0) / (np.linalg.norm(a) * np.linalg.norm(b)))[0, 0]
            # the result is transformed into a number
        else:
            return 0.0

    if stype == 'rbf':
        dif = a - b
        dis = np.linalg.norm(dif)
        return np.exp(- 0.5 * (dis ** 2) / (var ** 2))

    if stype == '2_norm':
        dif = a - b
        dis = np.linalg.norm(dif)
        return dis

    return 0.0


def build_adjacent_matrix(data_matrix, stype='WeJaccard', var=1):
    # build adjacent matrix according to similarity between each pair of sample
    # data_matrix shall ba a matrix, each sample a row
    # stype and var are the same as those in get_similarity()
    # return the adjacent matrix(matrix type)

    print('Building Graph')
    data_matrix = np.mat(data_matrix)
    nrow = data_matrix.shape[0]
    adjacent_matrix = np.zeros((nrow, nrow))

    if stype == 'WeJaccard':
        weight = data_matrix.sum(axis=0)
    else:
        weight = []

    for row in range(nrow - 1):
        for col in range(row + 1, nrow):
            adjacent_matrix[row][col] = get_similarity(data_matrix[row], data_matrix[col], stype, var, weight)
            adjacent_matrix[col][row] = adjacent_matrix[row][col]
    return adjacent_matrix


if __name__ == '__main__':
    data_matrix = np.mat('1 1 1 0 0 0 1; 0 0 1 0 1 1 1; 0 0 0 0 0 0 1; 0 0 0 1 1 1 1')
    print('data matrix:')
    print(data_matrix)
    print('adjacent matrix(Weighted Jaccard):')
    print(build_adjacent_matrix(data_matrix))
    print('adjacent matrix(Jaccard):')
    print(build_adjacent_matrix(data_matrix, stype='Jaccard'))
    print('adjacent matrix(cosine):')
    print(build_adjacent_matrix(data_matrix, stype='cosine'))
    print('adjacent matrix(rbf):')
    print(build_adjacent_matrix(data_matrix, stype='rbf', var=1))
