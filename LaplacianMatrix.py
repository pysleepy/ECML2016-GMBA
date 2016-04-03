import numpy as np

__author__ = 'PengYan'
# modified @ 22th, Jan., 2016
# modified @ 19th, Jan., 2016
# modified @ 18th, Jan., 2016
# modified @ 14th, Jan., 2016
# modified @ 13th, Jan., 2016


def get_deg_matrix(adjacent_matrix):
    # Build the Degree Matrix
    # adjacent_matrix get from BuildGraph
    # return the degree matrix(matrix)
    adjacent_matrix = np.mat(adjacent_matrix)  # all operations will work on matrix rather than ndarray
    deg_matrix = adjacent_matrix.sum(axis=1).A1  # A1 transform it into 1 dimension ndarray or ** and diag won't work
    deg_matrix = np.mat(np.diag(deg_matrix))
    for d in range(deg_matrix.shape[0]):
        if deg_matrix[d, d] == 0:
            deg_matrix[d, d] = 0.01
    print("DEGREE MATRIX:")
    print deg_matrix
    return deg_matrix


def get_laplacian_matrix(adjacent_matrix, normalize=True):
    # adjacent_matrix get from BuildGraph
    # normalize denotes weather to normalize the laplacian matrix
    # return the laplacian matrix(matrix)

    # Build Laplacian Matrix
    adjacent_matrix = np.mat(adjacent_matrix)  # all operations will work on matrix rather than ndarray
    print("ADJACENT MATRIX:")
    print adjacent_matrix

    nrow = adjacent_matrix.shape[0]  # num of row
    ncol = adjacent_matrix.shape[1]  # num of col
    print("MATRIX SIZE:")
    print '%d * %d' % (nrow, ncol)

    # Get Degree Matrix
    deg_matrix = get_deg_matrix(adjacent_matrix)

    # Get Laplacian Matrix
    lap_matrix = deg_matrix - adjacent_matrix
    if normalize:
        sqrt_deg_matrix = np.mat(np.diag(np.diag(deg_matrix) ** (-0.5)))
        lap_matrix = sqrt_deg_matrix * lap_matrix * sqrt_deg_matrix
        #    lap_matrix[row, col] = lap_matrix[row, col] * (deg_matrix[row,row] ** (-0.5)) \
        #                           * (deg_matrix[col, col] ** (-0.5))
    print("LAPLACIAN MATRIX:")
    print lap_matrix

    return lap_matrix


def get_spectrum(adjacent_matrix, normalize=True):
    # adjacent_matrix get from BuildGraph
    # normalize denotes weather to normalize the laplacian matrix
    # return the sorted eigen value and corresponding eigen vectors(each column an eigen vector)( matrix)

    # get the spectrum of the given graph described by adjacent_matrix
    adjacent_matrix = np.mat(adjacent_matrix)  # all operations will work on matrix rather than ndarray

    # get laplacian matrix
    lap_matrix = get_laplacian_matrix(adjacent_matrix, normalize)

    # get eigen values and eigen vectors of the laplacian matrix
    (eigen_values, eigen_vectors) = np.linalg.eigh(lap_matrix)

    # sort eigen velues and eigen vectors according to eigen values in ascend order
    ind = eigen_values.argsort()
    eigen_values.sort()
    print('EIGEN VALUES:')
    print eigen_values
    eigen_vectors = eigen_vectors[:, ind]
    print("EIGEN VECTORS:")
    print eigen_vectors

    return eigen_values, eigen_vectors


if __name__ == '__main__':
    adjacent_matrix = np.zeros((5, 5))
    for r in range(0, 4):
        for c in range((r + 1), 5):
            adjacent_matrix[r, c] = np.random.random_sample()
            adjacent_matrix[c, r] = adjacent_matrix[r, c]

    (eigen_values, eigen_vectors) = get_spectrum(adjacent_matrix)
