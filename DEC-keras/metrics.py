import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    # y_true = np.array(y_true, dtype='int64')

    # print('y_pred.size :', y_pred.size)
    # print('y_true.size :', y_true.size)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    # print('D :', D)
    w = np.zeros((D, D), dtype=np.int64)
    # print('w :', w)
    # print('y_pred :', y_pred)
    # print('y_true :', y_true)
    # print('y_true[5] :', y_true[5])
    # print(w[y_pred[5], y_true[5]])
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    # print('ind :', ind[0], ind[1])
    # i, j = ind
    # print(w[i, j])
    # for i in ind[0]:
    #     for j in ind[1]:
    #         print([i, j])
    # print([i, j] for i, j in ind)
    # print(sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size)
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size