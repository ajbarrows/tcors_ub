import numpy as np
from collections import namedtuple

def initialize_weights(X, y, n_h):
    n_i = X.shape[1]
    n_t = y.shape[1]

    W1 = np.array(np.random.uniform(size = (n_i, n_h)))
    W2 = np.array(np.random.uniform(size = (n_h, n_t)))

    weights = namedtuple("weights", ["W1", "W2"])

    return weights(W1, W2)

def min_euclidean_dist(V, W1):
    ''' Return node with smallest Euclidean distance'''
    n_h = W1.shape[1]
    dist_vec = list()

    for j in range(n_h):
       eucl = np.sqrt(np.sum(np.square(V - W1[:, j])))
       dist_vec.append(eucl)
    winning_node = np.argmin(dist_vec)

    return winning_node

def forward_pass(V, weights, n_h):
    a_j = np.zeros(n_h)
    winS_j = min_euclidean_dist(V, weights.W1)

    a_j[winS_j] = 1

    S_k = np.dot(a_j, weights.W2)

    fwd_pass = namedtuple("fwd_pass", ["winS_j", "a_j", "S_k", "W1", "W2"])

    return fwd_pass(
        winS_j,
        a_j,
        S_k,
        weights.W1,
        weights.W2
    )
    
def adjust_weights(fwd_pass, V, t, alpha_val, beta_val):

    W1 = fwd_pass.W1
    W2 = fwd_pass.W2

    # hidden weights
    W1win = W1[:, fwd_pass.winS_j]
    W1[:, fwd_pass.winS_j] = W1win + alpha_val * (V - W1win)

    # output weights
    W2win = W2[fwd_pass.winS_j, :]
    W2[fwd_pass.winS_j, :] = W2win + beta_val * (t - fwd_pass.S_k)

    new_weights = namedtuple("new_weights", ["W1", "W2"])

    return new_weights(W1, W2)

def rmse(X, y_true, y_pred):
    n_p = X.shape[0]
    n_o = y_true.shape[0]

    return np.sqrt(np.sum(np.square(y_pred - y_true) / (n_p * n_o)))


def train_model(X: np.array, y: np.array, n_hidden=10, epochs=100, alpha_val=0.7, beta_val=0.1):

    weights = initialize_weights(X, y, n_hidden)

    M = X.shape[0]
    rmse_values = []
    for i in range(epochs):
        outputs = []
        for m in range(M):
            V = X[m, :].T
            t = y[m, :]

            fwd_pass = forward_pass(V, weights, n_hidden)
            weights = adjust_weights(fwd_pass, V, t, alpha_val, beta_val)
            outputs.append(fwd_pass.S_k)
        rmse_i = rmse(X, y, outputs)
        rmse_values.append(rmse_i)
    
    model_fit = namedtuple("model_fit", ["outputs", "rmse_values"])

    return model_fit(outputs, rmse_values)