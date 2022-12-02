import numpy as np
from collections import namedtuple

np.random.seed(42)

def onehot_output(y):
    n_values = np.max(y) + 1
    return np.eye(n_values + 1)[y]

def initialize_weights(X, y, n_h, random_init=False):
    n_i = X.shape[1]
    n_t = y.shape[1]

    if random_init:
        W1 = np.array(np.random.uniform(size = (n_i, n_h)))
    else:
        W1 = np.array(X).T
    
    W2 = np.array(np.random.uniform(size = (n_h, n_t)))

    weights = namedtuple("weights", ["W1", "W2"])

    return weights(W1, W2)

def add_noise(W1):
    noise = np.random.normal(0, 0.1, W1.shape)
    W1_noisy = W1 + noise
    return W1_noisy

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
    
def adjust_weights(fwd_pass, V, t, alpha_val, beta_val, adjust_w2=True):

    W1 = fwd_pass.W1
    W2 = fwd_pass.W2

    # hidden weights
    W1win = W1[:, fwd_pass.winS_j]
    W1[:, fwd_pass.winS_j] = W1win + alpha_val * (V - W1win)

    if adjust_w2:
        # output weights
        W2win = W2[fwd_pass.winS_j, :]
        W2[fwd_pass.winS_j, :] = W2win + beta_val * (t - fwd_pass.S_k)

    new_weights = namedtuple("new_weights", ["W1", "W2"])

    return new_weights(W1, W2)

def rmse(X, y_true, y_pred):
    n_p = X.shape[0]
    n_o = y_true.shape[0]

    return np.sqrt(np.sum(np.square(y_pred - y_true) / (n_p * n_o)))



def train_model(X: np.array, y: np.array, random_init=False, n_hidden=10, epochs=100, alpha_val=0.7, beta_val=0.1):

    weights = initialize_weights(X, y, n_hidden, random_init=random_init)

    M = X.shape[0]
    rmse_values = []
    for i in range(epochs):
        outputs = []
        for m in range(M):
            V = X[m, :]
            t = y[m, :]
            fwd_pass = forward_pass(V, weights, n_hidden)
            weights = adjust_weights(fwd_pass, V, t, alpha_val, beta_val)
            outputs.append(fwd_pass.S_k)
        rmse_i = rmse(X, y, outputs)
        rmse_values.append(rmse_i)
    
    pass1_weights = weights

    # add noise to W1, retrain
    weights = weights._replace(W1 = add_noise(weights.W1))
    for i in range(epochs):
        outputs = []
        for m in range(M):
            V = X[m, :]
            t = y[m, :]
            fwd_pass = forward_pass(V, weights, n_hidden)
            weights = adjust_weights(fwd_pass, V, t, alpha_val, beta_val, adjust_w2=True)
            outputs.append(fwd_pass.S_k)
        rmse_i = rmse(X, y, outputs)
        rmse_values.append(rmse_i)

    pass2_weights = weights

    model_fit = namedtuple("model_fit", ["rmse_values", "pass1_weights", "pass2_weights"])

    return model_fit(rmse_values, pass1_weights, pass2_weights)

def predict_model(X: np.array, y: np.array, weights):
    M = X.shape[0]
    n_h = weights.W1.shape[1]

    outputs = []
    for m in range(M):
        V = X[m, :]
        t = y[m, :]
        fwd_pass = forward_pass(V, weights, n_h)
        outputs.append(fwd_pass.S_k)

    # winner-take-all
    predict_cat = np.argmax(outputs, axis=1)

    result = namedtuple("result", ["predict_prob", "predict_cat"])

    return result(outputs, predict_cat)



if __name__ == "__main__":
    X_train = np.load("../../data/processed/X_train.npy")
    y_train = np.load("../../data/processed/y_train.npy")

    y_train = onehot_output(y_train)
    # weights = initialize_weights(X_train, onehot_output(y_train), n_h = 10)
    mod = train_model(X_train, y_train,n_hidden = len(X_train), epochs=25)

    import matplotlib.pyplot as plt

    plt.plot(mod.rmse_values)
    plt.show()


    # print(np.unique(y_train))
    # print(onehot_output(y_train).shape[1])

    # print(X_train.shape)

    # print(add_noise(X_train))
 