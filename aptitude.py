# Name: Ashley Stephenson
# Project: APTitude Aptamer Classification
# Description: Convolutional model for optimizing SELEX for high-throughput experimentation.

# imports
import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt

# numpy configuration
rng = np.random.default_rng(4)
np.set_printoptions(precision=5)


# import data
def import_data(seq_data, threshold):
    print(f"  Importing SELEX data - {seq_data}")
    path = './data'
    with open(path + seq_data, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=' ', quotechar='|'))[1:-1]
        m, n = len(data), len(data[0][0].split(",")[1])
        X = np.zeros([m,n])
        Y = np.zeros([m,1])
        seq_keys = {}

        for i in range(m):
            row = data[i][0].split(",")
            # encode A, T, C, G = 1, 2, 3, 4
            seq = row[1]
            seq = seq.replace("A", "1")
            seq = seq.replace("T", "2")
            seq = seq.replace("C", "3")
            seq = seq.replace("G", "4")
            for j in range(n):
                X[i,j] = float(seq[j])
            Y[i,0] = 0 if (int(row[2]) < threshold) else 1
            seq_keys[seq] = 0 if (int(row[2]) < threshold) else 1
        return X, Y, m, n, seq_keys

# combine sequencing data and set positive and negative labels for binding
def combine_seq_rounds(files, threshold):
    print(f'\n Combining sequencing data according to threshold: {threshold}')
    master_keys = {}
    for i, f in enumerate(files):
        X, Y, m, n, seq_keys = import_data(f, threshold)
        # check if on last file, aka the final round of SELEX data
        if (i == len(files)):
            for j, key in enumerate(master_keys):
                # sequence survived to final round, set to 1 for binding
                if (seq_keys[key]):
                    master_keys[key] = Y[j]
                # sequence is not in final round, set to 0 for not binding
                else:
                    master_keys[key] = 0
        # populate dictionary with sequences from all SELEX rounds
        for j, key in enumerate(seq_keys):
            master_keys[key] = Y[j]
        seqs = np.matrix([list(seq) for seq in list(master_keys.keys())], dtype=np.float64)
        labels = np.matrix(list(master_keys.values()), dtype=np.float64)
    print(f"\n Combining SELEX data...")
    return seqs, labels

def create_experiment(experiment_params, i):
    prnt, epochs, alpha, lamb, train_size, threshold, l1, b, name = experiment_params
    return {
        "name": f'v{i}-{name}',          # experiment name
        "prnt": prnt,                   # how frequently to display cost, acc & error
        "hyperparams": {
            "epochs": epochs,           # number of gradient descent iterations
            "alpha": alpha,             # learning rate
            "lamb": lamb,               # regularization rate
            "train_size": train_size,   # dataset train/test ratio
            "threshold": threshold,     # sequencing read-count threshold for binding affinity classification
            "l1": l1,                   # width of layer 1
            "b": b                      # boundary for correct classification
        }
    }

def save_experiment(experiment_data):
    if not os.path.exists(f'./out/{experiment_data["name"]}'): os.makedirs(f'./out/{experiment_data["name"]}')
    out_path = f'./out/{experiment_data["name"]}/aptitude-model-results.txt'
    print(f'\n Writing experiment data to disk - {out_path}')
    f = open(out_path, 'w+')
    f.write(f'# --- APTitude Experimental Analytics Report ({experiment_data["name"]}) --- #')
    f.write(f'\n\n# --- EXPERIMENT HYPERPARAMETERS --- #')
    [f.write(f'\n{param[1]}: {experiment_data["hyperparams"][param[1]]}') for param in enumerate(experiment_data["hyperparams"])]
    f.write(f'\n\n# --- LEARNED MODEL PARAMETERS --- #')
    [f.write(f'\n\n{layer[1]}: \n{experiment_data["learned_params"][layer[1]]}') for layer in enumerate(experiment_data["learned_params"])]
    f.write(f'\n\n# --- EXPERIMENT COST, ACC. & ERROR HISTORY --- #')
    [f.write(f'\n\n{history[1]}: \n{experiment_data["history"][history[1]]}') for history in enumerate(experiment_data["history"])]
    f.close()

def run_experiment(Model, experiment_data):
    epochs, alpha, lamb, train_size, threshold, l1, b = experiment_data["hyperparams"].values()
    name, prnt = experiment_data["name"], experiment_data["prnt"]
    print(f'\n Running experiment "{name}" - {experiment_data["hyperparams"]}')

    # import SELEX data
    files = [
        '/HHHHHHHHHHHH_B1mut_control_strategy2_R2_top_seq.csv',
        '/HHHHHHHHHHHH_B1mut_control_strategy2_R7_top_seq.csv',
        '/HHHHHHHHHHHH_B1mut_control_strategy2_R9_top_seq.csv',
        '/HHHHHHHHHHHH_B1mut_control_strategy2_R11_top_seq.csv',
        '/HHHHHHHHHHHH_control_strategy1_R3_top_seq.csv',
        '/HHHHHHHHHHHH_control_strategy1_R6_top_seq.csv',
        '/HHHHHHHHHHHH_control_strategy1_R9_top_seq.csv'
    ]
    seqs, labels = combine_seq_rounds(files, threshold)     # encode & combine sequencing data; assign labels per threshold
    m, n = seqs.shape

    # shuffle data
    print("\n Creating training & test sets...")
    shuff = np.arange(m)
    np.random.shuffle(shuff)
    X_shuff, Y_shuff = seqs[shuff], labels[shuff]

    # split train & test sets
    X_train, Y_train = X_shuff[:int(train_size*X_shuff.shape[0])], Y_shuff[:int(train_size*X_shuff.shape[0])]
    X_test, Y_test = X_shuff[int(train_size*X_shuff.shape[0]):], Y_shuff[int(train_size*X_shuff.shape[0]):]

    # normalize data - numerically-encoded sequence normalization
    X_train_norm, X_test_norm = X_train/4, X_test/4

    # data dims
    m_train, n_train = X_train.shape[0], X_train.shape[1]
    m_test, n_test = X_test.shape[0], X_test.shape[1]

    # train models
    model = Model(alpha, lamb, epochs, [m_train, n_train, l1])
    hist = model.train(X_train, Y_train, m_train, n_train, b, prnt)

    # test model
    Z1, A1, Z2, A2 = model.compute_activations(X_test)
    cost, acc, err = model.test(Y_test, A2, b)
    print(f"\n Final Evaluation of Testing Performance\n cost: {np.round(cost, 5)}  acc: {np.round(acc, 5)}%\n")

    # plot cost and accuracy
    plot_data(hist[0], 'Cost During Training', 'Epochs', 'Cost', experiment_data["name"])
    plot_data(hist[1], 'Accuracy During Training', 'Epochs', 'Accuracy', experiment_data["name"])
    plot_data(hist[2], 'Error During Training', 'Epochs', 'Error', experiment_data["name"])

    # write plots & learned params to disk
    experiment_data["learned_params"] = model.params
    experiment_data["history"] = { "cost": hist[0], "accuracy": hist[1], "error": hist[2] }
    save_experiment(experiment_data)
    return np.average(hist[2]) / np.max(hist[2]), np.max(hist[2]) - np.max(np.min(hist[2]), 0)


# i/o utils
def plot_data(data, title, x_label, y_label, name):
    fig, ax1 = plt.subplots(figsize=[10,6])
    plt.title(title, fontsize=16, fontweight='bold')
    plt.style.use('seaborn-whitegrid')
    ax1.tick_params(axis='y')
    ax1.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax1.plot(data, '-', color='#0af', label=y_label)
    fig.legend(frameon=True, borderpad=0.5, facecolor='#fff', framealpha=1, edgecolor='#777', shadow=True)
    if not os.path.exists(f'./out/{name}'): os.makedirs(f'./out/{name}')
    plt.savefig(f'./out/{name}/aptitude-model-{y_label.lower()}-plot.png')
    plt.close("all")

def greet():
    if (os.name == 'posix'): os.system("clear")
    else: os.system("cls")
    print(f"\n Welcome to APTitude\n -------------------")


# math utils
def linear(A, W):
    return A.dot(W)

def conv1D(A, W):
    m, n, fn, fw = A.shape[0], A.shape[1], W.shape[0], W.shape[1]
    Z = np.zeros([m, (n-fw+1)*fn])
    for f in range(fn):
        for s in range((n-fw+1)*30):
            Z[:,s:s+fw] = np.sum(np.multiply(A[:, s:s+fw], W[f,:]))
    return Z

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def dSigmoid(A):
    return np.multiply(A, (1 - A))

def one_hot_encode(X):
    one_hot = np.zeros([X.shape[0], X.shape[1], 4])
    for i in range(X.shape[0]):
        for j in range(len(X[i])):
            base = X[i,j]
            one_hot[i,j,base-1] = 1
    return one_hot


# model
class APTitude():
    def __init__(self, alpha, lamb, epochs, params):
        m, n, l1 = params[0], params[1], params[2]
        print(f'\n Initializing new APTitude model...')
        self.alpha = alpha
        self.lamb = lamb
        self.epochs = epochs
        self.params = {
            "W1": rng.standard_normal([n, l1]) * 0.1,
            "W2": rng.standard_normal([l1, 1]) * 0.1
        }
        print(f'  W1: {self.params["W1"].shape}  -  W2: {self.params["W2"].shape}')

    def train(self, X, Y, m, n, b, prnt):
        print(f'\n  Beginning Gradient Descent - {self.epochs} epochs')
        hist = np.zeros([3, self.epochs])
        e = 0
        while e < self.epochs:
            if (e%prnt==0): print(f"\n  Starting Epoch: {e}")
            Z1, A1, Z2, A2 = self.compute_activations(X)
            dEW2, dEW1 = self.compute_gradients(Y, X, Z1, A1, Z2, A2)
            self.update_params(dEW2, dEW1, m)
            cost, acc, err = self.test(Y, A2, b)
            hist[0, e], hist[1,e], hist[2,e] = cost, acc, err
            if (e%prnt==0): print(f"   cost: {np.round(hist[0,e],4)}  acc: {np.round(hist[1,e],4)}%")
            e += 1
        return hist

    def compute_activations(self, A0):
        Z1 = linear(A0, self.params["W1"])
        A1 = sigmoid(Z1)
        Z2 = linear(A1, self.params["W2"])
        A2 = sigmoid(Z2)
        return Z1, A1, Z2, A2

    def compute_gradients(self, Y, A0, Z1, A1, Z2, A2):
        dEA2 = -(Y-A2)
        dA2Z2 = dSigmoid(A2)
        dZ2W2 = A1
        dZ2A1 = self.params["W2"]
        dEZ2 = np.multiply(dEA2, dA2Z2)
        dEW2 = dZ2W2.T.dot(dEZ2)
        dEA1 = dEZ2.dot(dZ2A1.T)
        dA1Z1 = dSigmoid(A1)
        dEZ1 =  np.multiply(dEA1, dA1Z1)
        dZ1W1 = A0
        dZ1A0 = self.params["W1"]
        dEW1 = dZ1W1.T.dot(dEZ1)
        return dEW2, dEW1

    def update_params(self, dw2, dw1, m):
        reg = (np.sum(np.square(self.params["W1"])) + np.sum(np.square(self.params["W2"]))) * self.lamb / (2*m)
        self.params["W1"] -= dw1 * self.alpha + reg * self.lamb
        self.params["W2"] -= dw2 * self.alpha + reg * self.lamb

    def test(self, Y, A2, b):
        cost = self.cost(Y, A2)
        acc = self.accuracy(Y, A2, b)
        err = self.error(Y, A2)
        return cost, acc, err

    def error(self, Y, A2):
        return np.sum(np.square(Y-A2)/2)

    def cost(self, Y, A2):
        lyh = np.log(A2+0.001)
        ylyh = np.multiply(Y, lyh)
        lnyh = np.log(1-A2+0.001)
        nylnyh = np.multiply((1-Y), lnyh)
        cost = np.average(ylyh + nylnyh) * -1
        return cost

    def accuracy(self, Y, A2, b):
        A2[A2<b] = 0
        A2[A2>b] = 1
        eq = Y[Y==A2]
        m = Y.shape[0]
        correct = np.sum(eq)
        acc = correct/m * 100
        return acc


# Test Model
def main(args):
    # clear screen & display application info
    greet()

    # define hyperparameters & run multiple experiments
    experiments = [
        [50000/10, 50000, 0.0001, 0.001, 0.90, 7000, 25, 0.8, 'baseline'],
        [50000/10, 50000, 0.00001, 0.001, 0.90, 7000, 25, 0.8, 'low-learning-rate'],
        [50000/10, 50000, 0.001, 0.001, 0.90, 7000, 25, 0.8, 'high-learning-rate'],
        [50000/10, 50000, 0.0001, 0.0001, 0.90, 7000, 25, 0.8, 'low-regularization-rate'],
        [50000/10, 50000, 0.0001, 0.01, 0.90, 7000, 25, 0.8, 'high-regularization-rate'],
        [50000/10, 50000, 0.0001, 0.001, 0.90, 5000, 25, 0.8, 'lower-sequence-threshold'],
        [50000/10, 50000, 0.0001, 0.001, 0.90, 9000, 25, 0.8, 'higher-sequence-threshold'],
        [50000/10, 50000, 0.0001, 0.001, 0.81, 7000, 25, 0.8, 'lower-training-ratio'],
        [50000/10, 50000, 0.0001, 0.001, 0.99, 7000, 25, 0.8, 'higher-training-ratio'],
        [50000/10, 50000, 0.0001, 0.001, 0.90, 7000, 25, 0.9, 'stricter-classification-bounds'],
        [50000/10, 50000, 0.0001, 0.001, 0.90, 7000, 25, 0.6, 'looser-classification-bounds'],
        [50000/10, 50000, 0.0001, 0.001, 0.90, 7000, 35, 0.8, 'wider-layer'],
        [100000/20, 100000, 0.0001, 0.001, 0.90, 7000, 25, 0.8, 'higher-epochs'],
        [200000/50, 200000, 0.0001, 0.001, 0.90, 7000, 25, 0.8, 'highest-epochs']
    ]


    i = 0
    while i < len(experiments):
        experiment_data = create_experiment(experiments[i], i)
        acc_ratio, acc_range = run_experiment(APTitude, experiment_data)
        i += 1

    # MAIN END


if __name__ == "__main__":
    main(sys.argv)
