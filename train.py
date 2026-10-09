import argparse
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    return a * (1.0 - a)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class MLP:
    def __init__(self, input_dim, hidden_layers, output_dim):
        self.layer_sizes = [input_dim] + hidden_layers + [output_dim]
        self.weights = []
        self.biases = []
        
        # xavier иниц
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(2.0 / self.layer_sizes[i])
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        activations = [x]
        curr = x
        for i in range(len(self.weights) - 1):
            z = np.dot(curr, self.weights[i]) + self.biases[i]
            curr = sigmoid(z)
            activations.append(curr)
        
        # вых слой (softmax)
        z_out = np.dot(curr, self.weights[-1]) + self.biases[-1]
        out = softmax(z_out)
        activations.append(out)
        return activations

    def backward(self, activations, y_onehot, lr):
        m = y_onehot.shape[0]
        dz = activations[-1] - y_onehot
        
        for i in range(len(self.weights) - 1, -1, -1):
            dw = np.dot(activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * sigmoid_derivative(activations[i])
                
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

def compute_loss(y_true_onehot, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))

def compute_acc(y_true, y_pred):
    return np.mean(y_true == np.argmax(y_pred, axis=1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", nargs="+", type=int, default=[24, 24, 24])
    parser.add_argument("--epochs", type=int, default=84)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.0314)
    args = parser.parse_args()

    train_data = np.loadtxt("data_train.csv", delimiter=",")
    valid_data = np.loadtxt("data_valid.csv", delimiter=",")

    x_train, y_train = train_data[:, 1:], train_data[:, 0].astype(int)
    x_valid, y_valid = valid_data[:, 1:], valid_data[:, 0].astype(int)

    # one-hot encoding для 2 кл
    y_train_oh = np.eye(2)[y_train]
    y_valid_oh = np.eye(2)[y_valid]

    model = MLP(input_dim=x_train.shape[1], hidden_layers=args.layer, output_dim=2)

    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        
        for i in range(0, len(x_train), args.batch_size):
            batch_idx = indices[i:i+args.batch_size]
            x_b, y_b = x_train[batch_idx], y_train_oh[batch_idx]
            acts = model.forward(x_b)
            model.backward(acts, y_b, args.learning_rate)

        # вычисл метрик на всех выборкн
        train_acts = model.forward(x_train)
        valid_acts = model.forward(x_valid)

        tr_loss = compute_loss(y_train_oh, train_acts[-1])
        val_loss = compute_loss(y_valid_oh, valid_acts[-1])
        tr_acc = compute_acc(y_train, train_acts[-1])
        val_acc = compute_acc(y_valid, valid_acts[-1])

        history["loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:02d}/{args.epochs} - loss: {tr_loss:.4f} - val_loss: {val_loss:.4f} - acc: {tr_acc:.4f} - val_acc: {val_acc:.4f}")

    save_dict = {}
    for idx, (w, b) in enumerate(zip(model.weights, model.biases)):
        save_dict[f"w_{idx}"] = w
        save_dict[f"b_{idx}"] = b

    np.savez("saved_model.npz", **save_dict)
    print("> saving model './saved_model.npz' to disk...")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="training loss")
    plt.plot(history["val_loss"], label="validation loss")
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["acc"], label="training acc")
    plt.plot(history["val_acc"], label="validation acc")
    plt.title("Learning Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("learning_curves.png")
    plt.show()

if __name__ == "__main__":
    main()