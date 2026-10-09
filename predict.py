import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def predict(x, weights, biases):
    curr = x
    for i in range(len(weights) - 1):
        curr = sigmoid(np.dot(curr, weights[i]) + biases[i])
    out = softmax(np.dot(curr, weights[-1]) + biases[-1])
    return out

def main():

    data = np.load("saved_model.npz")
    
    weights = []
    biases = []
    
    # счит кл
    num_layers = len(data.files) // 2
    for i in range(num_layers):
        weights.append(data[f"w_{i}"])
        biases.append(data[f"b_{i}"])

    valid_data = np.loadtxt("data_valid.csv", delimiter=",")
    x_valid, y_valid = valid_data[:, 1:], valid_data[:, 0].astype(int)

    preds = predict(x_valid, weights, biases)
    
    # расчет потери
    eps = 1e-15
    y_valid_oh = np.eye(2)[y_valid]
    preds_clipped = np.clip(preds, eps, 1 - eps)
    loss = -np.mean(np.sum(y_valid_oh * np.log(preds_clipped), axis=1))
    acc = np.mean(y_valid == np.argmax(preds, axis=1))

    print(f"Validation Binary Cross-Entropy Loss: {loss:.4f}")
    print(f"Validation Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()