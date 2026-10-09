import numpy as np
import pandas as pd

def main():
    
    np.random.seed(42)

    df = pd.read_csv("data.csv", header=None)
    
    labels = df.iloc[:, 1].values
    features = df.iloc[:, 2:].values.astype(float)
    
    # перевод меток
    y = np.where(labels == 'M', 1, 0)
    
    # нормализ
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-8
    features_scaled = (features - mean) / std
    
    # перемеш инд
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    
    split_idx = int(len(features) * 0.8)
    train_idx, valid_idx = indices[:split_idx], indices[split_idx:]
    
    # сохр сплит вместе с метками
    train_data = np.column_stack((y[train_idx], features_scaled[train_idx]))
    valid_data = np.column_stack((y[valid_idx], features_scaled[valid_idx]))
    
    np.savetxt("data_train.csv", train_data, delimiter=",")
    np.savetxt("data_valid.csv", valid_data, delimiter=",")
    
    # сохр mean и std для новых предсказ
    np.savez("scaler.npz", mean=mean, std=std)
    print(f"data_train shape: {train_data.shape}")
    print(f"data_valid shape: {valid_data.shape}")

if __name__ == "__main__":
    main()