import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 31)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(31, 31)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(31, 31)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(31, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x


class dnn():
    def __init__(self, n_epochs=250, batch_size=10):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, X_train0, y_train0, weights0=None):
        """
        Trains a DNN neural network on X and y data.

        :param X_train0: Feature data (X_train).
        :param y_train0: Labels (y_train).
        :param weights0: Sample weights (optional).
        """
        # Converting Pandas DataFrame/Series to NumPy array if necessary
        if hasattr(X_train0, "to_numpy"):
            X_train0 = X_train0.to_numpy()
        if hasattr(y_train0, "to_numpy"):
            y_train0 = y_train0.to_numpy()

        # Ensuring the arrays are numeric
        X_train0 = np.array(X_train0, dtype=np.float32)
        y_train0 = np.array(y_train0, dtype=np.float32)

        if weights0 is None:
            weights0 = np.ones(X_train0.shape[0])
        weights0 = weights0 / np.sum(weights0)

        # Converting data to tensors
        X_train = torch.tensor(X_train0, dtype=torch.float32)
        y_train = torch.tensor(y_train0, dtype=torch.float32).reshape(-1, 1)
        weights = torch.tensor(weights0, dtype=torch.float32).reshape(-1, 1)

        # Initializing the DNN model
        input_dim = X_train.shape[1]
        self.model = DeepNN(input_dim)

        # Defining loss function and optimizer
        loss_fn = nn.BCELoss(reduction='none')  # Binary Cross-Entropy with weights
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        batch_size = self.batch_size
        batch_start = torch.arange(0, len(X_train), batch_size)

        # Training the model
        for epoch in range(self.n_epochs):
            self.model.train()
            with tqdm(batch_start, unit="batch", disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # Extracting batch
                    X_batch = X_train[start:start + batch_size]
                    y_batch = y_train[start:start + batch_size]
                    weights_batch = weights[start:start + batch_size]

                    # Forward pass
                    y_pred = self.model(X_batch)
                    loss = weights_batch * loss_fn(y_pred, y_batch)
                    loss = loss.sum()

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    def predict_proba(self, X_val0):
        """
        Predicts the probability of the positive class.

        :param X_val0: Feature data to make predictions on.
        :return: Probabilities for classes 0 and 1.
        """
        # Converting Pandas DataFrame/Series to NumPy array if necessary
        if hasattr(X_val0, "to_numpy"):
            X_val0 = X_val0.to_numpy()
        X_val0 = np.array(X_val0, dtype=np.float32)
        X_val = torch.tensor(X_val0, dtype=torch.float32)
        y_prob_tensor = self.model(X_val)
        y_prob = y_prob_tensor.detach().numpy()[:, 0]
        q_xall = np.zeros((X_val0.shape[0], 2))
        q_xall[:, 0] = 1 - y_prob
        q_xall[:, 1] = y_prob
        return q_xall
