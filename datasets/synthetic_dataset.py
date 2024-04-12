import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self,
                 dim_in,
                 dim_out,
                 n_samples,
                 func,
                 data_generator=None,
                 noise_scale=0.1,
                 **kwargs):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_samples = n_samples
        self.func = func
        self.noise_scale = noise_scale

        if data_generator is None:
            # create a function that returns Gaussian noise of shape (n_samples, dim_out)
            self.data_generator = lambda: np.random.normal(0, 1, (n_samples, dim_out))
        else:
            self.data_generator = data_generator

        self.X = self.data_generator()
        self.y = [self.func(x) for x in self.X]
        self.y = np.array(self.y)
        self.y += np.random.normal(0, noise_scale, (n_samples, dim_out))
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_data(self):
        return self.X, self.y

    def get_true_data(self):
        return self.X, self.func(self.X)

    def get_true_predictions(self):
        return self.func(self.X)

    def get_noisy_predictions(self):
        return self.func(self.X) + np.random.normal(0, self.noise_scale, (self.n_samples, self.dim_out))

    def get_params(self):
        pass

class LinearRegressionDataset(SyntheticDataset):
    def __init__(self,
                 n_samples=1000,
                 n_features=1,
                 dim_out=1,
                 bias_scale = 0,
                 noise_scale=0.1,
                 data_distribution="Gaussian",
                 set_weights=None,
                 set_bias=None,
        ):

        assert set_weights is None or set_weights.shape == (n_features, dim_out)
        assert set_bias is None or set_bias.shape == (dim_out,)
        self.W = np.random.normal(0, 1, (n_features, dim_out)) if set_weights is None else set_weights
        self.b = np.random.normal(0, bias_scale, (dim_out,)) if set_bias is None else set_bias

        assert data_distribution in ["Gaussian", "Uniform"] # only support Gaussian and Uniform distribution
        if data_distribution == "Gaussian":
            self.data_generator = lambda: np.random.normal(0, 1, (n_samples, n_features))
        elif data_distribution == "Uniform":
            self.data_generator = lambda: np.random.uniform(-1, 1, (n_samples, n_features))

        def linearfunc(x, W=self.W, b=self.b):
            return np.dot(x, W) + b
        super().__init__(n_features, dim_out, n_samples, linearfunc, self.data_generator, noise_scale)

    def get_params(self):
        return self.W, self.b

class SinusoidalDataset(SyntheticDataset):
    def __init__(self,
                 n_samples=1000,
                 n_features=1,
                 dim_out=1,
                 noise_scale=0.1,
                 bias_scale = 0,
                 data_distribution="Gaussian",
                 set_weights=None,
                 set_bias=None,
                 ):
        self.noise_scale = noise_scale

        assert data_distribution in ["Gaussian", "Uniform"] # only support Gaussian and Uniform distribution
        if data_distribution == "Gaussian":
            data_generator = lambda: np.random.normal(0, 1, (n_samples, n_features))
        elif data_distribution == "Uniform":
            data_generator = lambda: np.random.uniform(-1, 1, (n_samples, n_features))

        assert set_weights is None or set_weights.shape == (n_features,)
        assert set_bias is None or set_bias.shape == (1,)
        self.W = np.random.normal(0, 1, (n_features, 1)) if set_weights is None else set_weights
        self.b = np.random.normal(0, bias_scale, (1,)) if set_bias is None else set_bias

        def sinusoidalfunc(x, W=self.W, b=self.b):
            # print(x.shape, W.shape, b.shape)
            return np.sin(np.dot(x, W) + b)

        super().__init__(n_features, dim_out, n_samples, sinusoidalfunc, data_generator)

    def get_params(self):
        return self.W, self.b

if __name__ == "__main__":
    dataset = LinearRegressionDataset(
        n_samples=100,
        n_features=1,
        dim_out=1,
        bias_scale = 0,
        noise_scale=3e-2,
    )
    # dataset = SinusoidalDataset(
    #     n_samples=100,
    #     n_features=1,
    #     dim_out=1,
    #     noise_scale=3e-2,
    #     data_distribution="Uniform",
    #     set_weights=np.array([3.0]),
    # )
    print(dataset[0])
    import matplotlib.pyplot as plt
    X, y = dataset.get_data()
    print(dataset.get_params())
    # plt.scatter(X, y)
    # plt.savefig("vis/sin_regression_data.png")