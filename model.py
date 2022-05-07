import numpy as np

from scipy.special import softmax
from typing import Dict


class NeuralNet:
    
    def __init__(self, data: np.array, y: np.array, hidden_units: int):
        self.data = data
        self.data_norm = (data - data.mean(axis=0)) / data.std(axis=0)
        self.y = y
        self.hidden_units = hidden_units
        self.samples, self.input_dim, self.output_dim = data.shape[0], data.shape[1], np.unique(y).size

    def relu(self, x: np.array) -> np.array:
        return np.maximum(0, x)

    def sample_weights(self) -> np.array:
        return {
            'layer_0': {
                'weights': np.random.normal(0, 0.1, size=(self.input_dim, self.hidden_units)), 
                'bias': np.random.normal(0, 0.1, size=(self.hidden_units))
            }, 
            'layer_1': {
                'weights': np.random.normal(0, 0.1, size=(self.hidden_units, self.output_dim)), 
                'bias': np.random.normal(0, 0.1, size=(self.output_dim))
            }
        }

    def add_noise(self, params: Dict[str, np.array], eps: float) -> Dict[str, np.array]:
        layer_0 = {key: value + eps * np.random.normal(0, 1, size=value.shape) for key, value in params['layer_0'].items()}
        layer_1 = {key: value + eps * np.random.normal(0, 1, size=value.shape) for key, value in params['layer_1'].items()}
        return {'layer_0': layer_0, 'layer_1': layer_1}

    def size(self, params: Dict[str, np.array]) -> int:
        total_params = 0
        for layer in params.values():
            for key, value in layer.items():
                if key == 'weights': total_params += np.multiply(*value.shape)
                else: total_params += value.shape[0]
        return total_params

    def cross_entropy(self, y: np.array, p: np.array) -> float:
        log_likelihood = -np.log(p[np.arange(self.samples), y])
        loss = np.sum(log_likelihood) / self.input_dim
        return loss

    def J(self, params: Dict[str, np.array]) -> float:
        probs = self(self.data_norm, params)
        return self.cross_entropy(self.y, probs)

    def __call__(self, X: np.array, params: Dict[str, np.array]) -> np.array:
        weights_0, bias_0 = params['layer_0']['weights'], params['layer_0']['bias']
        weights_1, bias_1 = params['layer_1']['weights'], params['layer_1']['bias']
        
        hidden = self.relu(X @ weights_0 + np.tile(bias_0, (X.shape[0], 1)))
        probs = softmax(hidden @ weights_1 + np.tile(bias_1, (X.shape[0], 1)), axis=1)
        return probs