from sklearn.decomposition import PCA

import numpy as np
from sklearn.metrics import mean_squared_error

POSSIBLE_TRANSFORMATIONS: np.ndarray = np.array([[-1, 1], [1, -1], [-1, -1]], dtype=np.float32)


class WrappedPCA:
    def __init__(self, init):
        self._init = init
        self._pca = PCA(n_components=2)

    def fit_transform(self, features):
        projection = self._pca.fit_transform(features)
        if isinstance(self._init, str) and self._init == 'random':
            return projection

        lowest_mse = mean_squared_error(self._init, projection)
        best_projection = projection
        for transformation in POSSIBLE_TRANSFORMATIONS:
            transformed_projection = projection * transformation
            mse = mean_squared_error(self._init, transformed_projection)
            if mse < lowest_mse:
                lowest_mse = mse
                best_projection = transformed_projection
        return best_projection
