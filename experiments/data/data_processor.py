import pathlib
from itertools import product
from typing import Tuple, List, NamedTuple

import numpy as np
import umap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA_FOLDER: str = pathlib.Path(__file__).parent

IRIS_FILE_NAME: str = 'iris.npz'
SCALINGS: List[float] = [.25, .5, .75, 1]


class DataSplit(NamedTuple):
    features: np.ndarray
    projections: np.ndarray


class DataSet(NamedTuple):
    """
    A dataset, consiting of three data splits. Train, val, and test.
    """
    train: DataSplit
    val: DataSplit
    test: DataSplit


# class ScaledFeatures(NamedTuple):
#     features: np.ndarray
#     scaling: np.ndarray
#
#
# class ProjectedScaledFeatures(NamedTuple):
#     features: np.ndarray
#     scaling: np.ndarray
#     projection: np.ndarray
#
#
# class ProjectedDataSet(NamedTuple):
#     unscaled_features: np.ndarray
#     unscaled_projection: np.ndarray
#     targets: np.ndarray
#     projected_scaled_features: List[ProjectedScaledFeatures]


def load_data(filename: str) -> DataSet:
    """
    Loads the processed data set.
    :param filename: The properly formatted files' name.
    :return: The training, val, and test data.
    """
    data = np.load(pathlib.Path(DATA_FOLDER, filename))

    projections_train = data['projections_train']
    projections_val = data['projections_val']
    projections_test = data['projections_test']

    train = DataSplit(data['features_train'], projections_train)
    val = DataSplit(data['features_val'], projections_val)
    test = DataSplit(data['features_test'], projections_test)

    return DataSet(train, val, test)


def fetch_iris() -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetches the iris dataset from another source.
    :return: The features and labels matrices of the dataset.
    """
    features, labels = load_iris(return_X_y=True)
    return features, labels


def standardize_datatypes(
        features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make sure all datasets end up with the same datatypes.
    :param features: The features of the dataset.
    :param targets: The targets of the dataset. Can be none.
    :return: The features, targets tuple of the dataset.
    """
    features = features.astype(np.float32)
    targets = targets.astype(np.int32)
    targets = targets.reshape((-1, 1))
    return features, targets


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalizes the data between 0 and 1.
    :param data: The data to normalize.
    :return: The normalized data.
    """
    min = np.min(data, axis=0)
    data = data - min
    max = np.max(data, axis=0)
    data = np.divide(data, max, out=np.zeros_like(data), where=max != 0)
    return data


def scale_data(features: np.ndarray, scalings: List[float]) -> List[np.ndarray]:
    scaled_features = []
    unscaled_scaling = (1, 1, 1, 1)
    for scaling in product(scalings, repeat=4):
        if scaling != unscaled_scaling:
            scaled_features.append(features * scaling)
    return scaled_features


def project_data(unscaled_features: np.ndarray,
                 scaled_features: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    projected_scaled_features = []
    unscaled_projection = umap.UMAP().fit_transform(unscaled_features)
    for features in tqdm(scaled_features):  # type: np.ndarray
        projection = umap.UMAP(init=unscaled_projection, random_state=42).fit(features)
        projected_scaled_features.append(projection.embedding_)
    return unscaled_projection, projected_scaled_features


def train_val_test_split(features: np.ndarray, projections: np.ndarray) -> Tuple:
    """
    Splits the dataset into training, validation, and test data. 60 percent is used for training,
    20 percent for validation, and 20 percent for testing.
    :param features: The features to split.
    :param projections: The projections to split.
    :return: Training, validation, and test sets of the features, embeddings, and projections.
    """
    features_train, features_both, projections_train, projections_both = \
        train_test_split(features, projections, test_size=0.4, shuffle=True, random_state=42)
    features_val, features_test, projections_val, projections_test = \
        train_test_split(features_both,
                         projections_both,
                         test_size=0.5,
                         shuffle=True,
                         random_state=42)

    return features_train, \
           features_val, \
           features_test, \
           projections_train, \
           projections_val, \
           projections_test


def main():
    datasets: List[Tuple] = [(fetch_iris, IRIS_FILE_NAME)]
    np.random.seed(42)

    for dataset_fnc, file_name in datasets:
        features, targets = dataset_fnc()
        features, targets = standardize_datatypes(features, targets)
        features = normalize_data(features)
        scaled_features = scale_data(features, SCALINGS)
        projected, projected_scaled = project_data(features, scaled_features)

        all_features = np.concatenate([features] + scaled_features, axis=0)
        all_projections = np.concatenate([projected] + projected_scaled, axis=0)
        all_projections = normalize_data(all_projections)

        features_train, \
        features_val, \
        features_test, \
        projections_train, \
        projections_val, \
        projections_test = train_val_test_split(all_features, all_projections)

        np.savez(pathlib.Path(DATA_FOLDER, file_name),
                 features_train=features_train,
                 features_val=features_val,
                 features_test=features_test,
                 projections_train=projections_train,
                 projections_val=projections_val,
                 projections_test=projections_test)


if __name__ == '__main__':
    main()
