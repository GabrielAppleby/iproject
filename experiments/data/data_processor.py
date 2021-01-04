import pathlib
from itertools import product
from typing import Tuple, List, NamedTuple, Collection

import numpy as np
import umap
from sklearn.datasets import load_iris, fetch_openml, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

DATA_FOLDER: str = pathlib.Path(__file__).parent

IRIS: str = 'iris'
ECOLI: str = 'ecoli'
WINE: str = 'wine'
FILE_NAME_TEMPLATE: str = '{}.npz'
UNSPLIT_FILE_NAME_TEMPLATE: str = '{}_unsplit.npz'

POSSIBLE_SCALING_VALUES: List[float] = [.25, .5, .75, 1]


class DataSplit(NamedTuple):
    """
    A datasplit consisting of features and projections.
    """
    features: np.ndarray
    projections: np.ndarray


class DataSet(NamedTuple):
    """
    A dataset, consisting of three data splits. Train, val, and test.
    """
    train: DataSplit
    val: DataSplit
    test: DataSplit


class UnsplitDataSet(NamedTuple):
    """
    A db dataset.
    """
    features: np.ndarray
    targets: np.ndarray


def load_data(name: str) -> DataSet:
    """
    Loads the processed data set.
    :param name: The properly formatted files' name.
    :return: The training, val, and test data.
    """
    data = np.load(pathlib.Path(DATA_FOLDER, FILE_NAME_TEMPLATE.format(name)))

    projections_train = data['projections_train']
    projections_val = data['projections_val']
    projections_test = data['projections_test']

    train = DataSplit(data['features_train'], projections_train)
    val = DataSplit(data['features_val'], projections_val)
    test = DataSplit(data['features_test'], projections_test)

    return DataSet(train, val, test)


def load_unsplit_data(name: str) -> UnsplitDataSet:
    """
    Loads the processed data set.
    :param name: The properly formatted files' name.
    :return: The training, val, and test data.
    """
    data = np.load(pathlib.Path(DATA_FOLDER, UNSPLIT_FILE_NAME_TEMPLATE.format(name)))

    full_split = UnsplitDataSet(data['features'], data['targets'])

    return full_split


def fetch_iris() -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetches the iris dataset from another source.
    :return: The features and labels matrices of the dataset.
    """
    features, labels = load_iris(return_X_y=True)
    return features, labels


def fetch_wine() -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetches the wine dataset from another source.
    :return: The features and labels matrices of the dataset.
    """
    features, labels = load_wine(return_X_y=True)
    print(labels)
    return features, labels


def fetch_ecoli() -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetches the ecoli dataset from another source.
    :return: The features and labels matrices of the dataset.
    """
    features, labels = fetch_openml('ecoli', version=1, return_X_y=True)
    le = LabelEncoder()
    return features, le.fit_transform(labels)


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


def repeat_n_times(coll: Collection, n: int):
    """
    Repeats a given collection n times turning it into an n X d array, where d is the original size
    of the collection.
    :param coll: The collection to repeat.
    :param n: The number of times to repeat the collection.
    :return: The n X d array.
    """
    return np.repeat(np.array(coll).reshape(-1, len(coll)), n, axis=0)


def scale_data(
        features: np.ndarray,
        possible_scaling_values: List[float]) -> Tuple[List[np.ndarray], List[np.array]]:
    """
    Scale the data by all products of the scaling values.
    :param features: The features to scale. n X d.
    :param possible_scaling_values: The possible scaling values. Can be any number of values v.
    :return: All variations of the scaled features (n*product(v, d) - 1, d)., and all scalings
    (n*product(v, d), d). There is one fewer scaled features since we don't count the "unscaled
    scaling" (1, 1, 1, 1).
    """
    n, d = features.shape
    scaled_features = []
    scalings = []

    unscaled_scaling = (1,) * d
    scalings.append(repeat_n_times(unscaled_scaling, n))

    for scaling in product(possible_scaling_values, repeat=d):
        if scaling != unscaled_scaling:
            scaled_features.append(features * scaling)
            scalings.append(repeat_n_times(scaling, n))
    return scaled_features, scalings


def project_data(unscaled_features: np.ndarray,
                 scaled_features: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Project the data, using the a projection of the unscaled features as an initial embedding.
    :param unscaled_features: The unscaled features used to create the initial embedding.
    :param scaled_features: All of the scaled features to project.
    :return: The unscaled projection used as the initial embedding, and all of the scaled
    projections.
    """
    projected_scaled_features = []
    unscaled_projection = umap.UMAP(random_state=42).fit_transform(unscaled_features)
    for features in tqdm(scaled_features):  # type: np.ndarray
        projection = umap.UMAP(init=unscaled_projection, random_state=42).fit_transform(features)
        projected_scaled_features.append(projection)
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
        train_test_split(features, projections, shuffle=True, test_size=0.4, random_state=42)
    features_val, features_test, projections_val, projections_test = \
        train_test_split(features_both,
                         projections_both,
                         shuffle=True,
                         test_size=0.5,
                         random_state=42)

    return features_train, \
           features_val, \
           features_test, \
           projections_train, \
           projections_val, \
           projections_test


def main():
    # datasets: List[Tuple] = [(fetch_iris, IRIS)]
    datasets: List[Tuple] = [(fetch_ecoli, ECOLI), (fetch_wine, WINE)]
    np.random.seed(42)

    for dataset_fnc, name in datasets:
        features, targets = dataset_fnc()
        features, targets = standardize_datatypes(features, targets)
        features = normalize_data(features)

        np.savez(pathlib.Path(DATA_FOLDER, UNSPLIT_FILE_NAME_TEMPLATE.format(name)),
                 features=features,
                 targets=targets)

        scaled_features, all_scalings = scale_data(features, POSSIBLE_SCALING_VALUES)
        projected, projected_scaled = project_data(features, scaled_features)

        repeated_features = np.concatenate([features] * len(all_scalings), axis=0)
        nn_features = np.concatenate((repeated_features, np.concatenate(all_scalings, axis=0)), axis=1)
        nn_projections = np.concatenate([projected] + projected_scaled, axis=0)
        nn_projections = normalize_data(nn_projections)

        features_train, \
        features_val, \
        features_test, \
        projections_train, \
        projections_val, \
        projections_test = train_val_test_split(nn_features, nn_projections)

        np.savez(pathlib.Path(DATA_FOLDER, FILE_NAME_TEMPLATE.format(name)),
                 features_train=features_train,
                 features_val=features_val,
                 features_test=features_test,
                 projections_train=projections_train,
                 projections_val=projections_val,
                 projections_test=projections_test)


if __name__ == '__main__':
    main()
