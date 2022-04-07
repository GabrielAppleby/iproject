import numpy as np

from kerastuner import BayesianOptimization
from tensorflow import keras

from data.data_processor import IRIS, load_data, load_unsplit_data, DataSplit
from models.basic import build_basic_model
from models.save_utils import save_model_for_js, save_model_for_py, load_model_for_py
from models.tuning import build_model
from visualization.scatterplots import basic_scatter, movement_scatter


def tune(train, val):
    tuner = BayesianOptimization(
        build_model,
        objective='val_loss',
        max_trials=512,
        directory='tuning_results_distances_sampled',
        project_name='ipca_iris_distances_sampled')

    tuner.search(train.features,
                 train.projections,
                 epochs=300,
                 verbose=1,
                 validation_data=(val.features, val.projections),
                 callbacks=[keras.callbacks.EarlyStopping(patience=16, restore_best_weights=True)])
    print(tuner.get_best_hyperparameters()[0].get_config())


def train(train, val) -> keras.Model:
    model = build_basic_model()
    model.fit(train.features,
              train.projections,
              epochs=300,
              validation_data=(val.features, val.projections),
              callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    return model

def create_db_payload(filename):
    dataset = load_unsplit_data(filename)
    features = dataset.features
    model = load_model_for_py(filename)
    newData = []
    for arr in features:
        newData.append(np.concatenate((arr, [1, 1, 1, 1])))
    newData = np.concatenate(newData, axis=0).reshape(-1, 8)
    preds = model.predict(newData)
    np.savez('db_payload_{}_pca.npz'.format(filename),
             features=features,
             targets=dataset.targets,
             projections=preds)

def create_models(filename):
    dataset = load_data(filename)
    model = train(dataset.train, dataset.val)
    model.evaluate(dataset.train.features, dataset.train.projections)
    save_model_for_py(model, filename)
    save_model_for_js(model, filename)


def testing(name):
    dataset = load_unsplit_data(name)
    model = load_model_for_py(name)
    for true, features in zip(dataset.projections[0:20], dataset.features[0:20]):
        preds = model.predict(features)
        movement_scatter(true, preds, dataset.targets, features[0][4:])


def main():
    create_models(IRIS)
    create_db_payload(IRIS)
    # tune(dataset.train, dataset.val)
    # basic_scatter(preds)
    # basic_scatter(projections)


if __name__ == '__main__':
    main()
