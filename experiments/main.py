from kerastuner import BayesianOptimization
from tensorflow import keras

from data.data_processor import IRIS_FILE_NAME, load_data
from models.tuning import build_model


def main():
    dataset = load_data(IRIS_FILE_NAME)
    tuner = BayesianOptimization(
        build_model,
        objective='val_loss',
        max_trials=3,
        directory='tuning_results',
        project_name='ipca_iris')

    tuner.search(dataset.train.features,
                 dataset.train.projections,
                 epochs=300,
                 validation_data=(dataset.val.features, dataset.val.projections),
                 callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])


if __name__ == '__main__':
    main()
