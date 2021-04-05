import tensorflow as tf
import pathlib

import tensorflowjs as tfjs

JS_MODELS_FOLDER = pathlib.Path(pathlib.Path(__file__).parent, 'js_models')
PY_MODELS_FOLDER = pathlib.Path(pathlib.Path(__file__).parent, 'py_models')

IRIS_JS_FOLDER = pathlib.Path(JS_MODELS_FOLDER, 'iris')
IRIS_PY_FOLDER = pathlib.Path(JS_MODELS_FOLDER, 'iris')


def save_model_for_js(model, dataset_name):
    tfjs.converters.save_keras_model(model, pathlib.Path(JS_MODELS_FOLDER, dataset_name))


def save_model_for_py(model, dataset_name):
    model.save(pathlib.Path(PY_MODELS_FOLDER, dataset_name))


def load_model_for_py(dataset_name) -> tf.keras.Model:
    return tf.keras.models.load_model(pathlib.Path(PY_MODELS_FOLDER, dataset_name))
