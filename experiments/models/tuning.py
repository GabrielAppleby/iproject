from kerastuner import HyperParameters
from tensorflow import keras


def build_model(hp: HyperParameters):
    model = keras.Sequential()
    has_bn = hp.Choice('has_bn', [True, False])
    has_do = hp.Choice('has_do', [True, False])
    dropout = hp.Float('dropout', min_value=0.0, max_value=.5, step=.25)
    for i in range(hp.Int('num_layers', 2, 8)):
        model.add(keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                  min_value=4,
                                                  max_value=32,
                                                  step=4),
                                     activation='relu'))
        if has_bn:
            model.add(keras.layers.BatchNormalization())
        if has_do:
            model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(2, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model
