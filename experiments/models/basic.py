from tensorflow import keras


def build_basic_model():
    model = keras.Sequential()
    # for num_neurons in [512, 512, 512, 512, 288, 288]:
    for num_neurons in [320, 256, 352]:
        model.add(keras.layers.Dense(units=num_neurons,
                                     activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(.25))
    model.add(keras.layers.Dense(2, activation='linear'))
    model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mae'])
    return model
