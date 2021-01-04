from tensorflow import keras


def build_basic_model():
    model = keras.Sequential()
    for num_neurons in [512, 512, 512, 512, 288, 288]:
        model.add(keras.layers.Dense(units=num_neurons,
                                     activation='relu'))
    model.add(keras.layers.Dense(2, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    return model
