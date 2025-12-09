import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_ANN(hp, input_dim):
    model=keras.models.Sequential()

    model.add(layers.Input(shape=(input_dim,)))

    for i in range(hp.Int("num_layers", 1, 5)):
      model.add(
          layers.Dense(
              units=hp.Int(f"units_{i}", min_value=32, max_value=128, step=32),
              activation="relu",
              kernel_regularizer=keras.regularizers.l2(
                  hp.Choice("l2_reg", [0.0, 0.001, 0.01])
              )
          )
      )
      model.add(layers.Dropout(hp.Float("dropout", 0.0, 0.3, step=0.1)))

    model.add(layers.Dense(1,activation='linear'))

    lr = hp.Choice("lr", [1e-2, 1e-3, 5e-4, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

    return model
