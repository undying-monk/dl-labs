import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential

def build_model(hp: kt.HyperModel):
    model = Sequential()
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=3e-3, sampling="log")
    weight_decay = hp.Float('weight_decay', min_value=1e-5, max_value=1e-3, sampling="log")
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate,weight_decay=weight_decay)

    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(units=128,activation='relu'))
    model.add(Dense(units=10,activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy,
                  metrics=['accuracy'])

    return model