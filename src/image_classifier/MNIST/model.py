import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Conv2D,BatchNormalization,MaxPool2D,Rescaling, GlobalAveragePooling2D,Input,Reshape
from tensorflow.keras.models import Sequential

def build_cnn_model(hp: kt.HyperModel):
    model = Sequential()
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=3e-3, sampling="log")
    weight_decay = hp.Float('weight_decay', min_value=1e-5, max_value=1e-3, sampling="log")
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate,weight_decay=weight_decay)

    model.add(Input(shape=(28, 28)))
    model.add(Reshape((28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu')) # 26x26x32
    model.add(BatchNormalization())
    model.add(MaxPool2D())
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu')) # 13x13x64
    model.add(BatchNormalization())
    model.add(MaxPool2D()) # 6x6x64

    model.add(GlobalAveragePooling2D()) # 64 params
    model.add(Dense(units=32,activation='relu')) # 32 * 64 + 32 = 2080 parameters
    model.add(Dense(units=10,activation='softmax')) #  32 * 10 + 10 = 330 params

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy,
                  metrics=['accuracy'])

    return model

def build_model(hp: kt.HyperModel):
    model = Sequential()
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=3e-3, sampling="log")
    weight_decay = hp.Float('weight_decay', min_value=1e-5, max_value=1e-3, sampling="log")
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate,weight_decay=weight_decay)

    model.add(Flatten(input_shape=(28,28))) # 784 params
    model.add(Dense(units=128,activation='relu')) # 100,480 params
    model.add(Dense(units=10,activation='softmax')) # 128 * 10 + 10 = 1290 params
    # Total params 101,770

    # model.add(Rescaling(1./255, input_shape=(28, 28, 3)))
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy,
                  metrics=['accuracy'])

    return model
