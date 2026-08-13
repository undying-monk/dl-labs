import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, GlobalAveragePooling2D, RandomFlip, RandomRotation, BatchNormalization

def build_model(hp):
    model = Sequential()
    # preprocessing
    model.add(RandomFlip("horizontal"))
    model.add(RandomRotation(0.1))
    model.add(Rescaling(1./255, input_shape=(150, 150, 3)))
    
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=hp.Int('units',min_value=64,max_value=128,step=64),activation='relu'))

    # model.add(tf.keras.layers.Dropout(hp.Float(
    #     "dropout",
    #     min_value=0.0,
    #     max_value=0.5,
    #     step=0.1
    # )))

    model.add(Dense(units=2,activation='softmax'))

    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=3e-3, sampling="log")
    weight_decay = hp.Float('weight_decay', min_value=1e-5, max_value=1e-3, sampling="log")
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate = learning_rate,
        weight_decay = weight_decay,
    )

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy,
                  metrics=['accuracy'])

    return model