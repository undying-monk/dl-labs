import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Conv2D,BatchNormalization,MaxPool2D,Rescaling, GlobalAveragePooling2D,Input,Reshape,Dropout,RandomRotation,RandomContrast, RandomFlip
from tensorflow.keras.models import Sequential

def build_cnn_model(num_classes):
    model = Sequential()
    model.add(Input(shape=(256, 256,3)))
    model.add(Reshape((256, 256, 3)))
    ## data augmentation
    model.add(RandomRotation(factor=0.15)),
    model.add(RandomFlip())
    model.add(RandomContrast(factor=0.1))

    for filters, drop_rate in [(32,0.15), (64,0.2), (128, 0.25)]:
        model.add(Conv2D(filters=filters, kernel_size=(3,3), activation='relu')) # 253x253x32
        model.add(BatchNormalization())
        model.add(Conv2D(filters=filters, kernel_size=(3,3), activation='relu')) # 253x253x32
        model.add(BatchNormalization())

        model.add(MaxPool2D())  
        model.add(Dropout(drop_rate))


    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu')) # 253x253x32
    model.add(BatchNormalization())
    model.add(MaxPool2D()) 

    model.add(GlobalAveragePooling2D()) # 64 params
    model.add(Dense(units=256,activation='relu')) # 32 * 64 + 32 = 2080 parameters
    model.add(Dense(units=int(num_classes),activation='softmax')) #  32 * 10 + 10 = 330 params

    return model

def build_multi_tasks_model(num_plants, num_diseases):
    inputs = Input(shape=(256, 256,3))
    # x = Reshape((256, 256, 3)))

    ## data augmentation
    x = RandomRotation(factor=0.15)(inputs),
    x = RandomFlip("horizontal")(x)
    x = RandomContrast(factor=0.1)(x)

    for filters, drop_rate in [(32,0.15), (64,0.2), (128, 0.25)]:
        x = Conv2D(filters=filters, kernel_size=(3,3), activation='relu')(x) # 253x253x32
        x = BatchNormalization()(x)
        x = Conv2D(filters=filters, kernel_size=(3,3), activation='relu')(x) # 253x253x32
        x = BatchNormalization()(x)

        x = MaxPool2D((x))  
        x = Dropout(drop_rate(x))


    x = Conv2D(filters=256, kernel_size=(3,3), activation='relu')(x) # 253x253x32
    x = BatchNormalization()(x)
    x = MaxPool2D()(x) 

    x = GlobalAveragePooling2D()(x) # 64 params
    x = Dense(units=256,activation='relu')(x) # 32 * 64 + 32 = 2080 parameters

    plant_output = Dense(units=int(num_plants),activation='softmax',name='plant')(x)
    disease_output = Dense(units=int(num_diseases),activation='sigmoid',name='disease')(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs={
            "plant": plant_output,
            "disease": disease_output
    }
)

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
