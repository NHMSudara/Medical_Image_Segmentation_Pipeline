# models/unet.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

def build_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c2)
    c3 = concatenate([c3, c1])

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c3)

    return Model(inputs, outputs)
