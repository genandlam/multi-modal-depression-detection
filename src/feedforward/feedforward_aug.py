from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import sklearn
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Model
from keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    Activation,
    Dropout,
    MaxPooling1D,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Lambda,
    Concatenate,
    Dense,
    regularizers,
)
from keras.utils import np_utils
from keras import backend as K
from keras import optimizers, activations
from keras.optimizers import Adam
import json

import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
print("Using GPU: ", K.tensorflow_backend._get_available_gpus())

K.set_image_dim_ordering("tf")


def retrieve_file(file_name):
    path = "./"
    outfile = path + file_name
    X = np.load(outfile)
    X = X["arr_0"]
    return X


def retrieve_file_no_aug(file_name):
    path = "../data/raw_data/processed_topic/"
    outfile = path + file_name
    X = np.load(outfile, encoding="latin1")
    X = X["arr_0"]
    return X


def feedforward(X_train, y_train, X_test, y_test, class_weight):

    NUM_DENSE_LAYERS = 3
    DROPOUT = 0.1
    DENSE_SIZE = 64
    ACTIVATION = activation_function
    OPTIMIZER = 6.25e-5
    x_shape = X_train.shape[1]

    inputs = Input(batch_shape=(None, x_shape))

    x = inputs

    for layer_count in range(NUM_DENSE_LAYERS):
        x = Dense(units=DENSE_SIZE, activation="linear")(x)
        x = BatchNormalization()(x)
        x = Activation(ACTIVATION)(x)
        x = Dropout(DROPOUT)(x)

    x = Dense(nb_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=x)

    model.compile(
        loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        batch_size=1,
        epochs=5,
        validation_data=(X_test, y_test),
        verbose=0,
    )

    return model


if __name__ == "__main__":

    train_audio_features = np.loadtxt("train_audio_features_augm200.txt")
    print(train_audio_features.shape)
    test_audio_features = np.loadtxt("test_audio_features_augm200.txt")
    print(test_audio_features.shape)

    train_text_features = np.loadtxt("train_text_features_augm200.txt")
    print(train_text_features.shape)
    test_text_features = np.loadtxt("test_text_features_augm200.txt")
    print(test_text_features.shape)

    train_features_audio_text = np.concatenate(
        (train_text_features, train_audio_features), axis=1
    )
    print(train_features_audio_text.shape)
    test_features_audio_text = np.concatenate(
        (test_text_features, test_audio_features), axis=1
    )
    print(test_features_audio_text.shape)

    y_train_no_aug = retrieve_file_no_aug("train_labels.npz")
    y_train3 = retrieve_file("train_labels_200.npz")

    y_train = np.concatenate((y_train_no_aug, y_train3), axis=0)
    y_test = retrieve_file_no_aug("test_labels.npz")

    nb_classes = 2

    # Convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = feedforward(
        train_features_audio_text, Y_train, test_features_audio_text, Y_test, nb_classes
    )

    y_test_pred = np.argmax(model.predict(test_features_audio_text), axis=-1)
    f1_score = sklearn.metrics.f1_score(y_test, y_test_pred)
    print("f1_score:" + f1_score)
