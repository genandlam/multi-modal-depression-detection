from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

# from plot_metrics import plot_accuracy, plot_loss, plot_roc_curve

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
from keras.wrappers.scikit_learn import KerasClassifier

import json
import time

np.random.seed(15)  # for reproducibility

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
print("Using GPU: ", K.tensorflow_backend._get_available_gpus())

K.set_image_dim_ordering("tf")


def retrieve_file(file_name):
    path = "../data/raw_data/processed_data_aug/"
    outfile = path + file_name
    X = np.load(outfile)
    X = X["arr_0"]
    return X


def retrieve_file_no_aug(file_name):
    path = "../data/raw_data/processed_topic/"
    outfile = path + file_name
    X = np.load(outfile)
    X = X["arr_0"]
    return X


def preprocess(X, max_len, num_bins):
    """
    Preprocess input Xs to a numpy array where every training sample is zero padded
    to constant time dimension (max_len) and contains num_bins frequency bins.

    Args:
        X: numpy array of numpy arrays (X), each of which is of different time dimension
           but same mel dimension (usually 128)
        max_len: Length up to which each np.array in X is padded with 0s
        num_bins: Constant mel dimension

    Returns:
        X_proc: single numpy array of shape (X.shape[0], max_len, num_bins), which is fed into 1D CNN
    """
    X_proc = np.zeros([X.shape[0], max_len, num_bins])
    for idx, x in enumerate(X):
        if x.shape[0] < max_len:
            # Pad sequence (only in time dimension) with 0s
            x = np.pad(
                x, pad_width=((0, max_len - x.shape[0]), (0, 0)), mode="constant"
            )
        else:
            # Trim sequence to be within max_len timesteps
            x = x[:max_len, :]
        # Update processed sequences
        X_proc[idx, :, :] = x
    return X_proc


def cnn(
    X_train,
    y_train,
    X_test,
    y_test,
    max_len,
    num_bins,
    nb_classes,
):

    NUM_CONV_LAYERS = 4
    NUM_DENSE_LAYERS = 1
    NUM_FILTERS = 64
    KERNEL_SIZE = 125
    STRIDES = 1
    L2_LAMBDA = 0.01
    DROPOUT = 0.5
    POOL_SIZE = 2
    DENSE_SIZE = 64
    BATCH_SIZE = 30
    EPOCHS = 1

    inputs = Input(batch_shape=(None, max_len, num_bins))
    x = inputs
    for layer_count in range(NUM_CONV_LAYERS - 1):
        x = Conv1D(
            filters=NUM_FILTERS,
            kernel_size=KERNEL_SIZE,
            padding="valid",
            strides=STRIDES,
            dilation_rate=1,
            activation="linear",
            kernel_regularizer=regularizers.l2(L2_LAMBDA),
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(DROPOUT)(x)
        x = MaxPooling1D(pool_size=POOL_SIZE)(x)
    # Final Conv1D layer doesn't undergo MaxPooling1D
    x = Conv1D(
        filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        padding="valid",
        strides=STRIDES,
        dilation_rate=1,
        activation="linear",
        kernel_regularizer=regularizers.l2(L2_LAMBDA),
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(DROPOUT)(x)

    avgpool = GlobalAveragePooling1D()(x)
    maxpool = GlobalMaxPooling1D()(x)
    l2pool = Lambda(lambda a: K.l2_normalize(K.sum(a, axis=1)))(x)
    x = Concatenate()([avgpool, maxpool, l2pool])

    for layer_count in range(NUM_DENSE_LAYERS):
        x = Dense(
            units=DENSE_SIZE,
            activation="relu",
            kernel_regularizer=regularizers.l2(L2_LAMBDA),
        )(x)
        x = Dropout(DROPOUT)(x)

    # Create CNN featurizer model
    cnn_featurizer = Model(inputs=inputs, outputs=x)

    # Create CNN predictor model
    x = Dense(nb_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=x)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(lr=6.25e-4),
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=0,
    )
    # Evaluate accuracy on test and train sets
    score_train = model.evaluate(X_train, y_train, verbose=0)

    score_test = model.evaluate(X_test, y_test, verbose=0)

    return model, cnn_featurizer, history


if __name__ == "__main__":

    # topic model files
    X_train_no_aug = retrieve_file_no_aug("train_samples.npz")
    y_train_no_aug = retrieve_file_no_aug("train_labels.npz")

    X_train3 = retrieve_file("train_samples3.npz")
    y_train3 = retrieve_file("train_labels3.npz")

    X_train = np.concatenate((X_train_no_aug, X_train3), axis=0)
    y_train = np.concatenate((y_train_no_aug, y_train3), axis=0)

    # Number of mel bins in training samples
    NB_CLASSES = 2
    NUM_BINS = X_train[0].shape[1]
    # Maximum time duration among training samples
    MAX_LEN = np.max([X_train[i].shape[0] for i in range(len(X_train))])

    X_train = preprocess(X_train, max_len=MAX_LEN, num_bins=NUM_BINS)
    X_test = preprocess(X_test, max_len=MAX_LEN, num_bins=NUM_BINS)

    Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    model, cnn_featurizer, history = cnn(
        X_train, Y_train, X_test, Y_test, MAX_LEN, NUM_BINS, NB_CLASSES
    )

    y_test_pred = np.argmax(model.predict(X_test), axis=-1)
    f1_score = sklearn.metrics.f1_score(y_test, y_test_pred)
    print("f1 score: {}".format(f1_score))
    model.save("cnn_augm_run200_f1{}.h5".format(f1_score))
    cnn_featurizer.save("cnn_featurizer_augm_run200_{}f1.h5".format(f1_score))
