import argparse

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from finetune import Classifier
from finetune.config import (
    get_default_config,
    get_small_model_config,
    get_config,
    GridSearchable,
)

np.random.seed(15)


def get_custom_config():
    # conf = get_default_config()
    conf = get_small_model_config()
    conf.l2_reg = GridSearchable(0.01, [0.01])
    conf.lr = GridSearchable(6.25e-6, [6.25e-5, 6.25e-6])
    conf.n_epochs = GridSearchable(1, [1, 3])
    conf.batch_size = GridSearchable(5, [1, 5, 10])
    conf.class_weights = "linear"
    conf.oversample = False  # Do not use if using class_weights
    conf.low_memory_mode = True
    conf.val_size = 0.0  # Do not perform validation right now
    conf.val_interval = 10000  # Do not perform validation right now
    print("Config: ", conf)
    return conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass experiment type: aug/topic/full")
    parser.add_argument("-e", "--expt", help="Experiment type", default="aug")
    expt = vars(parser.parse_args())["expt"]

    # Load data
    data_train = pd.read_csv("data_train_{}.csv".format(expt))
    print("Training data: {}".format(data_train.shape))
    data_test = pd.read_csv("data_test_{}.csv".format(expt))
    print("Testing data: {}".format(data_test.shape))

    # Create features and targets
    trainX, trainY = np.array(data_train.Text), np.array(data_train.Targets)
    testX, testY = np.array(data_test.Text), np.array(data_test.Targets)

    # Perform grid search
    print("Performing CV grid search")
    best_config = Classifier.finetune_grid_search_cv(
        trainX, trainY, n_splits=3, config=get_custom_config(), eval_fn=f1_score
    )
    print("Best parameters: \n{}".format(best_config))

    # Finetune model
    model = Classifier(config=get_config(**best_config))
    model.finetune(trainX, trainY)

    # Compute metrics
    predY = model.predict(testX)
    accuracy = np.mean(predY == testY)
    class_balance = np.mean(testY)
    print(
        "Test Accuracy: {:0.2f} for a {:0.2f} class balance".format(
            accuracy, class_balance
        )
    )

    print("\nClassification Report:\n")
    print(
        classification_report(
            model.label_encoder.transform(testY), model.label_encoder.transform(predY)
        )
    )

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(testY, predY))

    f1 = f1_score(predY, testY)
    print("\nF1-Score: ", f1)

    model.save("models/model_{}_{}f1".format(expt, f1))
