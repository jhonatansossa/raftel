from utils import get_kedro_context

import math
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

import optuna
from packaging import version

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, 
    Conv2D, 
    MaxPool2D, 
    Dense, 
    Dropout, 
    Flatten, 
    Conv1D, 
    Bidirectional,
    GRU,
    concatenate,
    SpatialDropout1D,
    LSTM,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    Reshape)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import Metric
from keras.optimizers import Adam,SGD,RMSprop
from keras.regularizers import l2

if len(sys.argv) != 2:
    raise ValueError("Please provide the study name as the only argument")

study_name = sys.argv[1]

context = get_kedro_context()
augmented_data = context.catalog.load("windowed_augmented_data_aras_A_3@pickle")

X_train = augmented_data['augmented_X_train']
y_train = augmented_data['augmented_y_train']

X_test = augmented_data['X_test']
y_test = augmented_data['y_test']

X_val = augmented_data['X_val']
y_val = augmented_data['y_val']

if version.parse(tf.__version__) < version.parse("2.11.0"):
    raise RuntimeError("tensorflow>=2.11.0 is required for this example.")

BATCHSIZE = 128
CLASSES = 27
EPOCHS = 10


def create_model(trial):
    # We optimize the numbers of layers, their units and weight decay parameter.
    n_CNN_layers = trial.suggest_int("n_CNN_layers", 1, 3)
    n_Dense_layers = trial.suggest_int("n_Dense_layers", 1, 3)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.4)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
    kernel_size = trial.suggest_categorical('kernel_size', [(10, 5), (20, 5), (20, 10)])
    
    model = tf.keras.Sequential()
    model.add(Input(shape=input_shape))
    for i in range(n_CNN_layers):
        filters = trial.suggest_int("n_filters_l{}".format(i), 32, 256, log=True)
        model.add(
            Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                activation="relu",
                padding="valid"
            )
        )
        model.add(
            Dropout(
                dropout_rate
            )
        )

    model.add(
            Flatten()
        )
 
    for i in range(n_Dense_layers):
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 64, 256, log=True)
        model.add(
            tf.keras.layers.Dense(
                num_hidden,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            )
        )
        model.add(
            Dropout(
                dropout_rate
            )
        )
    model.add(
        tf.keras.layers.Dense(
            CLASSES, 
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
            )
    )
    return model


def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["weight_decay"] = trial.suggest_float("rmsprop_weight_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer


def learn(model, optimizer, dataset, mode="eval"):
    accuracy = tf.metrics.Accuracy("accuracy", dtype=tf.float32)
    f_score = tf.keras.metrics.F1Score(
        average='weighted'
    )

    for batch, (samples, labels) in enumerate(dataset):
        with tf.GradientTape() as tape:
            logits = model(samples, training=(mode == "train"))
            loss_value = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(labels, tf.float32), logits=logits
                )
            )
            if mode == "eval":
                f_score(
                    tf.cast(tf.greater(logits, 0.5), tf.float32), tf.cast(labels, tf.float32)
                )
            else:
                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables))

    if mode == "eval":
        return f_score


def transform_dataset(x_train, y_train, x_valid, y_valid):
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(x_train.shape[0]).batch(BATCHSIZE)

    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    valid_ds = valid_ds.shuffle(x_valid.shape[0]).batch(BATCHSIZE)
    return train_ds, valid_ds


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    # Get MNIST data.
    train_ds, valid_ds = transform_dataset(X_train, y_train, X_val, y_val)

    # Build model and optimizer.
    model = create_model(trial)
    optimizer = create_optimizer(trial)

    # Training and validating cycle.
    # with tf.device("/cpu:0"):
    for _ in tqdm(range(EPOCHS)):
        learn(model, optimizer, train_ds, "train")

    f_score = learn(model, optimizer, valid_ds, "eval")

    # Return last validation accuracy.
    return f_score.result()

input_shape = (180, 28, 1)
study = optuna.create_study(
    direction="maximize",
    study_name=study_name,
    storage=f"mysql+pymysql://root:raftel@localhost/{study_name}",
    load_if_exists=True
)
study.optimize(objective, n_trials=20)

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

