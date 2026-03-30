import tensorflow as tf
import numpy as np


def load_cifar10():
    """
    Charge CIFAR-10, normalise les pixels dans [0, 1]
    et retourne les splits train/test.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalisation : pixels entre 0 et 1
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # y reste en entiers (SparseCategoricalCrossentropy l'attend ainsi)
    y_train = y_train.astype("int32")
    y_test  = y_test.astype("int32")

    print(f"Train : {x_train.shape} | Test : {x_test.shape}")
    print(f"Labels train : {y_train.shape} | Labels test : {y_test.shape}")

    return (x_train, y_train), (x_test, y_test)


def make_datasets(x_train, y_train, x_test, y_test, batch_size=64):
    """
    Construit des tf.data.Dataset optimisés pour l'entraînement.
    - shuffle + batch + prefetch pour le train
    - batch + prefetch pour le test
    """
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=10_000)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_ds, test_ds


# Noms des 10 classes CIFAR-10
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]