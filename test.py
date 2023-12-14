import os
import pickle
import shutil

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


if os.path.isfile("model_trained.keras") and os.path.isfile("history.pkl"):
    model = tf.keras.models.load_model("model_trained.keras")
    with open("history.pkl", "rb") as f:
        history = pickle.load(f)
else:
    print("No model found")
    quit()

test = "../../data/test"
df = pd.read_csv("../sample_submission.csv", dtype=str)
df["id"] = df["id"] + ".tif"

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=df,
    directory=test,
    x_col="id",
    y_col=None,
    batch_size=32,
    shuffle=False,
    class_mode=None,
    target_size=(96, 96),
)

predictions = model.predict(test_generator, verbose=1)
df["label"] = predictions
df["label"] = df["label"].apply(lambda x: 1 if x >= 0.5 else 0)
df["id"] = df["id"].str.replace(".tif", "")
df.to_csv("submission.csv", index=False)
