import os
import pickle
import shutil

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras_tuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# setup and globals
BALANCE_METHOD = "under"  # "under", "over", "none"
RANDOM_SEED = None
TR_EPOCHS = 20

train_dir = "../../data/train"
test_dir = "../../data/test"
train_labels = pd.read_csv("../train_labels.csv")
train_labels["id"] = train_labels["id"] + ".tif"


# Helper Functions
def balance(train_labels, method):
    if method == "under":
        sample_size = min(train_labels.label.value_counts())
        df_0 = train_labels[train_labels.label == 0].sample(sample_size)
        df_1 = train_labels[train_labels.label == 1].sample(sample_size)
        train_labels = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)
        train_labels = train_labels.sample(frac=1).reset_index(drop=True)
    elif method == "over":
        sample_size = max(train_labels.label.value_counts())
        difference = (
            train_labels.label.value_counts()[0] - train_labels.label.value_counts()[1]
        )
        if difference < 0:
            df_0 = train_labels[train_labels.label == 0].sample(
                sample_size, replace=True
            )
            df_1 = train_labels[train_labels.label == 1]
        elif difference > 0:
            df_0 = train_labels[train_labels.label == 0]
            df_1 = train_labels[train_labels.label == 1].sample(
                sample_size, replace=True
            )
        else:
            return train_labels

        train_labels = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)
        train_labels = train_labels.sample(frac=1).reset_index(drop=True)
    elif method == "none":
        pass

    return train_labels


def generate_image_data(train_df, valid_df):
    train_datagen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    valid_datagen = ImageDataGenerator(rescale=1 / 255)

    train_df["label"] = train_df["label"].astype(str)
    valid_df["label"] = valid_df["label"].astype(str)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col="id",
        y_col="label",
        target_size=(96, 96),
        batch_size=32,
        class_mode="binary",
    )

    validation_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=train_dir,
        x_col="id",
        y_col="label",
        target_size=(96, 96),
        batch_size=32,
        class_mode="binary",
    )

    return train_generator, validation_generator


def display_img_sample():
    sample = train_labels.sample(n=9).reset_index()

    plt.figure(figsize=(3, 3))

    for i, row in sample.iterrows():
        img = mpimg.imread(f"{train_dir}/{row.id}.tif")
        label = row.label

        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.text(0, -5, f"Class {label}", color="k")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def build_model(hp):
    model = Sequential()

    model.add(
        Conv2D(  # noqa: F405
            filters=hp.Int("conv_1_filter", min_value=32, max_value=128, step=16),
            kernel_size=hp.Choice("conv_1_kernel", values=[3, 5]),
            activation="relu",
            input_shape=(96, 96, 3),
        )
    )
    model.add(
        Conv2D(
            filters=hp.Int("conv_2_filter", min_value=32, max_value=128, step=16),
            kernel_size=hp.Choice("conv_2_kernel", values=[3, 5]),
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            filters=hp.Int("conv_3_filter", min_value=32, max_value=128, step=16),
            kernel_size=hp.Choice("conv_3_kernel", values=[3, 5]),
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            filters=hp.Int("conv_4_filter", min_value=32, max_value=128, step=16),
            kernel_size=hp.Choice("conv_4_kernel", values=[3, 5]),
            activation="relu",
        )
    )

    model.add(Flatten())
    model.add(
        Dense(
            units=hp.Int("dense_1_units", min_value=32, max_value=128, step=16),
            activation="relu",
        )
    )
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_plot(history):
    # plt.plot(history["accuracy"])
    # plt.plot(history["val_accuracy"])
    # plt.title("Model Accuracy")
    # plt.ylabel("Accuracy")
    # plt.xlabel("Epoch")
    # plt.legend(["train", "validation"], loc="upper left")
    # plt.savefig("accuracy.png")
    # plt.clf()
    #
    # plt.plot(history["loss"])
    # plt.plot(history["val_loss"])
    # plt.title("Model Loss")
    # plt.ylabel("Loss")
    # plt.xlabel("Epoch")
    # plt.legend(["train", "validation"], loc="upper left")
    # plt.savefig("loss.png")
    # plt.clf()

    acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    loss = history["loss"]
    val_loss = history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.ylim([min(plt.ylim()), 1])
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Entropy")
    plt.ylim([0, max(plt.ylim())])
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    print("Saving training_results.png")
    plt.savefig("training_results.png")


# ----------------------- Main -----------------------

print("Number of images in train set", len(os.listdir(train_dir)))
print("Number of images in test set", len(os.listdir(test_dir)))

# display_img_sample()

print("------------------------ train_labels.csv -----------------------")
print(f"train_labels shape: {train_labels.shape}")
print(train_labels.head(), "\n")
print(train_labels["label"].value_counts(), "\n")
counts = train_labels.label.value_counts() / len(train_labels) * 100
print(f"ratio: {counts[0]:.0f}:{counts[1]:.0f}")

# Balance the classes of the training set
print(f"---- Balance({BALANCE_METHOD}) the classes of the training set -----")
train_labels = balance(train_labels, BALANCE_METHOD)
print(train_labels["label"].value_counts(), "\n")
counts = train_labels.label.value_counts() / len(train_labels) * 100
print(f"ratio: {counts[0]:.0f}:{counts[1]:.0f}")

# Splitting the data into train and validation sets
print("---------- Split train into train and validation sets ----------")
train, val = train_test_split(
    train_labels,
    stratify=train_labels.label,
    test_size=0.2,
    random_state=RANDOM_SEED,
)
# remove elements from train and val so that they are divisible by batch size
train = train.iloc[: (len(train) // 32) * 32]
val = val.iloc[: (len(val) // 32) * 32]

print(f"train shape: {train.shape}")
print(f"validation shape: {val.shape}")

print("-------------------- Generate image data ---------------------")
train_generator, validation_generator = generate_image_data(train, val)

TR_STEPS = len(train_generator)  # // 32  # batch size
VA_STEPS = len(validation_generator)  # // 32

print("Number of batches in the training set:", TR_STEPS)
print("Number of batches in the validation set:", VA_STEPS)

print("----------------------- Build the model -----------------------")
if os.path.isfile("model_with_hps.keras") and os.path.isfile("best_hps.pkl"):
    model = tf.keras.models.load_model("model_with_hps.keras")
    with open("best_hps.pkl", "rb") as f:
        best_hps = pickle.load(f)
else:
    shutil.rmtree("tuner", ignore_errors=True)

    tuner = RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=3,
        directory="tuner",
        project_name="kaggle",
    )

    tuner.search(
        train_generator,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=VA_STEPS,
    )

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    # save best_hps to file
    with open("best_hps.pkl", "wb") as f:
        pickle.dump(best_hps, f)

    model = tuner.hypermodel.build(best_hps)
    model.save("model_with_hps.keras")

model.summary()

if os.path.isfile("model_trained.keras") and os.path.isfile("history.pkl"):
    model = tf.keras.models.load_model("model_trained.keras")
    with open("history.pkl", "rb") as f:
        history = pickle.load(f)
else:
    print("----------------------- Train the model -----------------------")
    history = model.fit(
        train_generator,
        epochs=TR_EPOCHS,
        # steps_per_epoch=TR_STEPS,
        validation_data=validation_generator,
        # validation_steps=VA_STEPS,
    )
    print("----------------------- Save the model -----------------------")
    model.save("model_trained.keras")
    # save history to file
    with open("history.pkl", "wb") as f:
        pickle.dump(history.history, f)
    print(pd.DataFrame(history.history))
    history = history.history

print("--------------- Save plot of training results ----------------")
# Plot the results
save_plot(history)
