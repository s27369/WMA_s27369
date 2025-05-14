import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import json


IMG_SIZE = (100, 100)
DATA_DIR = "media/faces"


def get_dataset(DATA_DIR):
    class_names = sorted(os.listdir(DATA_DIR))
    label_map = {name: idx for idx, name in enumerate(class_names)}
    print("Label map:", label_map)
    with open("label_map.json", "w") as f:
        json.dump(label_map, f)

    X = []
    y = []

    for class_name in class_names:
        class_dir = os.path.join(DATA_DIR, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(class_dir, fname)
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0  # normalizacja
                X.append(img)
                y.append(label_map[class_name])

    X = np.array(X)
    y = np.array(y)
    y_cat = to_categorical(y, num_classes=len(class_names))  # one-hot encode

    # Train/val/test split (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_cat, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42,
                                                    stratify=np.argmax(y_temp, axis=1))
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_model(X_train, y_train, X_val, y_val, X_test, y_test):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(y_train.shape[1], activation='softmax')  # Output layer for multi-class
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint("model_best.keras", save_best_only=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop, checkpoint]
    )

    model.save("model_best.keras")


    # Eval
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.2%}")
    return model, history

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = get_dataset(DATA_DIR)
    get_model(X_train, y_train, X_val, y_val, X_test, y_test)
