
# ============================================================
# PROJECT: HANDWRITTEN DIGIT RECOGNIZER USING CNN
# Dataset: MNIST
# Deep Learning Framework: TensorFlow and Keras
# ============================================================

# 1. IMPORT REQUIRED LIBRARIES

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense
)

print("TensorFlow Version:", tf.__version__)
print("Libraries imported successfully!")


# 2. LOAD MNIST DATASET

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("\nDataset Loaded Successfully!")
print("Training Images:", x_train.shape)
print("Training Labels:", y_train.shape)
print("Testing Images:", x_test.shape)
print("Testing Labels:", y_test.shape)


# 3. DISPLAY SAMPLE IMAGES

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')

plt.suptitle("Sample Handwritten Digits")
plt.tight_layout()
plt.show()


# 4. DATA PREPROCESSING

# Normalize pixel values between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape images for CNN input
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("\nData Preprocessing Completed!")
print("Training Data Shape:", x_train.shape)
print("Testing Data Shape:", x_test.shape)


# 5. BUILD CNN MODEL

model = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation='relu'),

    Dense(10, activation='softmax')
])

print("\nCNN Model Created Successfully!")

model.summary()


# 6. COMPILE THE MODEL

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Compiled Successfully!")


# 7. TRAIN THE MODEL

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

print("\nModel Training Completed!")


# 8. EVALUATE THE MODEL

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nMODEL EVALUATION RESULTS")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")


# 9. PLOT MODEL ACCURACY

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("CNN Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# 10. PLOT MODEL LOSS

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("CNN Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# 11. PREDICT HANDWRITTEN DIGITS

predictions = model.predict(x_test[:10])

plt.figure(figsize=(12, 5))

for i in range(10):
    predicted_digit = np.argmax(predictions[i])

    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')

    plt.title(
        f"Actual: {y_test[i]}\nPredicted: {predicted_digit}"
    )

    plt.axis('off')

plt.suptitle("Handwritten Digit Predictions")
plt.tight_layout()
plt.show()


# 12. PREDICT A SINGLE DIGIT WITH CONFIDENCE

index = 100

image = x_test[index]

prediction = model.predict(
    image.reshape(1, 28, 28, 1),
    verbose=0
)

predicted_digit = np.argmax(prediction)
confidence = np.max(prediction) * 100

plt.figure(figsize=(4, 4))
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted Digit: {predicted_digit}")
plt.axis('off')
plt.show()

print("\nSINGLE IMAGE PREDICTION")
print("Actual Digit:", y_test[index])
print("Predicted Digit:", predicted_digit)
print("Confidence:", confidence, "%")


# 13. SAVE TRAINED MODEL

model.save("digit_model.keras")

print("\nModel Saved Successfully!")
print("File Name: digit_model.keras")


# 14. PROJECT COMPLETION

print("\n===================================")
print("HANDWRITTEN DIGIT RECOGNIZER")
print("PROJECT COMPLETED SUCCESSFULLY!")
print("===================================")
