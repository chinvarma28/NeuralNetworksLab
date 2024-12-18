"""
4. Design and implement a CNN model (with 4+ layers of convolutions) to classify multi category image datasets. Use the concept of regularization and dropout while designing the CNN model. Use the Fashion MNIST datasets. Record the Training accuracy and Test accuracy corresponding to the following architectures:
    
    a. Base Model
    
    b. Model with L1 Regularization
    
    c. Model with L2 Regularization
    
    d. Model with Dropout
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1, l2
import matplotlib.pyplot as plt

# Load and preprocess the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Simple function to build different CNN models with optional L1, L2, or Dropout
def build_model(regularizer=None, dropout=None):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=regularizer),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizer),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizer),
        layers.Dense(10, activation='softmax')
    ])
    if dropout:
        model.add(layers.Dropout(dropout))
    return model

# Function to compile, train and evaluate, and return history
def compile_and_train(model, name):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=5, validation_split=0.2)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'{name} Test Accuracy: {test_acc:.4f}')
    return history

# Plot training and validation accuracy and loss
def plot_history(history, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.show()

# Base CNN (No regularization, No dropout)
print("Training Base Model")
base_model = build_model()
base_history = compile_and_train(base_model, "Base Model")
plot_history(base_history, "Base Model")

# L1 Regularization
print("\nTraining L1 Regularization Model")
l1_model = build_model(regularizer=l1(0.001))
l1_history = compile_and_train(l1_model, "L1 Model")
plot_history(l1_history, "L1 Model")

# L2 Regularization
print("\nTraining L2 Regularization Model")
l2_model = build_model(regularizer=l2(0.001))
l2_history = compile_and_train(l2_model, "L2 Model")
plot_history(l2_history, "L2 Model")

# Dropout Model
print("\nTraining Dropout Model")
dropout_model = build_model(dropout=0.5)
dropout_history = compile_and_train(dropout_model, "Dropout Model")
plot_history(dropout_history, "Dropout Model")
