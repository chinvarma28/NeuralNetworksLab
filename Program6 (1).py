import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout

# Load and preprocess the IMDB dataset
vocab_size = 20000  # Increased vocabulary size for more word coverage
max_length = 200  # Max length of each review

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Padding sequences to ensure uniform input size
x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post')

# Build the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=256, input_length=max_length),  # Increased embedding dimension
    Bidirectional(LSTM(256, return_sequences=True)),  # Increased units for better capacity
    Dropout(0.5),  # Dropout to reduce overfitting
    Bidirectional(LSTM(128)),  # Second LSTM layer with reduced units
    Dropout(0.5),
    Dense(64, activation='relu'),  # Added dense layer for more representation power
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification for sentiment analysis
])

# Learning rate scheduler to gradually decrease learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Train the model with increased epochs
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model on test data
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
