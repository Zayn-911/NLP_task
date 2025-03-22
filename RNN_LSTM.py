import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

num_words = 10000  
max_length = 500   

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=max_length),  
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),  
    Dense(1, activation='sigmoid')  
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))


loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")


model.save("sentiment_model.h5")

print("Model training completed and saved as 'sentiment_model.h5'")
