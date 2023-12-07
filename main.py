import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Larger dataset for training
names = ["Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace", "Henry", "Ivy", "Jack", "Kate", "Leo", "Mia", "Nathan", "Olivia", "Peter", "Quinn", "Rachel", "Sam", "Tom", "Ursula", "Victor", "Wendy", "Xavier", "Yvonne", "Zane"]

# Concatenate all names to create training text
text = " ".join(names)

# Tokenization
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])
total_chars = len(tokenizer.word_index) + 1

# Create input sequences and labels
input_sequences = []
for line in names:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_chars)

# Model Architecture
model = Sequential()
model.add(Embedding(total_chars, 50, input_length=max_sequence_length-1))
model.add(LSTM(100))
model.add(Dense(total_chars, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
model.fit(X, y, epochs=100, verbose=1)

# User input for searching names
search_input = input("Enter the starting characters for name generation: ")

# Text Generation
num_names = 5  # Number of names to generate
generated_names = []

for _ in range(num_names):
    seed_name = search_input  # User input as the seed for generating names
    next_chars = 3
    generated_name = seed_name

    for _ in range(next_chars):
        token_list = tokenizer.texts_to_sequences([seed_name])
        if not token_list:
            break

        token_list = token_list[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs)

        predicted_char = tokenizer.index_word[predicted_index]
        seed_name += predicted_char
        generated_name += predicted_char

    generated_names.append(generated_name)

# Output generated names
print("Generated Names:")
for name in generated_names:
    print(name)
