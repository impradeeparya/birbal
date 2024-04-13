import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src import StateInfo


class StateTrainer:
    def __init__(self):
        self.state_info = StateInfo()

        # Example dataset (query, state)
        data = []
        for state, districts in self.state_info.data.items():
            state_queries = [
                f"What districts are there in {state}?",
                f"List districts in {state}",
                f"Tell me about districts of {state}",
                f"Show districts of {state}",
                f"How many districts in {state}?",
            ]
            data.extend([(query, state, districts) for query in state_queries])
        # Separate features (queries) and labels (states)
        queries, states, districts = zip(*data)

        # Initialize TensorFlow tokenizer
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(queries)

        # Convert text data to sequences
        sequences = self.tokenizer.texts_to_sequences(queries)

        # Pad sequences to ensure uniform length
        self.max_sequence_length = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)

        # Convert labels to one-hot encoding
        self.label_map = {state: i for i, state in enumerate(set(states))}
        num_classes = len(self.label_map)
        labels = [self.label_map[state] for state in states]
        one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

        # Split data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(padded_sequences, one_hot_labels, test_size=0.2,
                                                            random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Define the model architecture
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=64,
                                      input_length=self.max_sequence_length),
            tf.keras.layers.Conv1D(128, 5, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        # Compile the model
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        self.model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))

        # Evaluate the model on test set
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    def fetch_token(self, query):
        # Example query prediction
        sequence = self.tokenizer.texts_to_sequences([query])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length)
        predicted_probs = self.model.predict(padded_sequence)[0]
        predicted_state_index = np.argmax(predicted_probs)
        predicted_state = [state for state, index in self.label_map.items() if index == predicted_state_index][0]
        print(f"Predicted State: {predicted_state}")
        return predicted_state, self.state_info.data[predicted_state]
