import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Load the Excel file
df = pd.read_excel("DATASET.xlsx")

# Define the mapping
mapping = {
    "A": "A", "B": "B", "C": "C", "D": "AB", "E": "AC", "F": "BA",
    "G": "CA", "H": "ABC", "I": "BCA", "J": "CAB", "K": "CBA",
    "L": "ABCA", "M": "ABCB", "N": "ABCC", "O": "BCAA", "P": "BCAB",
    "Q": "BCAC", "R": "CABA", "S": "CABB", "T": "CABC", "U": "BACA",
    "V": "BACB", "W": "BACC", "X": "CBAA", "Y": "CBAB", "Z": "CBAC",
    "1": "ABCAA", "2": "ABCAB", "3": "ABCAC", "4": "ABCBA", "5": "ABCBB",
    "6": "ABCBC", "7": "ABCCA", "8": "ABCCB", "9": "ABCCC", "0": "CCCCC",
    " ": "BBBBB"  # Space maps to "BBBBB"
}

# Inverse mapping for decoding
inverse_mapping = {v: k for k, v in mapping.items()}

def decode_sequences(sequences):
    decoded_passage = ""
    for sequence in sequences:
        if sequence in inverse_mapping:
            decoded_passage += inverse_mapping[sequence]
        else:
            print(f"Unrecognized sequence: {sequence}")
    return decoded_passage

# Prepare data for RNN
def prepare_sequences(sequence, char_to_index, seq_length=5):
    input_sequences = []
    target_sequences = []
    for i in range(len(sequence) - seq_length):
        input_seq = sequence[i:i + seq_length]
        target_seq = sequence[i + seq_length]
        input_sequences.append([char_to_index[c] for c in input_seq])
        target_sequences.append(char_to_index[target_seq])
    return np.array(input_sequences), np.array(target_sequences)

# Custom callback to compute F1 score
class F1ScoreCallback(keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.X_val), axis=-1)
        f1 = f1_score(self.y_val, y_pred, average='weighted')
        self.f1_scores.append(f1)
        print(f"Epoch {epoch + 1} - F1 Score: {f1:.4f}")

# Build RNN model
def build_rnn_model(vocab_size, embedding_dim=256, rnn_units=1024):
    model = keras.Sequential([
        layers.Input(shape=(None,)),  # Accept input as a sequence of indices
        layers.Embedding(vocab_size, embedding_dim),
        layers.SimpleRNN(rnn_units, return_sequences=False),
        layers.Dense(vocab_size, activation='softmax')
    ])
    return model

def main():
    # Use a single sequence
    final_passage = "THIS PROJECT DEMONSTRATES THE EFFECTIVENESS OF WAVELETBASED REALTIME IMAGE DENOISING AND ENHANCEMENT USING AN INTERACTIVE MATLAB GUI BY INTEGRATING HAAR WAVELET TRANSFORMS AND OTHER DENOISING TECHNIQUES THE APPLICATION PROVIDES USERS WITH PRECISE CONTROL OVER NOISE REDUCTION COLOR ADJUSTMENTS AND RESOLUTION SCALING THE PROJECT HIGHLIGHTS THE ADAPTABILITY OF WAVELET TRANSFORMS IN MANAGING VARIOUS NOISE TYPES WITHOUT SACRIFICING CRITICAL IMAGE FEATURES MAKING IT SUITABLE FOR DIVERSE APPLICATIONS IN FIELDS LIKE MEDICAL IMAGING SATELLITE SENSING AND DIGITAL MEDIA THIS TOOL EXEMPLIFIES HOW ADVANCED SIGNAL PROCESSING TECHNIQUES CAN BE TRANSLATED INTO USERFRIENDLY APPLICATIONS PROVIDING A POWERFUL RESOURCE FOR ENHANCING IMAGE QUALITY IN REALWORLD SCENARIOS"

    # Create character mapping for RNN training
    chars = sorted(set(final_passage))
    char_to_index = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)

    # Prepare training data
    seq_length = 5  # You can adjust this as needed
    input_sequences, target_sequences = prepare_sequences(final_passage, char_to_index, seq_length)

    # Reshape input data for the RNN: (batch_size, time_steps)
    input_sequences = np.array(input_sequences)

    # Check if there are input sequences
    if len(input_sequences) == 0:
        print("No input sequences were generated. Consider lowering the sequence length.")
        return

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(input_sequences, target_sequences, test_size=0.2, random_state=42)

    # Build and compile the RNN model
    model = build_rnn_model(vocab_size)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Create F1 score callback
    f1_callback = F1ScoreCallback(X_test, y_test)

    # Train the model and capture the training history
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[f1_callback])

    # Predict and evaluate on the test set
    y_pred = np.argmax(model.predict(X_test), axis=-1)

    # Compute evaluation metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print("\nEvaluation Metrics:")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Plot training & validation accuracy, loss, F1 score, and overall evaluation metrics
    plt.figure(figsize=(14, 10))

    # Accuracy Plot
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    # Loss Plot
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    # F1 Score Plot
    plt.subplot(2, 2, 3)
    plt.plot(f1_callback.f1_scores, label='F1 Score', color='orange')
    plt.title('F1 Score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend(loc='upper left')

    # Overall Evaluation Metrics Bar Chart
    plt.subplot(2, 2, 4)
    metrics = [f1, accuracy, precision, recall]
    metrics_names = ['F1 Score', 'Accuracy', 'Precision', 'Recall']
    plt.bar(metrics_names, metrics, color=['orange', 'blue', 'green', 'red'])
    plt.title('Overall Evaluation Metrics')
    plt.ylim(0, 1)  # Set limit from 0 to 1 for percentage display
    plt.ylabel('Score')
    plt.grid(axis='y')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
