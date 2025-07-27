import pandas as pd
import numpy as np
import pickle
import ast
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from CNN import CNNTextClassifier


# Create a simple dataset class for loading data in batches
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
    

# Tokenize the text
def tokenize(text):
    return text.lower().split() 

# Convert text to sequences of integers
def text_to_sequence(text):
    return torch.tensor([vocab[token] for token in tokenize(text)])

# Pad sequences to the same length
def pad_or_truncate(sequences, seq_len=16):
    padded_sequences = []
    for seq in sequences:
        # Truncate or pad the sequences to seq_len
        if len(seq) > seq_len:
            padded_sequences.append(seq[:seq_len])  # Truncate to seq_len
        else:
            padded_sequences.append(torch.cat([seq, torch.zeros(seq_len - len(seq))]))  # Pad to seq_len
    return torch.stack(padded_sequences)

if __name__ == '__main__':
    # prepare the data
    train_data = pd.read_csv('src/data/train_data.csv')
    test_data  = pd.read_csv('src/data/valid_data.csv')
    train_data['tokens'] = train_data['tokens'].apply(ast.literal_eval)
    test_data['tokens']  = test_data['tokens'].apply(ast.literal_eval)
    train_data = train_data.dropna()
    test_data = test_data.dropna()
    train_texts = train_data['cleaned_text']
    train_labels = train_data['label']
    test_texts = test_data['cleaned_text']
    test_labels = test_data['label']
    train_tokens = train_data['tokens']
    test_tokens = test_data['tokens']

    # Build vocabulary from training data
    vocab = build_vocab_from_iterator([tokenize(text) for text in train_texts], specials=["<pad>"])
    vocab.set_default_index(vocab["<pad>"])  # Handle unknown words
    torch.save(vocab, 'src/vocab/vocab.pth')

    # Convert all texts
    train_sequences = [text_to_sequence(text) for text in train_texts]
    test_sequences  = [text_to_sequence(text) for text in test_texts]

    # Pad sequences to the same length
    sentence_len = train_tokens.apply(len)
    ninety_perc = np.percentile(sentence_len, 90)
    seq_len = int(ninety_perc)
    train_padded_sequences = pad_or_truncate(train_sequences, seq_len=seq_len)
    test_padded_sequences  = pad_or_truncate(test_sequences, seq_len=seq_len)

    # Convert labels to tensors
    train_labels_tensor = torch.tensor(train_labels.values)
    test_labels_tensor  = torch.tensor(test_labels.values)

    # Create dataset objects
    train_dataset = SentimentDataset(train_padded_sequences, train_labels_tensor)
    test_dataset  = SentimentDataset(test_padded_sequences, test_labels_tensor)

    # DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Check if CUDA is available and use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    embed_size = 100  # Size of word embeddings
    num_classes = 3   # Number of sentiment classes: 0 (negative), 1 (neutral), 2 (positive)
    kernel_size = 2   # Size of the convolution filter
    num_filters = 50 # Number of convolution filters
    dropout = 0.7     # Dropout rate

    sentence_len = train_data['tokens'].apply(len)
    ninety_perc = np.percentile(sentence_len, 90)
    seq_len = int(ninety_perc)
    padding_idx = vocab["<pad>"]
    model = CNNTextClassifier(vocab_size=len(vocab), embed_size=embed_size, num_classes=num_classes, seq_len=seq_len, padding_idx=padding_idx, kernel_size=kernel_size, num_filters=num_filters, dropout=dropout)
    model = model.to(device)

    if os.path.exists("src/models/cnn_weights.pth"):
        model.load_state_dict(torch.load("src/models/cnn_weights.pth"))
        print("Loading weights from {src/models/cnn_weights.pth}")

    print(f"Device: {device}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    if os.path.exists("src/cnn_training_results.pkl"):
        with open("src/cnn_training_results.pkl", 'rb') as f:
            train_losses, eval_losses, train_corrects, eval_corrects, best_accuracy = pickle.load(f)
    else:
        train_losses = []
        eval_losses = []
        train_corrects = []
        eval_corrects = []
        best_accuracy = 0.0  # Track the best accuracy

    parser = argparse.ArgumentParser(description='Train a model.')
    # Add the `epochs` argument
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    args = parser.parse_args()

    # Training loop
    epochs = args.epochs
    test_accuracies, epoch_errors = [], []
    best_model_state = model.state_dict()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        errors = 0
        for texts, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'):
            texts, labels = texts.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()
            # Forward pass
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            errors += (predicted != labels).sum().item()

        # Record training loss and accuracy
        train_losses.append(running_loss / len(train_loader))
        accuracy = 100 * correct / total
        train_corrects.append(correct)
        epoch_errors.append(errors / total)

        # Evaluate on test data
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.to(device)  # Move data to GPU
                outputs = model(texts)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                predicted = torch.max(outputs, 1)[1]
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        # Record test accuracy
        eval_loss = running_loss / len(test_loader)
        eval_losses.append(eval_loss)
        eval_corrects.append(correct)
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = model.state_dict()  # Save model state

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_losses[-1]:.4f}, Train Accuracy: {accuracy:.2f}%, Test Accuracy: {test_accuracies[-1]:.2f}%, Errors: {epoch_errors[-1]:.2f}")


    if best_model_state:
        torch.save(best_model_state, 'src/models/cnn_weights.pth')
        print("Best model saved with accuracy:", best_accuracy)

    with open('src/cnn_training_results.pkl', 'wb') as f:
        pickle.dump([train_losses, eval_losses, train_corrects, eval_corrects, best_accuracy], f)