import random
import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from tqdm import tqdm


mnli_dataset = load_dataset("nyu-mll/multi_nli")

def preprocess_data(dataset, split_ratio=0.8):
    premises = [premise.lower() for premise in dataset['premise']]
    hypotheses = [hypothesis.lower() for hypothesis in dataset['hypothesis']]
    labels = dataset['label']

    data_size = len(labels)
    train_size = int(data_size * split_ratio)

    indices = list(range(data_size))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_data = {
        'premises': [premises[idx] for idx in train_indices],
        'hypotheses': [hypotheses[idx] for idx in train_indices],
        'labels': [labels[idx] for idx in train_indices]
    }

    test_data = {
        'premises': [premises[idx] for idx in test_indices],
        'hypotheses': [hypotheses[idx] for idx in test_indices],
        'labels': [labels[idx] for idx in test_indices]
    }

    return train_data, test_data

train_data, test_data = preprocess_data(mnli_dataset['train'])

train_premises = train_data['premises']
train_hypotheses = train_data['hypotheses']
train_labels = train_data['labels']

test_premises = test_data['premises']
test_hypotheses = test_data['hypotheses']
test_labels = test_data['labels']

val_data = mnli_dataset['validation_matched']
val_premises = [premise.lower() for premise in val_data['premise']]
val_hypotheses = [hypothesis.lower() for hypothesis in val_data['hypothesis']]
val_labels = val_data['label']

tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
premise_sequences_train = [tokenizer(premise) for premise in train_premises]
hypothesis_sequences_train = [tokenizer(hypothesis) for hypothesis in train_hypotheses]
premise_sequences_val = [tokenizer(premise) for premise in val_premises]
hypothesis_sequences_val = [tokenizer(hypothesis) for hypothesis in val_hypotheses]
premise_sequences_test = [tokenizer(premise) for premise in test_premises]
hypothesis_sequences_test = [tokenizer(hypothesis) for hypothesis in test_hypotheses]

vocab = torchtext.vocab.build_vocab_from_iterator(
    premise_sequences_train + hypothesis_sequences_train +
    premise_sequences_val + hypothesis_sequences_val +
    premise_sequences_test + hypothesis_sequences_test,
    specials=["<unk>"]
)

premise_sequences_train = [[vocab[token] for token in premise] for premise in premise_sequences_train]
hypothesis_sequences_train = [[vocab[token] for token in hypothesis] for hypothesis in hypothesis_sequences_train]
premise_sequences_val = [[vocab[token] for token in premise] for premise in premise_sequences_val]
hypothesis_sequences_val = [[vocab[token] for token in hypothesis] for hypothesis in hypothesis_sequences_val]
premise_sequences_test = [[vocab[token] for token in premise] for premise in premise_sequences_test]
hypothesis_sequences_test = [[vocab[token] for token in hypothesis] for hypothesis in hypothesis_sequences_test]

MAX_SEQ_LEN = 100
premise_sequences_train = [seq[:MAX_SEQ_LEN] + [0] * (MAX_SEQ_LEN - len(seq)) for seq in premise_sequences_train]
hypothesis_sequences_train = [seq[:MAX_SEQ_LEN] + [0] * (MAX_SEQ_LEN - len(seq)) for seq in hypothesis_sequences_train]
premise_sequences_val = [seq[:MAX_SEQ_LEN] + [0] * (MAX_SEQ_LEN - len(seq)) for seq in premise_sequences_val]
hypothesis_sequences_val = [seq[:MAX_SEQ_LEN] + [0] * (MAX_SEQ_LEN - len(seq)) for seq in hypothesis_sequences_val]
premise_sequences_test = [seq[:MAX_SEQ_LEN] + [0] * (MAX_SEQ_LEN - len(seq)) for seq in premise_sequences_test]
hypothesis_sequences_test = [seq[:MAX_SEQ_LEN] + [0] * (MAX_SEQ_LEN - len(seq)) for seq in hypothesis_sequences_test]

train_premises = torch.tensor(premise_sequences_train, dtype=torch.long)
train_hypotheses = torch.tensor(hypothesis_sequences_train, dtype=torch.long)
train_labels = torch.tensor(train_labels, dtype=torch.long)
val_premises = torch.tensor(premise_sequences_val, dtype=torch.long)
val_hypotheses = torch.tensor(hypothesis_sequences_val, dtype=torch.long)
val_labels = torch.tensor(val_labels, dtype=torch.long)
test_premises = torch.tensor(premise_sequences_test, dtype=torch.long)
test_hypotheses = torch.tensor(hypothesis_sequences_test, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super(BiGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 4, output_dim)  
        self.dropout = nn.Dropout(0.5)

    def forward(self, premise, hypothesis):
        embedded_premise = self.dropout(self.embedding(premise))
        embedded_hypothesis = self.dropout(self.embedding(hypothesis))
        _, premise_hidden = self.gru(embedded_premise)
        _, hypothesis_hidden = self.gru(embedded_hypothesis)
        hidden = torch.cat((premise_hidden[-2,:,:], premise_hidden[-1,:,:], hypothesis_hidden[-2,:,:], hypothesis_hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
model = BiGRU(vocab_size=len(vocab), embedding_dim=200, hidden_dim=256, output_dim=3, pad_idx=0).to(device)  # Move model to GPU if available

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def create_dataloader(premises, hypotheses, labels, batch_size, shuffle):
    filtered_indices = [idx for idx, label in enumerate(labels) if label != -1]
    filtered_premises = premises[filtered_indices]
    filtered_hypotheses = hypotheses[filtered_indices]
    filtered_labels = labels[filtered_indices]

    dataset = TensorDataset(filtered_premises, filtered_hypotheses, filtered_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

train_loader = create_dataloader(train_premises, train_hypotheses, train_labels, batch_size=64, shuffle=True)
val_loader = create_dataloader(val_premises, val_hypotheses, val_labels, batch_size=64, shuffle=False)
test_loader = create_dataloader(test_premises, test_hypotheses, test_labels, batch_size=64, shuffle=False)

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for batch in tqdm(iterator, desc="Training"):  
        optimizer.zero_grad()
        premise, hypothesis, labels = batch
        premise, hypothesis, labels = premise.to(device), hypothesis.to(device), labels.to(device)  # Move data to GPU if available
        predictions = model(premise, hypothesis)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        _, predicted = torch.max(predictions, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return epoch_loss / len(iterator), accuracy

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):  
            premise, hypothesis, labels = batch
            premise, hypothesis, labels = premise.to(device), hypothesis.to(device), labels.to(device)  # Move data to GPU if available
            predictions = model(premise, hypothesis)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            _, predicted = torch.max(predictions, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return epoch_loss / len(iterator), accuracy

N_EPOCHS = 20
best_val_loss = float('inf')
patience = 3
counter = 0

for epoch in range(N_EPOCHS):
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion)
    val_loss, val_accuracy = evaluate(model, val_loader, criterion)

    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Accuracy: {100 * train_accuracy:.2f}%, Val Loss: {val_loss:.3f}, Val Accuracy: {100 * val_accuracy:.2f}%')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Validation loss hasn't improved for {} epochs. Early stopping...".format(patience))
            break

test_loss, test_accuracy = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.3f}, Test Accuracy: {100 * test_accuracy:.2f}%')
