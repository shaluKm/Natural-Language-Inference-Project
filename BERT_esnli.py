import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

esnli_dataset = load_dataset("esnli")

def preprocess_data(dataset):
    premises = [premise.lower() for premise in dataset['premise']]
    hypotheses = [hypothesis.lower() for hypothesis in dataset['hypothesis']]
    labels = dataset['label']
    return premises, hypotheses, labels

train_premises, train_hypotheses, train_labels = preprocess_data(esnli_dataset['train'])
val_premises, val_hypotheses, val_labels = preprocess_data(esnli_dataset['validation'])
test_premises, test_hypotheses, test_labels = preprocess_data(esnli_dataset['test'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_premises, train_hypotheses, truncation=True, padding=True)
val_encodings = tokenizer(val_premises, val_hypotheses, truncation=True, padding=True)
test_encodings = tokenizer(test_premises, test_hypotheses, truncation=True, padding=True)

train_inputs = torch.tensor(train_encodings['input_ids'])
train_labels = torch.tensor(train_labels)
val_inputs = torch.tensor(val_encodings['input_ids'])
val_labels = torch.tensor(val_labels)
test_inputs = torch.tensor(test_encodings['input_ids'])
test_labels = torch.tensor(test_labels)

train_dataset = TensorDataset(train_inputs, train_labels)
val_dataset = TensorDataset(val_inputs, val_labels)
test_dataset = TensorDataset(test_inputs, test_labels)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 10)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


criterion = nn.CrossEntropyLoss()


def train(model, iterator, optimizer, scheduler, criterion):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for inputs, labels in tqdm(iterator, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
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
        for inputs, labels in tqdm(iterator, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[0]
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return epoch_loss / len(iterator), accuracy

best_val_loss = float('inf')
patience = 3
counter = 0

model.train()

num_epochs = 3

for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, optimizer, scheduler, criterion)
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
