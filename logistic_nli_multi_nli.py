import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datasets import load_dataset
import random
import seaborn as sns

mnli_dataset = load_dataset("multi_nli")

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

train_data_concat = [premise + ' ' + hypothesis for premise, hypothesis in zip(train_premises, train_hypotheses)]
test_data_concat = [premise + ' ' + hypothesis for premise, hypothesis in zip(test_premises, test_hypotheses)]

vectorizer = CountVectorizer(lowercase=True, tokenizer=None, stop_words='english', max_features=5000)
train_features = vectorizer.fit_transform(train_data_concat)
test_features = vectorizer.transform(test_data_concat)

train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

model = LogisticRegression(max_iter=1000)

model.fit(train_features, train_labels)

train_predictions = model.predict(train_features)
train_accuracy = accuracy_score(train_labels, train_predictions)
print(f'Train Accuracy: {train_accuracy:.2f}')

test_predictions = model.predict(test_features)
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f'Test Accuracy: {test_accuracy:.2f}')

print("Classification Report:")
print(classification_report(test_labels, test_predictions))

cm = confusion_matrix(test_labels, test_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
