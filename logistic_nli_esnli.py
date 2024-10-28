import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

esnli_dataset = load_dataset("esnli")

def preprocess_data(dataset):
    premises = [premise.lower() for premise in dataset['premise']]
    hypotheses = [hypothesis.lower() for hypothesis in dataset['hypothesis']]
    labels = dataset['label']
    return premises, hypotheses, labels

train_premises, train_hypotheses, train_labels = preprocess_data(esnli_dataset['train'])
val_premises, val_hypotheses, val_labels = preprocess_data(esnli_dataset['validation'])
test_premises, test_hypotheses, test_labels = preprocess_data(esnli_dataset['test'])

train_data = [premise + ' ' + hypothesis for premise, hypothesis in zip(train_premises, train_hypotheses)]
val_data = [premise + ' ' + hypothesis for premise, hypothesis in zip(val_premises, val_hypotheses)]
test_data = [premise + ' ' + hypothesis for premise, hypothesis in zip(test_premises, test_hypotheses)]

vectorizer = CountVectorizer(lowercase=True, tokenizer=None, stop_words='english', max_features=5000)
train_features = vectorizer.fit_transform(train_data)
val_features = vectorizer.transform(val_data)
test_features = vectorizer.transform(test_data)

train_labels = torch.tensor(train_labels, dtype=torch.long)
val_labels = torch.tensor(val_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

model = LogisticRegression(max_iter=1000)

model.fit(train_features, train_labels)

train_predictions = model.predict(train_features)
train_accuracy = accuracy_score(train_labels, train_predictions)
print(f'Train Accuracy: {train_accuracy:.2f}')

val_predictions = model.predict(val_features)
val_accuracy = accuracy_score(val_labels, val_predictions)
print(f'Validation Accuracy: {val_accuracy:.2f}')

test_predictions = model.predict(test_features)
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f'Test Accuracy: {test_accuracy:.2f}')

print("Classification Report:")
print(classification_report(val_labels, val_predictions))

cm = confusion_matrix(val_labels, val_predictions)

target_names = ['entailment', 'neutral', 'contradiction']

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()