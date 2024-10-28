# Natural Language Inference (NLI) Project

This project explores methods for natural language inference (NLI), a key task in natural language processing (NLP) that focuses on determining the logical relationship between two text fragments, specifically if one text (hypothesis) can be inferred from another (premise). It covers multiple models, datasets, and analytical techniques for NLI, contributing to applications in question answering, sentiment analysis, and information retrieval.


## Objective

Natural language inference determines if a hypothesis is entailed, contradicted, or neutral in relation to a given premise. This task is foundational for various NLP applications such as:
- **Textual entailment recognition**
- **Automated reasoning**
- **Improved information retrieval**

The goal is to build models capable of accurately classifying pairs into one of the three classes: **Entailment**, **Contradiction**, and **Neutral**.

## Datasets

### SNLI
- **Full Name**: Stanford Natural Language Inference
- **Characteristics**: Contains 550,152 train, 10,000 validation, and 10,000 test samples, derived from Flickr30 image captions. 
- **Purpose**: Benchmark for simpler, genre-specific NLI tasks.

### MultiNLI
- **Full Name**: Multi-Genre Natural Language Inference
- **Characteristics**: 392,702 train samples, 20,000 each for dev and test sets; comprises diverse genres.
- **Purpose**: Provides a more challenging benchmark due to its multi-genre nature, enhancing model robustness.

### e-SNLI
- **Full Name**: Extended Stanford Natural Language Inference
- **Characteristics**: 569,033 sentence pairs; annotations include explanations for entailment, contradiction, or neutrality.
- **Purpose**: Augments SNLI for better model explanation capability.

## Models and Techniques

### Logistic Regression
- **Approach**: Basic classifier using a Bag-of-Words model for feature extraction.
- **Performance**:
  - e-SNLI: ~52.31% accuracy
  - MultiNLI: ~36% accuracy
  - SNLI: ~51% accuracy

### Bidirectional LSTM
- **Approach**: Utilizes bidirectional LSTM for capturing long-term dependencies in both directions.
- **Performance**:
  - e-SNLI: ~68.24% accuracy
  - MultiNLI: ~55.55% accuracy
  - SNLI: ~68.29% accuracy

### Bidirectional GRU
- **Approach**: Reduces computational load compared to LSTM while preserving sequential dependencies.
- **Performance**:
  - e-SNLI: ~67.71% accuracy
  - MultiNLI: ~55.06% accuracy
  - SNLI: ~67.35% accuracy

### BERT
- **Approach**: Transformer-based model leveraging bidirectional attention for enhanced semantic understanding.
- **Performance**:
  - Achieved 88.54% accuracy on e-SNLI, outperforming other models.

## Results

| Model                | e-SNLI Accuracy | MultiNLI Accuracy | SNLI Accuracy |
|----------------------|-----------------|-------------------|---------------|
| Logistic Regression  | 52.31%          | 36.00%           | 51.00%        |
| Bidirectional LSTM   | 68.24%          | 55.55%           | 68.29%        |
| Bidirectional GRU    | 67.71%          | 55.06%           | 67.35%        |
| BERT                 | **88.54%**      | -                | -             |

## Applications

NLI models can support multiple NLP tasks:
- **Question Answering**: Ensures answers align logically with posed questions.
- **Textual Entailment**: Validates semantic similarity in paraphrase detection.
- **Information Retrieval**: Increases retrieval precision by recognizing entailment relationships.
- **Machine Translation and Summarization**: Validates coherence and consistency in generated content.

## Conclusion

This project demonstrates that deep learning models (Bi-LSTM, Bi-GRU) outperform simpler models like logistic regression in NLI tasks. BERT achieves the highest accuracy, showcasing the advantages of transformer-based architectures for understanding complex language patterns.

