# Aegis: AI Code Detection Model

## Overview
Aegis is a fine-tuned CodeBERT model that classifies LLM-generated and human code. CodeBERT has 125 million parameters, but using LoRA (Low-Rank Adaptation), Aegis was efficiently trained locally with only a subset of the original parameters being updated. The dataset used to train and evaluate Aegis was APPS, a Python benchmark for code generation.

## Features
Aegis's key features reside in human vs LLM-generated Python code classification. As an encoder, the model aggregates the features of the sample to create an understanding of the input through 12 transformer layers after tokenization. Then, the CLS token is run through a linear classifier and used to determine whether the code is LLM-generated or human. Aegis was efficiently fine-tuned using LoRA (Low-Rank Adaptation) to both reduce the trainable parameters and save computational resources and time. It's capable of binary classification of Python code snippets.

**Key Capabilities:**
- Binary classification of Python code snippets
- Efficient fine-tuning using LoRA (Low-Rank Adaptation)
- Local training with reduced parameter updates
- High accuracy on APPS benchmark dataset

### Requirements
- Python 3.13+
- PyTorch
- Transformers
- PEFT (Parameter Efficient Fine-Tuning)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/aegis.git
cd aegis

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
cd model
python train.py
```

### Evaluation
```bash
cd model
python eval.py
```

### Data Preprocessing
```bash
cd data
python preprocess.py
```

## Model Performance
- **Accuracy**: 80.75%
- **F1-Score**: 82.93%

## Dataset
The model was trained on the APPS (Automated Programming Problem Solving) dataset, which contains:
- 2,000 human-written Python code samples
- 2,000 LLM-generated Python code samples
- Balanced train/validation/test splits (80%/10%/10%)

## Technical Details

### Architecture
- **Base Model**: Microsoft CodeBERT (125M parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Configuration**:
  - Rank (r): 4
  - Alpha: 16
  - Dropout: 0.1
  - Target modules: query, value, key

### Training Configuration
- **Epochs**: 3
- **Batch Size**: 4 (effective batch size: 32 with gradient accumulation)
- **Learning Rate**: 1e-4
- **Weight Decay**: 0.01
- **Warmup Ratio**: 0.1

## Contributing
To contribute, please submit a PR. 
