## Fine-Tuned Helsinki-NLP Opus-MT German-English Model

This repository contains the fine-tuned version of the [Helsinki-NLP/opus-mt-de-en](https://huggingface.co/Helsinki-NLP/opus-mt-de-en) model, fine-tuned on the WMT14 German-English translation dataset. The model is intended for high-quality German-to-English translation tasks and has been trained using the Hugging Face `transformers` library.

## Table of Contents
- [Introduction](#introduction)
- [Model Overview](#model-overview)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Training Details](#training-details)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This model is based on the `opus-mt-de-en` architecture and fine-tuned on the WMT14 German-English dataset. It is designed for machine translation tasks and can be used for real-time German to English translation.

## Model Overview

- **Model Name**: Helsinki-NLP/opus-mt-de-en (Fine-Tuned)
- **Source Language**: German (de)
- **Target Language**: English (en)
- **Training Dataset**: WMT14 German-English dataset (subset)
- **Fine-Tuning Framework**: Hugging Face `transformers` and `datasets`
- **Training Duration**: 3 epochs
- **Fine-Tuning Hyperparameters**:
  - Learning Rate: 2e-5
  - Batch Size: 8
  - Epochs: 3

## Getting Started

### Installation

To use the model, clone this repository and install the required dependencies.

```
git clone https://github.com/your-username/fine-tuned-opus-mt-de-en.git
cd fine-tuned-opus-mt-de-en
pip install -r requirements.txt
```
Here are the necessary Python libraries:
* transformers
* datasets
* torch

You can view them via pip:
```
pip install transformers datasets torch
```

### Usage
Once the model is set up, you can use it to translate German sentences into English. Here’s an example of how to load and use the fine-tuned model for inference:
```
from transformers import MarianMTModel, MarianTokenizer

# Load the fine-tuned model and tokenizer from Hugging Face
model = MarianMTModel.from_pretrained("iSathyam03/fine-tuned-opus-mt-de-en")
tokenizer = MarianTokenizer.from_pretrained("iSathyam03/fine-tuned-opus-mt-de-en")

# Translate a German sentence
def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Example translation
german_sentence = "Wie geht es dir?"
english_translation = translate(german_sentence)
print(f"German: {german_sentence}")
print(f"English: {english_translation}")
```
This will output the translated English sentence.

## Training Details
* Dataset: The model was fine-tuned using the `wmt14` German-English dataset from Hugging Face’s datasets library.
* Preprocessing: The `dataset` was tokenized using the `MarianTokenizer`, and sequences were truncated/padded to a maximum length of 128 tokens.
* Fine-Tuning: We used Hugging Face's `Trainer` class to fine-tune the model for 3 epochs (As this was a practice for me).

## Results
The model is now available on Hugging Face for usage in any translation application. The model has been trained for 1 epochs on a subset of the WMT14 dataset. The model can handle typical German-to-English translation tasks but may need further fine-tuning for more specialized use cases.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
* [Helsinki-NLP/opus-mt-de-en](https://huggingface.co/Helsinki-NLP/opus-mt-de-en) for the base model.
* [Hugging Face](https://huggingface.co/) for providing the tools and libraries.
* [WMT14](https://huggingface.co/datasets/wmt/wmt14) Dataset for the training data.















