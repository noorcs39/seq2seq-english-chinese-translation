# English-to-Chinese Translator Using LSTM-Based Seq2Seq Architecture

This project implements a basic English-to-Chinese translator using a Sequence-to-Sequence (Seq2Seq) model with Long Short-Term Memory (LSTM) networks in TensorFlow and Keras.

## 📌 Overview

The model takes English sentences as input and learns to generate corresponding Chinese translations using a neural encoder-decoder structure. It demonstrates core concepts of neural machine translation (NMT), including sequence encoding, decoding, and evaluation using BLEU score.

---

## 🧠 Features

- Encoder-Decoder architecture with LSTM
- Tokenization and padding for bilingual text
- One-hot encoded decoder targets
- Training with `categorical_crossentropy` loss
- Inference mode for translation
- BLEU score evaluation

---

## 🛠️ Requirements

Install the dependencies using pip:

```bash
pip install tensorflow nltk numpy
