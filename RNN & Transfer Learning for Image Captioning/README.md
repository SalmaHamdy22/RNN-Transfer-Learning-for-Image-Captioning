
# Image Captioning System Using RNNs and Transfer Learning
    This project is an implementation of an image captioning system that generates natural language descriptions for images.
    It leverages a pre-trained Convolutional Neural Network (CNN) for feature extraction and a Recurrent Neural Network (RNN) for caption generation.
    The system is trained on the Flickr8k dataset, which contains images paired with descriptive captions.



# Dataset description

    The Flickr8k dataset consists of 8,000 images, each paired with 5 captions describing the content of the image.
    The captions are written in natural language and cover a wide range of scenarios, from people performing activities to animals and landscapes.

    Source: Kaggle - Flickr8k Dataset.
    Structure:
        Images: Flickr8k_Dataset/ directory.
        Captions: Flickr8k_text/Flickr8k.token.txt file,containing image-caption pairs.



# Dataset Preprocessing

    - Image Preprocessing:

        Images are resized and normalized to be compatible with the CNN model used for feature extraction (e.g., InceptionV3).
        Features are extracted using the CNN and saved as numpy arrays for efficient processing.

    - Caption Preprocessing:

        Captions are cleaned to remove special characters, numbers, and extra spaces.
        Tokenization is performed to convert captions into sequences of integers based on a vocabulary built from the dataset.
        Start (<start>) and end (<end>) tokens are added to each caption to indicate the start and end of sentences.
# RNN Model

The RNN-based model consists of:

    - Feature Extractor Input: A dense layer processes image feature vectors extracted by the CNN.
    - Embedding Layer: Converts tokenized words into dense vector representations.
    - LSTM: Processes the sequence of words in captions and learns the relationship between words and the image features.
    - Output Layer: A dense layer with softmax activation predicts the next word in the caption sequence.
    - Loss Function: sparse_categorical_crossentropy.
        Optimizer: Adam.
# Steps to Run the Code in Jupyter Notebook

## Run import cell
    import os
    import numpy as np
    import pandas as pd
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.decomposition import PCA
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    from tqdm import tqdm
    import cv2
    import re
    from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Add
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from collections import Counter
    import string 
    from nltk.translate.bleu_score import sentence_bleu

## Run extract feature cell
    features = extract_features(IMAGE_DIR, use_pca=True, n_components=128)

## Run loading caption cell
    descriptions = load_and_preprocess_captions(CAPTION_FILE)

## Run tokenize cell
    tokenizer, vocab_size = tokenize_captions(descriptions)

## Run preparing data cell
    (X1_train, X2_train, y_train), (X1_val, X2_val, y_val) = prepare_data(features, descriptions, tokenizer, max_length)

## Run train model (LSTM) cell 
    history = train_model(model, X1_train, X2_train, y_train, X1_val, X2_val, y_val, epochs=15, batch_size=64)

## Run save model cell
    model.save('D:/image_captioning_model.keras') 

## Run evaluate model cell using BLeu metric
    bleu_score = sentence_bleu(tokenized_reference, tokenized_generated, smoothing_function=SmoothingFunction().method1)

    print(f"BLEU score: {bleu_score}")

# BLEU Score
    The BLEU (Bilingual Evaluation Understudy) score is used to evaluate the quality of the generated captions by comparing them with the reference captions in the dataset.

    - Metric: A value between 0 and 1, where higher values indicate better alignment between the generated and reference captions.
    - Evaluation:
        BLEU-1 (unigram precision).
        BLEU-2, BLEU-3, and BLEU-4 (n-gram precision for 2, 3, and 4-grams).