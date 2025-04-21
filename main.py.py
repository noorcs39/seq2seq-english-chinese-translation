import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu

# Sample data for demonstration purposes
english_sentences = ['Hello, how are you?', 'Go to there.', 'I love you.']
chinese_sentences = ['你好，你怎么样？', '去那里。', '我爱你。']

# Define maximum sequence lengths based on your data
max_encoder_seq_length = 10  # Maximum length of encoder sequences
max_decoder_seq_length = 10  # Maximum length of decoder sequences
units = 256  # LSTM units
batch_size = 64
epochs = 100

# Tokenization and padding for the input (English) sentences
input_tokenizer = Tokenizer(filters='')
input_tokenizer.fit_on_texts(english_sentences)
input_sequences = input_tokenizer.texts_to_sequences(english_sentences)
encoder_input_data = pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding='post')

# Define num_encoder_tokens based on the tokenizer's word index
num_encoder_tokens = len(input_tokenizer.word_index) + 1  # Adding 1 for padding

# Tokenization and padding for the target (Chinese) sentences
target_tokenizer = Tokenizer(filters='')
target_tokenizer.fit_on_texts(chinese_sentences)
target_sequences = target_tokenizer.texts_to_sequences(chinese_sentences)
decoder_input_data = pad_sequences(target_sequences, maxlen=max_decoder_seq_length, padding='post')

# Define num_decoder_tokens based on the tokenizer's word index
num_decoder_tokens = len(target_tokenizer.word_index) + 1  # Adding 1 for padding

# Create the decoder target data by shifting the decoder_input_data
decoder_target_data = np.zeros((len(chinese_sentences), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, seq in enumerate(target_sequences):
    for t, token in enumerate(seq):
        if t > 0:
            decoder_target_data[i, t - 1, token] = 1.0

# Define the encoder model
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(num_encoder_tokens, units)(encoder_inputs)
encoder_lstm = LSTM(units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Define the decoder model
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(num_decoder_tokens, units)(decoder_inputs)
decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the Seq2Seq model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Train the model
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

# Define inference models for translation
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(units,))
decoder_state_input_c = Input(shape=(units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to text
reverse_target_char_index = {i: char for char, i in target_tokenizer.word_index.items()}
reverse_target_char_index[0] = ''  # Add an empty string for padding

# Translate an English sentence to Chinese
def translate(english_sentence):
    # Preprocessing the input
    input_seq = input_tokenizer.texts_to_sequences([english_sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_encoder_seq_length, padding='post')
    # Encode the input
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1 with only the start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index.get('\t', 1)  # Default to 1 if '\t' not in tokenizer
    # Sampling loop for a batch of sequences
    decoded_sentence = ''
    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index.get(sampled_token_index, '')
        decoded_sentence += sampled_char
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            break
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return decoded_sentence

# Translate a sample sentence and calculate BLEU score
sample_english_sentence = 'Hello, how are you?'
translated_chinese_sentence = translate(sample_english_sentence)
print('Translated sentence:', translated_chinese_sentence)

# BLEU score calculation
reference_translations = [['你好，你怎么样？'.split()]]
translated_sentence_tokens = translated_chinese_sentence.split()
bleu_score = corpus_bleu(reference_translations, [translated_sentence_tokens])
print('BLEU score:', bleu_score)
