# Neural Translation Model

## Overview
This project implements a neural translation model that translates English sentences into German using a sequence-to-sequence model with RNNs. It utilizes a pre-trained word embedding model for English, an encoder-decoder architecture with LSTMs, and a custom training loop. The project is structured as a capstone project, with step-by-step implementation instructions.

## Features
- Preprocessing pipeline for text data, including tokenization and padding.
- Encoder-decoder architecture with LSTM layers.
- Pre-trained word embeddings for English input processing.
- Custom training loop with loss monitoring.
- Translation inference loop for generating German sentences from English inputs.

## Dataset
The dataset used for training is sourced from [ManyThings.org](http://www.manythings.org/anki/), consisting of over 200,000 sentence pairs. For faster training, we limit it to 20,000 pairs.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- TensorFlow 2.x
- TensorFlow Hub
- NumPy
- Matplotlib
- Scikit-learn

### Install Dependencies
```bash
pip install tensorflow tensorflow-hub numpy matplotlib scikit-learn
```

## Usage
### 1. Preprocess Text Data
- Load English-German sentence pairs.
- Normalize and tokenize the text.
- Add `<start>` and `<end>` tokens to German sentences.
- Convert text sequences into padded integer sequences.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize German sentences
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(german_sentences)
german_sequences = tokenizer.texts_to_sequences(german_sentences)
german_sequences_padded = pad_sequences(german_sequences, padding='post')
```

### 2. Prepare Dataset with TensorFlow
- Load English word embeddings from TensorFlow Hub.
- Create `tf.data.Dataset` objects for training and validation.
- Apply tokenization, embedding, and batching.

```python
import tensorflow_hub as hub
embedding_layer = hub.KerasLayer("./models/tf2-preview_nnlm-en-dim128_1", output_shape=[128], input_shape=[], dtype=tf.string)
```

### 3. Build Model
#### Encoder Network
- Embeds English words into a 128-dimensional space.
- Uses an LSTM layer to encode sentence representations.

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Masking

inputs = Input(shape=(None, 128))
x = Masking(mask_value=0.0)(inputs)
_, hidden_state, cell_state = LSTM(512, return_state=True)(x)
encoder = Model(inputs=inputs, outputs=[hidden_state, cell_state])
```

#### Decoder Network
- Uses an embedding layer for German words.
- An LSTM layer generates translated words iteratively.

```python
from tensorflow.keras.layers import Embedding, Dense

class Decoder(Model):
    def __init__(self, vocab_size, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True)
        self.lstm = LSTM(512, return_sequences=True, return_state=True)
        self.dense = Dense(vocab_size)

    def call(self, inputs, hidden_state=None, cell_state=None):
        x = self.embedding(inputs)
        x, hidden_state, cell_state = self.lstm(x, initial_state=[hidden_state, cell_state])
        x = self.dense(x)
        return x, hidden_state, cell_state
```

### 4. Train the Model
- Uses a custom training loop with backpropagation.
- Tracks training and validation loss.

```python
@tf.function
def train_step(encoder, decoder, english_input, german_input, german_output, loss_function, optimizer):
    with tf.GradientTape() as tape:
        encoder_hidden, encoder_cell = encoder(english_input)
        decoder_output, _, _ = decoder(german_input, hidden_state=encoder_hidden, cell_state=encoder_cell)
        loss = loss_function(german_output, decoder_output)

    trainable_vars = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    return loss
```

### 5. Translate Sentences
- Given an English sentence, preprocess and embed it.
- Use the trained model to generate a German translation.

```python
def translate_sentence(sentence):
    sentence = preprocess_sentence(sentence)
    embedded_sentence = embedding_layer(tf.constant([sentence]))
    encoder_hidden, encoder_cell = encoder(embedded_sentence)
    decoder_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []
    
    for _ in range(100):
        predictions, decoder_hidden, decoder_cell = decoder(decoder_input, hidden_state=encoder_hidden, cell_state=encoder_cell)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])
        if tokenizer.index_word[predicted_id] == '<end>':
            return ' '.join(result[:-1])
        decoder_input = tf.expand_dims([predicted_id], 0)
    return ' '.join(result)
```

## Example Output
```bash
English: "Where is the train station?"
German Translation: "Wo ist der Bahnhof?"
```

## License
This project is licensed under the MIT License.

