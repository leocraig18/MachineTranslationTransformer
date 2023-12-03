import random
from classes import PositionalEmbedding, TransformerEncoder, TransformerDecoder
import tensorflow as tf
import string
import re
from tensorflow import keras
from keras import layers, callbacks

def main():
    def load_data():
        text_file = "spa-eng/spa.txt"
        text_pairs = []
        with open(text_file, "r") as f:
            lines = f.read().split("\n")[:-1]
        for line in lines:
            eng, spanish = line.split("\t")
            spanish = "[start] " + spanish + " [end]"
            text_pairs.append((eng, spanish))
        return text_pairs

    def split_data(text_pairs):
        random.shuffle(text_pairs)
        num_val_samples = int(0.15 * len(text_pairs))
        num_train_samples = len(text_pairs) - 2 * num_val_samples
        train_pairs = text_pairs[:num_train_samples]
        val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
        test_pairs = text_pairs[num_train_samples + num_val_samples:]
        return train_pairs, val_pairs, test_pairs

    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")


    def custom_standardization(input_string):
        lowercase = tf.strings.lower(input_string)
        return tf.strings.regex_replace(
            lowercase, f"[{re.escape(strip_chars)}]", "")


    vocab_size = 15000
    sequence_length = 20

    source_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    target_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length + 1,
        standardize=custom_standardization,
    )

    train_pairs, val_pairs, test_pairs = split_data(load_data())
    train_english_texts = [pair[0] for pair in train_pairs]
    train_spanish_texts = [pair[1] for pair in train_pairs]
    source_vectorization.adapt(train_english_texts)
    target_vectorization.adapt(train_spanish_texts)

    batch_size = 64


    def format_dataset(eng, spa):
        eng = source_vectorization(eng)
        spa = target_vectorization(spa)
        return ({
            "english": eng,
            "spanish": spa[:, :-1],
        }, spa[:, 1:])


    def make_dataset(pairs):
        eng_texts, spa_texts = zip(*pairs)
        eng_texts = list(eng_texts)
        spa_texts = list(spa_texts)
        dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(format_dataset, num_parallel_calls=4)
        return dataset.shuffle(2048).prefetch(16).cache()


    train_ds = make_dataset(train_pairs)
    val_ds = make_dataset(val_pairs)

    embed_dim = 256
    dense_dim = 2048
    num_heads = 8

    encoder_inputs = keras.Input(shape=(None,), dtype='int64', name='english')
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="spanish")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
    x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(vocab_size, activation='softmax')(x)
    transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    transformer.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        )
    transformer.fit(train_ds, epochs=30, validation_data=val_ds, callbacks=[])

if __name__ == '__main__':
    main()


