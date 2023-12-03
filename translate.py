import numpy as np
import random
import string
from main import source_vectorization, target_vectorization, transformer, test_pairs


def translate():
    spa_vocab = target_vectorization.get_vocabulary()
    spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
    max_decoded_sentence_length = 20

    def decode_sequence(input_sentence):
        tokenized_input_sentence = source_vectorization([input_sentence])
        decoded_sentence = "[start]"
        for i in range(max_decoded_sentence_length):
            tokenized_target_sentence = target_vectorization(
                [decoded_sentence])[:, :-1]
            predictions = transformer(
                [tokenized_input_sentence, tokenized_target_sentence])
            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = spa_index_lookup[sampled_token_index]
            decoded_sentence += " " + sampled_token
            if sampled_token == "[end]":
                break
        return decoded_sentence

    test_eng_texts = [pair[0] for pair in test_pairs]
    for _ in range(20):
        input_sentence = random.choice(test_eng_texts)
        print("-")
        print(input_sentence)
        print(decode_sequence(input_sentence))
