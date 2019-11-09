import torch
import numpy as np
from config import config
from model import sequence_labeling
from randomness import apply_random_seed
from data_io import DataReader, gen_embedding_from_file, read_tag_vocab
from todo import get_char_sequence


UNKNOWN_WORD = "<UNK_WORD>"
UNKNOWN_CHAR = "<UNK_CHAR>"


sen1 = ['Potion', 'Mastery', 'is', 'specialization', 'of', 'Alchemy', '.']
sen2 = ['A', 'Guild', 'is', 'association', 'of', 'craftsmen', '.']

_config = config()
apply_random_seed()

tag_dict = read_tag_vocab(config.output_tag_file)
reversed_tag_dict = {v: k for (k, v) in tag_dict.items()}
word_embedding, word_dict = gen_embedding_from_file(config.word_embedding_file, config.word_embedding_dim)
char_embedding, char_dict = gen_embedding_from_file(config.char_embedding_file, config.char_embedding_dim)

_config.nwords = len(word_dict)
_config.ntags = len(tag_dict)
_config.nchars = len(char_dict)
model = sequence_labeling(_config, word_embedding, char_embedding)


def get_word_ids(w):
    word = w.lower()
    if word in word_dict:
        return word_dict[word]
    else:
        return word_dict[UNKNOWN_WORD]

def get_char_ids(c):
    if c in char_dict:
        return char_dict[c]
    else:
        return char_dict[UNKNOWN_CHAR]



sentence_list = [sen1] + [sen2]

word_index_lists = [[get_word_ids(word) for word in sentence] for sentence in sentence_list]
char_index_matrix = [[[get_char_ids(char) for char in word] for word in sentence] for sentence in sentence_list]
word_len_lists = [[len(word) for word in sentence] for sentence in char_index_matrix]
sentence_len_list = [len(x) for x in word_len_lists]

batch_char_index_matrices = np.zeros((len(word_index_lists), max(sentence_len_list), max(map(max, word_len_lists))),
                                     dtype=int)
for i, (char_index_matrix, word_len_list) in enumerate(zip(char_index_matrix, word_len_lists)):
    for j in range(len(word_len_list)):
        batch_char_index_matrices[i, j, :word_len_list[j]] = char_index_matrix[j]

batch_word_len_lists = np.ones((len(word_index_lists), max(sentence_len_list)),
                               dtype=int)  # cannot set default value to 0
for i, (word_len, sent_len) in enumerate(zip(word_len_lists, sentence_len_list)):
    batch_word_len_lists[i, :sent_len] = word_len

batch_word_len_lists = torch.from_numpy(np.array(batch_word_len_lists)).long()
batch_char_index_matrices = torch.from_numpy(batch_char_index_matrices).long()

answer = get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists)


answer = answer.data.numpy()
result = np.load('./answer.npy')



try:
    assert np.allclose(np.asarray(answer.tolist()), np.asarray(result.tolist()), atol=0.001)
    print('Your implementation is Correct')
except:
    print('Your implementation is not Correct')