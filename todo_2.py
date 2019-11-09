import torch
from config import config
import torch.nn.functional as F
import torch.nn._functions.thnn.rnnFusedPointwise as fusedBackend
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

_config = config()


def evaluate(golden_list, predict_list):
    t_positive = 0
    f_positive = 0
    f_negative = 0
    matches = []
    for i in range(len(golden_list)):
        match = True
        for j in range(len(predict_list[i])):
            g = golden_list[i][j]
            p = predict_list[i][j]
            if g == 'O':
                if p == 'O':
                    f_positive = f_positive + 1
                else:
                    t_positive = t_positive + 1
            else:
                g = golden_list[i][j][-3:]
                p = predict_list[i][j][-3:]
                if g == p:
                    f_negative = f_negative + 1
                else:
                    match = False
        matches.append(match)
    precise = t_positive / (t_positive + f_positive)
    recall = t_positive / (t_positive + f_negative)
    f1 = (2 * precise * recall) / (precise + recall)
    return f1


def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    if input.is_cuda:
        igates = F.linear(input, w_ih)
        hgates = F.linear(hidden[0], w_hh)
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + ((1 - forgetgate) * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    # get embeds according the char_index.
    input_embeds = model.char_embeds(batch_char_index_matrices)
    #input_embeds = model.non_recurrent_dropout(input_char_embeds)
    s, g, s2, g2 = [], [], [], []
    for i in range(len(batch_word_len_lists)):
        s = []
        perm_idx, sorted_batch_word_len_list = model.sort_input(batch_word_len_lists[i])
        sorted_input_embeds = input_embeds[i][perm_idx]
        _, desorted_indices = torch.sort(perm_idx, descending=False)
        output_sequence = pack_padded_sequence(sorted_input_embeds, lengths=sorted_batch_word_len_list.data.tolist(),batch_first=True)
        output_sequence, state = model.char_lstm(output_sequence)
        output_sequence, _ = pad_packed_sequence(output_sequence, batch_first=True)
        output_sequence = output_sequence[desorted_indices]

        for j in range((len(output_sequence))):
            s2.append(output_sequence[j][0])
            for n in range(len(output_sequence[j])):
                add = False
                for m in range(len(output_sequence[j][n])):
                    if output_sequence[j][len(output_sequence[j])-1-n][m] != 0:
                        s.append(output_sequence[j][len(output_sequence[j])-1-n])
                        add = True
                        break
                if add:
                    if j == len(output_sequence) - 1:
                        g.append(s)
                        g2.append(s2)
                    break

    z = torch.Tensor(len(batch_word_len_lists), len(batch_word_len_lists[0]), _config.char_embedding_dim * 2)
    for i in range(len(z)):
        for j in range(len(z[i])):
            for m in range(len(z[i][j])):
                if m < _config.char_embedding_dim:
                    z[i][j][m] = g2[i][j][m]
                else:
                    z[i][j][m] = g[i][j][m]
    return z

