
import torch.nn.functional as F
import torch.nn._functions.thnn.rnnFusedPointwise as fusedBackend
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import config


import torch



_config = config()


def evaluate(golden_list, predict_list):

    num_predict_tag, num_golden_tag = 0, 0

    num_correct = 0

    times = 0

    for golden, predict in zip(golden_list, predict_list):

        times+=1

        pre_predict, pre_golden = [], []

        time = 0

        for _gol, _pred in zip(golden, predict):

            time += 1

            gol_list = _gol.split('-')

            pred_list = _pred.split('-')

            if len(gol_list)<2: gol_list.append('')
            if len(pred_list) < 2: pred_list.append('')


            len_gol, len_pred = len(pre_golden), len(pre_predict)

            if len_gol == len_pred and len_gol > 0 and '|'.join(pre_golden) == '|'.join(pre_predict):
                if gol_list[0] != 'I' and pred_list[0] != 'I':
                    num_correct += 1
                else:
                    _list_g = pre_golden[-1].split('-')
                    _list_p = pre_predict[-1].split('-')
                    if gol_list[1]!=_list_g[1] and pred_list[1]!=_list_p[1]:
                        num_correct += 1


            if gol_list[0] != 'I':
                if len_gol > 0 : num_golden_tag += 1
                pre_golden = []
                if gol_list[0] == 'B': pre_golden.append(_gol)
            else:
                if len_gol > 0:
                    _list = pre_golden[-1].split('-')
                    if gol_list[1] == _list[1]: pre_golden.append(_gol)
                    else:
                        pre_golden = [_gol]
                else: pre_golden = [_gol]

            if pred_list[0] != 'I':
                if len_pred > 0 : num_predict_tag += 1
                pre_predict = []
                if pred_list[0] == 'B': pre_predict.append(_pred)
            else:
                if len_pred > 0:
                    _list = pre_predict[-1].split('-')
                    if pred_list[1] == _list[1]: pre_predict.append(_pred)
                    else:
                        pre_predict = [_pred]
                else: pre_predict = [_pred]

        if len(pre_golden) > 0: num_golden_tag += 1
        if len(pre_predict) > 0: num_predict_tag += 1

        if '|'.join(pre_predict) == '|'.join(pre_golden): num_correct += 1

    #print("predict: %d\n golden: %d\n correct: %d\n" % (num_predict_tag, num_golden_tag, num_correct))

    F_1 = 0

    try:
        precision = float(num_correct)/float(num_predict_tag)
        recall = float(num_correct)/float(num_golden_tag)
        F_1 = 2*(precision*recall)/(precision+recall)

    except:
        pass

    finally:
        return  F_1

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

    #maxlen = max([len(e) for e in batch_word_len_lists])

    #input_char_embeds = batch_char_index_matrices.view(batch_char_index_matrices.shape[0]*batch_char_index_matrices.shape[1], batch_char_index_matrices.shape[2])

    input_char_embeds = model.char_embeds(batch_char_index_matrices)

    #input_word_len_lists = batch_word_len_lists.view(len(batch_word_len_lists)*maxlen)

    input_embeds = input_char_embeds.view(input_char_embeds.shape[0]*input_char_embeds.shape[1], input_char_embeds.shape[2], input_char_embeds.shape[3])

    input_batch_word_len_lists = batch_word_len_lists.view(batch_word_len_lists.shape[0]*batch_word_len_lists.shape[1])


    perm_idx, sorted_input_word_len_list = model.sort_input(input_batch_word_len_lists)

    sorted_input_embeds = input_embeds[perm_idx]

    _, desorted_indices = torch.sort(perm_idx, descending=False)

    output_sequence = pack_padded_sequence(sorted_input_embeds, lengths=sorted_input_word_len_list.data.tolist(),
                                           batch_first=True)

    output_sequence, state = model.char_lstm(output_sequence)

    output_sequence, _ = pad_packed_sequence(output_sequence, batch_first=True)
    output_sequence = output_sequence[desorted_indices]

    result = torch.Tensor(len(batch_word_len_lists), len(batch_word_len_lists[0]), _config.char_embedding_dim * 2)

    len_of_sentence = batch_word_len_lists.shape[1]


    for index_word in range(len(input_batch_word_len_lists)):

        if(input_batch_word_len_lists[index_word] == 0):
            result[index_word // len_of_sentence][index_word % len_of_sentence] = torch.FloatTensor([0 for e in range(_config.char_embedding_dim*2)])


        forward = output_sequence[index_word][0][:_config.char_embedding_dim]

        backward = output_sequence[index_word][input_batch_word_len_lists[index_word]-1][_config.char_embedding_dim:_config.char_embedding_dim*2]

        res = torch.cat([forward, backward], dim=-1)
            #.cat([forward, backward], dim=-1)

        result[index_word//len_of_sentence][index_word%len_of_sentence] = res

        #result[index_word // len_of_sentence][index_word % len_of_sentence].append(backward)


    return result












