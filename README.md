# CS6714_Project_Stage_2
[project specification](https://github.com/zyjwarlock/CS6714_Project_Stage_2/blob/master/COMP6714%20Project%20Specification%20(stage%202).ipynb) 
## COMP6714 18s2 Project Report

### How do you implement evaluate() ?
This task aims to output the f1-score. To calculate the *f1-score*, the values of *precision* and *recall*
should be figured out. I define 3 variables *num_predict_tag*, *num_golden_tag* and *num_correct* to
store the label amount of predict list and golden list, as well as the number of correct prediction, respectively during the traversal through the predict list and golden list.
Importantly, a label should satisfy
 * First tag and only the first tag starts with ‘B’ prefix. 
 * Each tag inside the label has the same suffix.
 * There is not any ‘O’ tag inside the label
 
Each time a tag starts with ‘B’ or is ‘O’ whatever in predict list or golden list, the previous tags should be stored as a label and the tag numbers of current list added 1. In addition, compare this label’s length, position and tags to the other one. If these 3 condition are satisfied, the labels are matched, the correct number should be added 1.
Moreover, the value of false positive is the value of *num_predict_tag - num_correct* and value of false negative is the value of *num_golden_tag - num_correct*, as well as the true positive equals to true negative which is the value of *num_correct*.
Finally, the values of *f1-score*, *precision* and *recall* can be drawn out via those 3 values.

### How does Modification 1 (i.e., storing model with best performance on the development set) affect the performance ?
F1-score can be employed in optimization process, as it considers both values of the precision and the recall. The f1-score used in the real data set of application, the performance can be more stable.

### How do you implement new_LSTMCell() ?
Importing the functional from torch.nn.
Using the source code in ```torch.nn._fuuctions.run.LSTMCell()``` and modify the cy calculation line.
    
### How does Modification 2 (i.e., re-implemented LSTM cell) affect the performance ?
The changes can let the model achieved better fitting of training data, the loss of training data will be reduced.

### How do you implement get_char_sequence() ?
The ```input_char_emdeds``` is got by using model’s ```char_embeds``` function, then this variable should be reshaped via using ```view()``` function, and the word_len_lists array should also be reshaped as the first dimension of the input_char_emdeds matrix. Moreover, both of the word_len_lists array and input_char_emdeds should be sorted in a decreasing order. After that, follow the steps of ```_rnn()``` function in *model.py*, and the result of pack_padded_sequence will be feed the char_lstm model. Then, the return value of char_lstm model should be desorted. The first dimension of output_squence is the word index, as well as the second dimension is the index of each character
in words. I concatenate the last 50 vectors of first character and the first 50 vectors of last character of each word (the index of last character is stored in the word_list array) as the new vectors. Finally, return the matrix after reshaping its dimension.

### How does Modification 3 (i.e., adding Char BiLSTM layer) affect the performance ?
Obviously, the performance of the model will be better. In many cases, the interior of a word semantic information, word embedding won't consider these information, but character embedding will consider these additional information to achieve better performance. Also, character embeddings work better than word embeddings when dealing words which is not in vocabulary. However, running the char bilstm layer will raise the time cost.
   
