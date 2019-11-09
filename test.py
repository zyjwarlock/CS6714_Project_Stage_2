from todo import evaluate


list_1 = [['B-TAR', 'I-TAR', 'I-TAR', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
#[['B-TAR', 'I-TAR','I-TAR', 'I-TAR','O', 'B-HYP']]
#[['B-TAR', 'I-TAR', 'O', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]

list_2 =  [['B-TAR', 'B-TAR', 'I-HYP', 'O'], ['I-TAR', 'O', 'O', 'O']]
#[['B-TAR','I-TAR', 'B-HYP','I-HYP','O', 'B-HYP']]
#[['B-TAR', 'O', 'O', 'O'], ['B-TAR', 'O', 'B-HYP', 'I-HYP']]

print (evaluate(list_1, list_2))