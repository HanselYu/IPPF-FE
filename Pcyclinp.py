import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
from sklearn.ensemble import ExtraTreesClassifier
from Extract_feature import *
import pickle
import sys


def Load_data():
    print('Data Loading...')
    Sequence = []
    with open('Cyclin protein/Dataset.fasta', 'r') as myfile:
        for line in myfile:
            if line[0] != '>':
                Sequence.append(line.strip('\n'))
    for i in range(len(Sequence)):
        Sequence[i] = Sequence[i][:1000]
    Mysequence = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i])-1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        Mysequence.append(zj)
    return Sequence, Mysequence


def ALL_features(Sequence, sequences_Example):
    # Crafted features
    features_crafted = Get_features(Sequence, 104)
    # Automatic extracted features
    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
    gc.collect()
    print(torch.cuda.is_available())
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print('For sequence ', str(i+1))
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    print(len(features_normalize), len(features_normalize[0]))
    return features_crafted, features_normalize


def Protein_cyc(features_crafted, features_normalize):
    features_ensemble = np.concatenate((features_normalize, features_crafted), axis=1)
    Label = np.concatenate((np.ones([166], dtype=int), np.zeros([167], dtype=int)), axis=0)
    model = ExtraTreesClassifier()
    model.fit(features_ensemble, Label)
    with open('Protein_cyc.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    Tag = sys.argv[1]
    Dir = sys.argv[2]
    if Tag == 'Train':
        Sequence, Mysequence = Load_data()
        features_crafted, features_normalize = ALL_features(Sequence, Mysequence)
        Protein_cyc(features_crafted, features_normalize)
    elif Tag == 'Predict':
        Sequence = []
        with open(Dir, 'r') as myfile:
            for line in myfile:
                if line[0] != '>':
                    Sequence.append(line.strip('\n'))
        Mysequence = []
        for i in range(len(Sequence)):
            zj = ''
            for j in range(len(Sequence[i])-1):
                zj += Sequence[i][j] + ' '
            zj += Sequence[i][-1]
            Mysequence.append(zj)
        features_crafted, features_normalize = ALL_features(Sequence, Mysequence)
        features_ensemble = np.concatenate((features_normalize, features_crafted), axis=1)
        with open('Protein_cyc.pkl', 'rb') as myfile:
            model = pickle.load(myfile)
            Pre_label = model.predict(features_ensemble)
            print(Pre_label)
        Pre_label = np.array(Pre_label).T
        res = pd.DataFrame({'Pre_label': Pre_label})
        res.to_excel('Pre_label.xlsx')
    else:
        print('Please input Train/Test/Predict !')
