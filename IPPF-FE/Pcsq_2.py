import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
import xgboost
import lightgbm
from Extract_feature import *


def Load_data():
    print('Data Loading...')
    Sequence = []
    Label_Resolution = []
    Label_Rfree = []
    with open('Crystal_structure_quality/Training.fasta', 'r') as myfile:
        for line in myfile:
            if line[0] != '>':
                Sequence.append(line.strip('\n'))
            else:
                i = 6
                while line[i] != '_':
                    i += 1
                i += 1
                start = i
                while line[i] != '_':
                    i += 1
                Label_Resolution.append(float(line[start:i]))
                i += 1
                Label_Rfree.append(float(line[i:-1]))
    with open('Crystal_structure_quality/Test.fasta', 'r') as myfile:
        for line in myfile:
            if line[0] != '>':
                Sequence.append(line.strip('\n'))
            else:
                i = 6
                while line[i] != '_':
                    i += 1
                i += 1
                start = i
                while line[i] != '_':
                    i += 1
                Label_Resolution.append(float(line[start:i]))
                i += 1
                Label_Rfree.append(float(line[i:-1]))
    # sequence truncation
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:1000]
    Mysequence = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i])-1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        Mysequence.append(zj)
    Label_Resolution = np.array(Label_Resolution)
    Label_Rfree = np.array(Label_Rfree)
    return Sequence, Mysequence, Label_Resolution, Label_Rfree


def ALL_features(Sequence, sequences_Example):
    # Crafted features
    features_crafted, min_length = Get_features(Sequence)
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
    return features_crafted, features_normalize, min_length


def Peptide_predict(features_crafted, features_normalize, Label_Resolution):
    features_ensemble = np.concatenate((features_normalize, features_crafted), axis=1)
    Features_packed = (features_crafted, features_normalize, features_ensemble)
    Label = Label_Resolution
    Peptide_data = np.zeros([3, 7, 1], dtype=float)
    for i in range(len(Features_packed)):
        Ifeature = Features_packed[i]
        model1 = RandomForestRegressor()
        model2 = ExtraTreesRegressor()
        model3 = AdaBoostRegressor()
        model4 = BaggingRegressor()
        model5 = xgboost.XGBRegressor()
        model6 = lightgbm.LGBMRegressor()
        model7 = GradientBoostingRegressor()
        j = 0
        for model in (model1, model2, model3, model4, model5, model6, model7):
            kf = KFold(n_splits=5, shuffle=True)
            Pcc = 0
            for train_index, test_index in kf.split(Ifeature, Label):
                Train_data, Train_label = Ifeature[train_index], Label[train_index]
                Test_data, Test_label = Ifeature[test_index], Label[test_index]
                model.fit(Train_data, Train_label)
                Pre_label = model.predict(Test_data)
                Pcc += np.corrcoef(Pre_label, Test_label)[0][1]
            Pcc *= 0.2
            print('Pcc:', Pcc)
            Peptide_data[i][j][0] = Pcc
            j += 1
    data = Peptide_data.reshape((21, 1)).T
    res = pd.DataFrame({"Pcc:": data[0]})
    res.to_excel('Csq_pcc_2.xlsx')


def Peptide_independent(features_crafted, features_normalize, Label_Resolution):
    features_ensemble = np.concatenate((features_normalize, features_crafted), axis=1)
    Label = Label_Resolution
    model = ExtraTreesRegressor()
    Train_data = features_ensemble[:42770]
    Test_data = features_ensemble[42770:]
    Train_label = Label[:42770]
    Test_label = Label[42770:]
    model.fit(Train_data, Train_label)
    Pre_label = model.predict(Test_data)
    Pcc = np.corrcoef(Pre_label, Test_label)[0][1]
    print('Pcc:', Pcc)


def Write_features(features_crafted, features_normalize, min_length):
    features_ensemble = np.concatenate((features_crafted, features_normalize), axis=1).T
    Mydic= {}
    for i in range(len(features_ensemble)):
        if i < 20:
            Mydic['AAC_'+str(i+1)] = features_ensemble[i]
        elif i < 420:
            Mydic['DPC_'+str(i+1-20)] = features_ensemble[i]
        elif i < 763:
            Mydic['Ctriad_'+str(i+1-420)] = features_ensemble[i]
        elif i < 768:
            Mydic['GAAC_'+str(i+1-763)] = features_ensemble[i]
        elif i < 793:
            Mydic['GDPC_'+str(i+1-768)] = features_ensemble[i]
        elif i < 918:
            Mydic['GTPC_'+str(i+1-793)] = features_ensemble[i]
        elif i < 957:
            Mydic['CTDC_'+str(i+1-918)] = features_ensemble[i]
        elif i < 958:
            Mydic['SE_'+str(i+1-957)] = features_ensemble[i]
        elif i < 997:
            Mydic['CTDT_'+str(i+1-958)] = features_ensemble[i]
        elif i < 1192:
            Mydic['CTDD_'+str(i+1-997)] = features_ensemble[i]
        elif i < 1192+16*(min_length-1):
            Mydic['Moran_'+str(i+1-1192)] = features_ensemble[i]
        elif i < 1192+18*(min_length-1):
            Mydic['SOCN_'+str(i+1-1192-16*(min_length-1))] = features_ensemble[i]
        else:
            Mydic['Embedding_'+str(i+1-18*(min_length-1))] = features_ensemble[i]
    res = pd.DataFrame(Mydic)
    res.to_excel('Features_csq_2.xlsx')


if __name__ == '__main__':
    Sequence, Mysequence, Label_Resolution, Label_Rfree = Load_data()
    ###### Train
    # Sequence, Mysequence = Sequence[:42770], Mysequence[:42770]
    # Sequence, Mysequence, Label_Resolution, Label_Rfree = Sequence[:42770], Mysequence[:42770], Label_Resolution[:42770], Label_Rfree[:42770]
    # features_crafted, features_normalize, _ = ALL_features(Sequence, Mysequence)
    # Peptide_predict(features_crafted, features_normalize, Label_Rfree)
    ###### Test
    # Sequence, Mysequence, Label_Resolution, Label_Rfree = Sequence[:100], Mysequence[:100], Label_Resolution[:100], Label_Rfree[:100]
    features_crafted, features_normalize, min_length = ALL_features(Sequence, Mysequence)
    Peptide_independent(features_crafted, features_normalize, Label_Rfree)
    Write_features(features_crafted, features_normalize, min_length)
