import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
import xgboost
import lightgbm
from Extract_feature import *
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


def Load_data():
    print('Data Loading...')
    Sequence = []
    with open('Golgi protein localization/D3_Train.fasta', 'r') as myfile:
        for line in myfile:
            if line[0] != '>':
                Sequence.append(line.strip('\n'))
    with open('Golgi protein localization/D4_Test.fasta', 'r') as myfile:
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
    features_crafted, min_length = Get_features(Sequence)
    # Automatic extracted features
    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
    gc.collect()
    print(torch.cuda.is_available())
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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


def Peptide_predict(features_crafted, features_normalize):
    features_ensemble = np.concatenate((features_normalize, features_crafted), axis=1)
    Features_packed = (features_crafted, features_normalize, features_ensemble)
    Label = np.concatenate((np.ones([87], dtype=int), np.zeros([217], dtype=int)), axis=0)
    Peptide_data = np.zeros([3, 7, 5], dtype=float)
    for i in range(len(Features_packed)):
        Ifeature = Features_packed[i]
        model1 = RandomForestClassifier()
        model2 = ExtraTreesClassifier()
        model3 = AdaBoostClassifier()
        model4 = BaggingClassifier()
        model5 = xgboost.XGBClassifier()
        model6 = lightgbm.LGBMClassifier()
        model7 = GradientBoostingClassifier()
        j = 0
        for model in (model1, model2, model3, model4, model5, model6, model7):
            kf = KFold(n_splits=5, shuffle=True)
            Acc = 0
            Sens = 0
            Spec = 0
            MCC = 0
            auROC = 0
            for train_index, test_index in kf.split(Ifeature, Label):
                Train_data, Train_label = Ifeature[train_index], Label[train_index]
                Test_data, Test_label = Ifeature[test_index], Label[test_index]
                model.fit(Train_data, Train_label)
                Pre_label = model.predict(Test_data)
                Acc += metrics.accuracy_score(Test_label, Pre_label)
                MCC += metrics.matthews_corrcoef(Test_label, Pre_label)
                CM = metrics.confusion_matrix(Test_label, Pre_label)
                Pre_label_prob = model.predict_proba(Test_data)
                auROC += metrics.roc_auc_score(Test_label, Pre_label_prob[:, 1])
                Spec += CM[0][0] / (CM[0][0] + CM[0][1])
                Sens += CM[1][1] / (CM[1][0] + CM[1][1])
            Acc *= 0.2
            Sens *= 0.2
            Spec *= 0.2
            MCC *= 0.2
            auROC *= 0.2
            print('Accuracy:', Acc, " Sensitivity", Sens, " Specificity", Spec, "MCC", MCC, "auROC", auROC)
            Peptide_data[i][j][0] = Acc
            Peptide_data[i][j][1] = Sens
            Peptide_data[i][j][2] = Spec
            Peptide_data[i][j][3] = MCC
            Peptide_data[i][j][4] = auROC
            j += 1
    data = Peptide_data.reshape((21, 5)).T
    res = pd.DataFrame({"Accuracy:": data[0], " Sensitivity": data[1], " Specificity": data[2],
                        "MCC": data[3], "auROC": data[4]})
    res.to_excel('Protein_gpl.xlsx')


def Peptide_independent(features_crafted, features_normalize):
    features_ensemble = np.concatenate((features_normalize, features_crafted), axis=1)
    Label = np.concatenate((np.ones([87], dtype=int), np.zeros([217], dtype=int),
                            np.ones([13], dtype=int), np.zeros([51], dtype=int)), axis=0)
    model = lightgbm.LGBMClassifier()
    Train_data = features_ensemble[:304]
    Test_data = features_ensemble[304:]
    Train_label = Label[:304]
    Test_label = Label[304:]
    model.fit(Train_data, Train_label)
    Pre_label = model.predict(Test_data)
    Acc = metrics.accuracy_score(Test_label, Pre_label)
    MCC = metrics.matthews_corrcoef(Test_label, Pre_label)
    CM = metrics.confusion_matrix(Test_label, Pre_label)
    Pre_label_prob = model.predict_proba(Test_data)
    auROC = metrics.roc_auc_score(Test_label, Pre_label_prob[:, 1])
    fpr, tpr, thresholds = metrics.roc_curve(Test_label, Pre_label_prob[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, color="darkred", lw=1, label="ROC curve (area = %0.2f)" % auROC)
    plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Golgi protein localization receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig('Pgpl.jpg')
    Spec = CM[0][0] / (CM[0][0] + CM[0][1])
    Sens = CM[1][1] / (CM[1][0] + CM[1][1])
    print('Accuracy:', Acc, " Sensitivity", Sens, " Specificity", Spec, "MCC", MCC, "auROC", auROC)


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
            Mydic['Embedding_'+str(i+1-1192-18*(min_length-1))] = features_ensemble[i]
    res = pd.DataFrame(Mydic)
    res.to_excel('Features_gpl.xlsx')


if __name__ == '__main__':
    Sequence, Mysequence = Load_data()
    ###### Train
    # Sequence, Mysequence = Sequence[:304], Mysequence[:304]
    # features_crafted, features_normalize, _ = ALL_features(Sequence, Mysequence)
    # Peptide_predict(features_crafted, features_normalize)
    # Write_features(features_crafted, features_normalize, min_length)
    ###### Test
    features_crafted, features_normalize, min_length = ALL_features(Sequence, Mysequence)
    Peptide_independent(features_crafted, features_normalize)
