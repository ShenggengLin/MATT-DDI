
###########################prepareData
import numpy as np
from pandas import DataFrame



def prepare(df_drug, feature_list, mechanism, action, drugA, drugB):
    d_label = {}
    d_feature = {}

    # Transfrom the interaction event to number
    d_event = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])

    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    event_num = len(count)
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i

    vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)  # vector=[]
    for i in feature_list:
        # vector = np.hstack((vector, feature_vector(i, df_drug, vector_size)))#1258*1258
        tempvec = feature_vector(i, df_drug)
        vector = np.hstack((vector, tempvec))
    # Transfrom the drug ID to feature vector
    for i in range(len(np.array(df_drug['name']).tolist())):
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]

    # Use the dictionary to obtain feature vector and label
    new_feature = []
    new_label = []

    for i in range(len(d_event)):
        temp = np.hstack((d_feature[drugA[i]], d_feature[drugB[i]]))
        new_feature.append(temp)
        new_label.append(d_label[d_event[i]])

    new_feature = np.array(new_feature)  # 323539*....
    new_label = np.array(new_label)  # 323539

    return new_feature, new_label, drugA, drugB, event_num

# In[104]:


def feature_vector(feature_name, df):
    def Jaccard(matrix):
        matrix = np.mat(matrix)

        numerator = matrix * matrix.T

        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T

        return numerator / denominator

    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # obtain all the features
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 1

    df_feature = np.array(df_feature)
    #sim_matrix = np.array(Jaccard(df_feature))

    print(feature_name + " len is:" + str(len(df_feature[0])))
    return df_feature

####################################model#########
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
#from hyperPara import *
import math

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class MultiHeadSelfAttentionDot(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadSelfAttentionDot, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output

class MultiHeadCroAttentionDot(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadCroAttentionDot, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim)

    def forward(self, X,Y):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(Y).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(Y).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output
# In[107]:

class MultiHeadSelfAttentionBil(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadSelfAttentionBil, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)

        self.linear_w = torch.nn.Linear(input_dim//n_heads, input_dim//n_heads)

        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        alpha = torch.matmul(Q, self.linear_w(K).transpose(-1, -2))
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(alpha)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output


class MultiHeadCroAttentionBil(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadCroAttentionBil, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)

        self.linear_w = torch.nn.Linear(input_dim//n_heads,input_dim//n_heads)

        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim)

    def forward(self, X,Y):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(Y).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(Y).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        alpha = torch.matmul(Q, self.linear_w(K).transpose(-1, -2))
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(alpha)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output

class EncoderLayerSelfDot(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayerSelfDot, self).__init__()
        self.attn1 = MultiHeadSelfAttentionDot(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)
        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

        self.attn2 = MultiHeadSelfAttentionDot(input_dim, n_heads)
        self.AN3 = torch.nn.LayerNorm(input_dim)
        self.l2 = torch.nn.Linear(input_dim, input_dim)
        self.AN4 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn1(X)
        X = self.AN1(output + X)
        output = self.l1(X)
        X = self.AN2(output + X)

        output = self.attn2(X)
        X = self.AN3(output + X)
        output = self.l2(X)
        X = self.AN4(output + X)

        return X

class EncoderLayerCroDot(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayerCroDot, self).__init__()
        self.attn = MultiHeadCroAttentionDot(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X, Y):
        output = self.attn(X, Y)
        Z = self.AN1(output + X + Y)

        output = self.l1(Z)
        output = self.AN2(output + Z)

        return output
# In[108]:
class EncoderLayerSelfBil(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayerSelfBil, self).__init__()
        self.attn1 = MultiHeadSelfAttentionBil(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)
        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

        self.attn2 = MultiHeadSelfAttentionBil(input_dim, n_heads)
        self.AN3 = torch.nn.LayerNorm(input_dim)
        self.l2 = torch.nn.Linear(input_dim, input_dim)
        self.AN4 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn1(X)
        X = self.AN1(output + X)
        output = self.l1(X)
        X = self.AN2(output + X)

        output = self.attn2(X)
        X = self.AN3(output + X)
        output = self.l2(X)
        X = self.AN4(output + X)

        return X
class EncoderLayerCroBil(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayerCroBil, self).__init__()
        self.attn = MultiHeadCroAttentionBil(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X,Y):
        output = self.attn(X,Y)
        Z = self.AN1(output + X+Y)

        output = self.l1(Z)
        output = self.AN2(output + Z)

        return output


class DotModel(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(DotModel, self).__init__()
        self.selfDotA = EncoderLayerSelfDot(input_dim, n_heads)


        self.selfDotB = EncoderLayerSelfDot(input_dim, n_heads)


        self.CroDot = EncoderLayerCroDot(input_dim, n_heads)

        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.bn1 = torch.nn.BatchNorm1d(input_dim)

        self.ac = gelu
        self.dr = torch.nn.Dropout(drop_out_rating)

    def forward(self, X, Y):
        X = self.selfDotA(X)
        Y = self.selfDotB(Y)


        output=self.CroDot(X,Y)

        output = self.dr(self.ac(self.bn1(self.linear1(output))))

        return output


class BilModel(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(BilModel, self).__init__()
        self.selfBilA = EncoderLayerSelfBil(input_dim, n_heads)
        self.selfBilB = EncoderLayerSelfBil(input_dim, n_heads)



        self.CroBil = EncoderLayerCroBil(input_dim, n_heads)

        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.bn1 = torch.nn.BatchNorm1d(input_dim)

        self.ac = gelu
        self.dr = torch.nn.Dropout(drop_out_rating)

    def forward(self, X, Y):
        X = self.selfBilA(X)

        Y = self.selfBilB(Y)

        output = self.CroBil(X, Y)

        output = self.dr(self.ac(self.bn1(self.linear1(output))))

        return output

class Model(torch.nn.Module):
    def __init__(self, input_dim, n_heads, event_num):
        super(Model, self).__init__()

        self.dim = input_dim//2

        self.DotModel = DotModel(self.dim, n_heads)
        self.BilModel = BilModel(self.dim, n_heads)

        self.l1 = torch.nn.Linear(self.dim*5,self.dim)
        self.bn1 = torch.nn.BatchNorm1d(self.dim)

        self.l2 = torch.nn.Linear(self.dim , (self.dim  + event_num) // 2)
        self.bn2 = torch.nn.BatchNorm1d((self.dim  + event_num) // 2)

        self.l3 = torch.nn.Linear((self.dim+event_num)//2, event_num)

        self.ac = gelu
        self.dr = torch.nn.Dropout(drop_out_rating)

    def forward(self, X,X_pair):

        drugXA=X[:,:self.dim]
        drugXB = X[:, self.dim:]

        drugXpairA = X_pair[:, :self.dim]
        drugXpairB = X_pair[:, self.dim:]

        dot_out1=self.DotModel(drugXA,drugXB)
        bil_out1 = self.BilModel(drugXpairA,drugXpairB)

        dot_out2 = self.DotModel(drugXpairA, drugXpairB)
        bil_out2 = self.BilModel(drugXA,drugXB)



        dot_bil_out=dot_out1+bil_out2

        output = torch.cat((dot_out1, bil_out1, dot_out2, bil_out2,dot_bil_out), 1)

        output = self.dr(self.ac(self.bn1(self.l1(output))))

        output = self.dr(self.ac(self.bn2(self.l2(output))))

        output = self.l3(output)

        return output





#####################modelTrain#######

import numpy as np


import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

#from hyperPara import *



class DDIDataset(Dataset):
    def __init__(self, data_train_sour,data_train_pair,label_train):
        self.len = len(label_train)
        self.data_train_sour= torch.from_numpy(np.array(data_train_sour))
        self.data_train_pair = torch.from_numpy(np.array(data_train_pair))

        self.y_data = torch.from_numpy(np.array(label_train))

    def __getitem__(self, index):
        return self.data_train_sour[index], self.data_train_pair[index],self.y_data[index]

    def __len__(self):
        return self.len




def Model_train(model, data_train_sour,data_train_pair,label_train,X_test,data_test_pair,y_test, event_num):
    model_optimizer = torch.optim.Adam(model.parameters(),lr=learn_rating,weight_decay=weight_decay_rate)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    classi_loss = torch.nn.CrossEntropyLoss()
    len_train=len(label_train)
    len_test=len(y_test)

    train_dataset = DDIDataset(data_train_sour,data_train_pair,label_train)
    test_dataset = DDIDataset(X_test,data_test_pair,np.array(y_test))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epo_num):

        running_loss = 0.0

        model.train()
        for batch_idx, data in enumerate(train_loader, 0):
            x_sour, x_pair,y = data





            x_sour=x_sour.to(device)
            x_pair = x_pair.to(device)
            y = y.to(device)



            model_optimizer.zero_grad()
            # forward + backward+update
            X= model(x_sour.float(),x_pair.float())

            loss = classi_loss(X,y)

            loss.backward()
            model_optimizer.step()
            running_loss += loss.item()

        model.eval()
        testing_loss = 0.0
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader, 0):
                x_sour, x_pair,y = data

                x_sour = x_sour.to(device)
                x_pair = x_pair.to(device)
                y = y.to(device)

                X= model(x_sour.float(),x_pair.float())

                loss = classi_loss(X, y)
                testing_loss += loss.item()
        print('epoch [%d] loss: %.6f testing_loss: %.6f ' % (
        epoch + 1, running_loss / len_train, testing_loss / len_test))

    pre_score = np.zeros((0, event_num), dtype=float)
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            x_sour, x_pair,y = data
            x_sour = x_sour.to(device)
            x_pair = x_pair.to(device)
            X= model(x_sour.float(),x_pair.float())
            pre_score = np.vstack((pre_score, F.softmax(X).cpu().numpy()))
    return pre_score



############################evaluate

import numpy as np

from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize


def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)
def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 6
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, classes=np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    #result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[2] = roc_auc_score(y_one_hot, pred_score, average='micro')
    #result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    #result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[3] = f1_score(y_test, pred_type, average='macro')
    #result_all[7] = precision_score(y_test, pred_type, average='micro')
    result_all[4] = precision_score(y_test, pred_type, average='macro')
    #result_all[9] = recall_score(y_test, pred_type, average='micro')
    result_all[5] = recall_score(y_test, pred_type, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                         average=None)
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, result_eve]



#######################



from numpy.random import seed
import csv
import time
import numpy as np
import random
import pandas as pd
from pandas import DataFrame
import sqlite3

import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

#from model import Model
#from prepareData import prepare
#from evaluate import evaluate
#from modelTrain import Model_train
#from hyperPara import *

import warnings
warnings.filterwarnings("ignore")

seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# In[117]:
def cos_sim(array1,array2):
    cos_sim = cosine_similarity(array1,array2)
    index = []
    for i in range(len(cos_sim)):
        index.append(pd.Series(cos_sim[i]).sort_values(ascending=False).index[:pair_num].tolist())
    return index


def cross_val(feature, label, drugA, drugB, event_num):
    y_true = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    y_pred = np.array([])

    temp_drug1 = [[] for i in range(event_num)]
    temp_drug2 = [[] for i in range(event_num)]
    for i in range(len(label)):
        temp_drug1[label[i]].append(drugA[i])
        temp_drug2[label[i]].append(drugB[i])
    drug_cro_dict = {}
    for i in range(event_num):
        for j in range(len(temp_drug1[i])):
            drug_cro_dict[temp_drug1[i][j]] = j % cross_ver_tim
            drug_cro_dict[temp_drug2[i][j]] = j % cross_ver_tim
    train_drug = [[] for i in range(cross_ver_tim)]
    test_drug = [[] for i in range(cross_ver_tim)]
    for i in range(cross_ver_tim):
        for dr_key in drug_cro_dict.keys():
            if drug_cro_dict[dr_key] == i:
                test_drug[i].append(dr_key)
            else:
                train_drug[i].append(dr_key)

    for cross_ver in range(cross_ver_tim):

        model = Model(len(feature[0]), Att_n_heads, event_num)

        X_train = []
        X_test = []
        y_train = []
        y_test = []

        for i in range(len(drugA)):
            if (drugA[i] in np.array(train_drug[cross_ver])) and (drugB[i] in np.array(train_drug[cross_ver])):
                X_train.append(feature[i])
                y_train.append(label[i])

            if (drugA[i] not in np.array(train_drug[cross_ver])) and (drugB[i] in np.array(train_drug[cross_ver])):
                X_test.append(feature[i])
                y_test.append(label[i])

            if (drugA[i] in np.array(train_drug[cross_ver])) and (drugB[i] not in np.array(train_drug[cross_ver])):
                X_test.append(feature[i])
                y_test.append(label[i])

        print("train len", len(y_train))
        print("test len", len(y_test))

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        train_pair_index = np.array(cos_sim(X_train, X_train))
        data_train_pair = []
        for i in range(len(X_train)):
            temp_X = X_train[train_pair_index[i, 1]]  # delete oneself
            for j in range(2, pair_num):
                X_pair = X_train[train_pair_index[i, j]]
                temp_X = np.sum([temp_X, X_pair], axis=0).tolist()
            temp_X = (np.array(temp_X) / (pair_num - 1)).tolist()
            data_train_pair.append(temp_X)

        x_train1 = np.hstack(
            (X_train[:, len(X_train[0]) // 2:], X_train[:, :len(X_train[0]) // 2]))  # change drugA drugB location
        x_train2 = X_train
        x_train3 = np.hstack((X_train[:, len(X_train[0]) // 2:], X_train[:, :len(X_train[0]) // 2]))
        x_train = np.vstack((X_train, x_train1, x_train2, x_train3))

        data_train_pair = np.array(data_train_pair)
        data_train_pair1 = data_train_pair
        data_train_pair2 = np.hstack((data_train_pair[:, len(data_train_pair[0]) // 2:], data_train_pair[:, :len(data_train_pair[0]) // 2]))
        data_train_pair3 = np.hstack((data_train_pair[:, len(data_train_pair[0]) // 2:], data_train_pair[:, :len(data_train_pair[0]) // 2]))
        data_train_pair = np.vstack((data_train_pair, data_train_pair1, data_train_pair2, data_train_pair3))

        y_train = np.hstack((y_train, y_train, y_train, y_train))

        data_test_pair = []
        test_pair_index = np.array(cos_sim(X_test, X_train))
        for i in range(len(X_test)):
            temp_X = X_train[test_pair_index[i, 0]]
            for j in range(1, pair_num - 1):  # Keep the same number of paired drugs  with train
                X_pair = X_train[test_pair_index[i, j]]
                temp_X = np.sum([temp_X, X_pair], axis=0).tolist()
            temp_X = (np.array(temp_X) / (pair_num - 1)).tolist()
            data_test_pair.append(temp_X)

        pred_score = Model_train(model, x_train, data_train_pair, y_train, X_test, data_test_pair, y_test, event_num)

        pred_type = np.argmax(pred_score, axis=1)
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))

        y_true = np.hstack((y_true, y_test))

    result_all, result_eve = evaluate(y_pred, y_score, y_true, event_num)

    return result_all, result_eve
########################hyperPara
#!/usr/bin/env python
# coding: utf-8
import os
import torch


Att_n_heads=4
drop_out_rating=0.5
batch_size=128
learn_rating=1.0e-5
epo_num=100
cross_ver_tim=5
pair_num=51
weight_decay_rate=1.0e-5
feature_list = ["smile","target","enzyme"]
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
###############################################################


file_path="./"


def save_result(filepath,result_type,result):
    with open(filepath+result_type +'task2'+ '.csv', "w", newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

def main():
    
    conn = sqlite3.connect("./event.db")
    
    df_drug = pd.read_sql('select * from drug;', conn)
    extraction = pd.read_sql('select * from extraction;', conn)
    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']

    new_feature, new_label, drugA, drugB, event_num = prepare(df_drug, feature_list, mechanism, action, drugA, drugB)
    np.random.seed(seed)
    np.random.shuffle(new_feature)
    np.random.seed(seed)
    np.random.shuffle(new_label)
    np.random.seed(seed)
    np.random.shuffle(drugA)
    np.random.seed(seed)
    np.random.shuffle(drugB)
    print("dataset len", len(new_feature))
    
    start=time.time()
    result_all, result_eve=cross_val(new_feature,new_label,drugA,drugB,event_num)
    print("time used:", (time.time() - start) / 3600)
    save_result(file_path,"all",result_all)
    save_result(file_path,"each",result_eve)

main()

