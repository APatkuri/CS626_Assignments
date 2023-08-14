# Lot to be done!
# Optimize hyperparameter v_size, epochs
# Need a good architecture (FFNN) for POS tagging, also some preprocessing as the data is biased 
# Viterbi symbolic and Viterbi vectorized give similar results
# Upload in Colab
# Suppress Warnings
# Vectorize Vectorize Vectorize, even Viterbi as the current version is damn slow
import nltk
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import metrics
from collections import Counter
from collections import defaultdict
from nltk.corpus import brown
import gensim
from gensim.models import Word2Vec
import torch
from torch import nn
import torch.nn.functional as F 

# The following has to be performed only once
# nltk.download('brown')
# nltk.download('universal_tagset')

stime = time.time()
v_size = 60 # Word Vector Size

tagset = ['NOUN', 'VERB', '.', 'ADP', 'DET', 'ADJ', 'ADV', 'PRON', 'CONJ', 'PRT', 'NUM', 'X']
identity = torch.eye(12, dtype =torch.double)
tagset_dict = {'NOUN':identity[0,:], 'VERB':identity[1,:], '.':identity[2,:], 'ADP':identity[3,:], 'DET':identity[4,:], 'ADJ':identity[5,:], 'ADV':identity[6,:], 'PRON':identity[7,:], 'CONJ':identity[8,:], 'PRT':identity[9,:], 'NUM':identity[10,:], 'X':identity[11,:]}

words = list(brown.tagged_words(tagset='universal'))
print(len(words))


def tag_frequency(tag, words):
    total = len(words)
    tagcount = Counter(tag) 
    tagfreq = {}

    for i in tagcount.keys():
        tagfreq[i] = tagcount[i]/total

    return tagcount, tagfreq

def emission_probs(tokens, words, tagcount):
    tokenTags = defaultdict(Counter)

    for token_i, tag_i in words:
        tokenTags[token_i][tag_i] += 1
    
    for tag_i in tagset:
        for token_i in tokens:
            if tokenTags[token_i][tag_i] >= 1:
                tokenTags[token_i][tag_i] = tokenTags[token_i][tag_i]/(tagcount[tag_i])

    return tokenTags

def transition_probs(tag, words, tagcount):
    tokens, tag = zip(*words)
    pl = len(tag) - 1
    tagtags = defaultdict(Counter)

    for i in range(pl-1):
        tagtags[tag[i]][tag[i+1]] += 1

    for tag_i in tagset:
        for tag_j in tagset:
            tagtags[tag_i][tag_j] = tagtags[tag_i][tag_j]/(tagcount[tag_i])

    return tagtags

print("gandu")
def Viterbi(words, test): 

    print(test[0])
    tokens, tag = zip(*words)
    tag_count, tag_freq = tag_frequency(tag, words)
    emission_probability = emission_probs(tokens, words, tag_count)
    transition_probability = transition_probs(tag, words, tag_count)

    per_pos = dict.fromkeys(tag_freq.keys(), 0.0)
    per_pos_count = dict.fromkeys(tag_freq.keys(), 0)
    pred = []

    score = 0.0
    viterbi = {}
    for tag_i in tagset:
      viterbi[tag_i] = transition_probability["."][tag_i]

    k = 1

    n = len(test)
    for i in range(n):

        maxi = 0.0
        max_tag = "tag"
        c = False

        viterbi_h = {}
        for tag_i in tagset:
            em = emission_probability[test[i][0]][tag_i]
            val = viterbi[tag_i]*em*100
            viterbi_h[tag_i] = val
            
            if val >= maxi:
                maxi = val
                max_tag = tag_i

            if em != 0:
                c = True

        if not c:
            maxi = 0.0
            max_tag = "tag"
            for tag_i in tagset:
                val = viterbi[tag_i]*100*(1/(1+tag_count[tag_i]))
                viterbi_h[tag_i] = val
                if val >= maxi:
                    maxi = val
                    max_tag = tag_i

        per_pos_count[test[i][1]] = per_pos_count[test[i][1]] + 1
        pred.append(max_tag)

        if max_tag == test[i][1]:
            per_pos[test[i][1]] = per_pos[test[i][1]] + 1.0
            score = score + 1.0


        viterbi_next = dict.fromkeys(tagset, 0)
        for tag_i in tagset:
            for tag_j in tagset:
                viterbi_next[tag_j] = max(viterbi_next[tag_j], viterbi_h[tag_i]*transition_probability[tag_i][tag_j])

        if k%7 == 0:
            k = 0
            viterbi = tag_freq.copy()
        
        else:
            viterbi = viterbi_next

        k = k+1

    for tag_i in tagset:
        per_pos[tag_i] = per_pos[tag_i]/per_pos_count[tag_i]

    return score*100/n, per_pos, pred

def Viterbi_vec(words, test): 
    # print(test[0])
    tokens, tag = zip(*words)
    
    data_w = []
    temp_w = []
    n_w = len(tokens)
    for i in range(n_w):
        temp_w.append(tokens[i])
        if tokens[i] == ".":
            data_w.append(temp_w)
            temp_w = []
    data_w.append(temp_w)
    model_train = gensim.models.Word2Vec(data_w, min_count = 1, vector_size = v_size, window = 5)

    data_t = []
    temp_t = []
    
    n_t = len(test)
    for i in range(n_t):
        temp_t.append(test[i][0])
        if test[i][0] == ".":
            data_t.append(temp_t)
            temp_t = []
    data_t.append(temp_t)
    model_test = gensim.models.Word2Vec(data_t, min_count = 1, vector_size = v_size, window = 5)

    tag_count, tag_freq = tag_frequency(tag, words)
    emission_probability = emission_probs(tokens, words, tag_count)
    transition_probability = transition_probs(tag, words, tag_count)

    per_pos = dict.fromkeys(tag_freq.keys(), 0.0)
    per_pos_count = dict.fromkeys(tag_freq.keys(), 0)
    pred = []

    score = 0.0
    viterbi = {}
    for tag_i in tagset:
      viterbi[tag_i] = transition_probability["."][tag_i]

    k = 1
    n = len(test)
    for i in range(n):

        maxi = 0.0
        max_tag = "tag"
        c = False

        for tag_i in tagset:
            em = emission_probability[test[i][0]][tag_i]

            if em != 0:
                c = True

        if not c:
            lst = list(test[i])
            print("1",lst)
            lst[0] = model_train.wv.similar_by_vector(model_test.wv[test[i][0]], topn=1)[0][0]
            print("2",lst)
            test[i] = tuple(lst)

        viterbi_h = {}
        for tag_i in tagset:
            em = emission_probability[test[i][0]][tag_i]
            val = viterbi[tag_i]*em*100
            viterbi_h[tag_i] = val
            
            if val >= maxi:
                maxi = val
                max_tag = tag_i


        per_pos_count[test[i][1]] = per_pos_count[test[i][1]] + 1
        pred.append(max_tag)

        if max_tag == test[i][1]:
            per_pos[test[i][1]] = per_pos[test[i][1]] + 1.0
            score = score + 1.0


        viterbi_next = dict.fromkeys(tagset, 0)
        for tag_i in tagset:
            for tag_j in tagset:
                viterbi_next[tag_j] = max(viterbi_next[tag_j], viterbi_h[tag_i]*transition_probability[tag_i][tag_j])

        if k%7 == 0:
            k = 0
            viterbi = tag_freq.copy()
        
        else:
            viterbi = viterbi_next

        k = k+1

    for tag_i in tagset:
        per_pos[tag_i] = per_pos[tag_i]/per_pos_count[tag_i]

    return score*100/n, per_pos, pred




def FFNN(words, test):
    tokens, tag = zip(*words)
    tag_count, tag_freq = tag_frequency(tag, words)


    data_w = []
    temp_w = []
    n_w = len(tokens)
    for i in range(n_w):
        temp_w.append(tokens[i])
        if tokens[i] == ".":
            data_w.append(temp_w)
            temp_w = []
    data_w.append(temp_w)
    model_train = gensim.models.Word2Vec(data_w, min_count = 1, vector_size = v_size, window = 5)

    data_t = []
    temp_t = []
    n_t = len(test)
    for i in range(n_t):
        temp_t.append(test[i][0])
        if test[i][0] == ".":
            data_t.append(temp_t)
            temp_t = []
    data_t.append(temp_t)
    model_test = gensim.models.Word2Vec(data_t, min_count = 1, vector_size = v_size, window = 5)

    class net(nn.Module):
        def __init__(self):
            super(net,self).__init__()
            self.l1 = nn.Linear(v_size,200)
            self.l2 = nn.Linear(200,300)
            self.l3 = nn.Linear(300,200)
            self.l4 = nn.Linear(200,12)
            self.relu = nn.ReLU()
            self.activation = nn.ReLU()
        
        def forward(self,x):
            x = self.l1(x) 
            x = self.relu(x)
            x = self.l2(x) 
            x = self.relu(x)
            x = self.l3(x) 
            x = self.relu(x)
            x = self.l4(x) 
            x = self.relu(x)
            output = F.softmax(x)
            return output.double()

    model = net()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
    epochs = 5

    for _ in range(epochs):
        for i in range(n_w):
            y_pred = model(torch.from_numpy(model_train.wv[tokens[i]]))
            cost = criterion(y_pred, tagset_dict[tag[i]])
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        print("lopi")

    per_pos = dict.fromkeys(tag_freq.keys(), 0.0)
    per_pos_count = dict.fromkeys(tag_freq.keys(), 0)
    pred = []
    score = 0.0

    for i in range(n_t):
        predict = model(torch.from_numpy(model_test.wv[test[i][0]]))
        index = torch.argmax(predict)
        per_pos_count[test[i][1]] = per_pos_count[test[i][1]] + 1
        pred.append(tagset[index])
        if tagset[index] == test[i][1]:
            per_pos[test[i][1]] = per_pos[test[i][1]] + 1.0
            score = score + 1.0

    for tag_i in tagset:
        per_pos[tag_i] = per_pos[tag_i]/per_pos_count[tag_i]

    return score*100/n_t, per_pos, pred



score_list_v = [0, 0, 0, 0, 0]
per_pos_v = [{}, {}, {}, {}, {}]
total_pred_v = np.empty([0,0])
score_list_nn = [0, 0, 0, 0, 0]
per_pos_nn = [{}, {}, {}, {}, {}]
total_pred_nn = np.empty([0,0])
score_list = [0, 0, 0, 0, 0]
per_pos = [{}, {}, {}, {}, {}]
total_pred = np.empty([0,0])

tokens, tag = zip(*words)
total_length = len(words)

# Five fold cross validation
for i in range(5):
    x = int(i*total_length/5)
    y = int((i+1)*total_length/5)
    a = [(x.lower(), y) for x, y in words[:x]+words[y:]]
    b = [(key.lower(), val) for key, val in words[x:y]]
    score_list_nn[i], per_pos_nn[i], pred_nn = FFNN(a, b)
    total_pred_nn = np.append(total_pred_nn, np.array(pred_nn))
    # score_list_v[i], per_pos_v[i], pred_v = Viterbi_vec(a, b)
    # total_pred_v = np.append(total_pred_v, np.array(pred_v))
    # score_list[i], per_pos[i], pred = Viterbi(a, b)
    # total_pred = np.append(total_pred, np.array(pred))


# print("---------------------------------------- Viterbi Vector Evaluation ----------------------------------------------------")

# print("Scores list: ", score_list_v, '\t', 'Avg: ', np.average(score_list_v), '\n')
# print("Per POS Accuracy: ")
# print(per_pos_v[0])
# print(per_pos_v[1])
# print(per_pos_v[2])
# print(per_pos_v[3])
# print(per_pos_v[4], '\n')

# var = metrics.precision_recall_fscore_support(tag, total_pred_v, average=None, labels=tagset, zero_division=0)
# print("Precision: ", var[0], '\t', 'Weighted Avg: ', metrics.precision_recall_fscore_support(tag, total_pred_v, average='weighted', labels=tagset, zero_division=0)[0], '\n')
# print("Recall: ", var[1], '\t', 'Weighted Avg: ', metrics.precision_recall_fscore_support(tag, total_pred_v, average='weighted', labels=tagset, zero_division=0)[1], '\n')

# f100 = metrics.fbeta_score(tag, total_pred_v, average=None, beta=1, labels=tagset, zero_division=0)
# print("F1-score: ", f100, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred_v, average='weighted', beta=1, labels=tagset, zero_division=0), '\n')
# f50 = metrics.fbeta_score(tag, total_pred_v, average=None, beta=0.5, labels=tagset, zero_division=0)
# print("F0.5-score: ", f50, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred_v, average='weighted', beta=0.5, labels=tagset, zero_division=0), '\n')
# f200 = metrics.fbeta_score(tag, total_pred_v, average=None, beta=2, labels=tagset, zero_division=0)
# print("F2-score: ", f200, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred_v, average='weighted', beta=0.5, labels=tagset, zero_division=0), '\n')

# confusion_matrix = np.transpose(metrics.confusion_matrix(tag, total_pred_v))
# np.set_printoptions(precision=2)
# confusion_matrix = np.round_(np.transpose(confusion_matrix/np.sum(confusion_matrix,0)),decimals=2)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = tagset)
# fig, ax = plt.subplots(figsize=(10,10))
# cm_display.plot(xticks_rotation='vertical',ax=ax)
# plt.savefig("img_Viterbi_vec.png")

# print("---------------------------------------- FFNN Evaluation ----------------------------------------------------")

# print("Scores list: ", score_list_nn, '\t', 'Avg: ', np.average(score_list_nn), '\n')
# print("Per POS Accuracy: ")
# print(per_pos_nn[0])
# print(per_pos_nn[1])
# print(per_pos_nn[2])
# print(per_pos_nn[3])
# print(per_pos_nn[4], '\n')

# var = metrics.precision_recall_fscore_support(tag, total_pred_nn, average=None, labels=tagset, zero_division=0)
# print("Precision: ", var[0], '\t', 'Weighted Avg: ', metrics.precision_recall_fscore_support(tag, total_pred_nn, average='weighted', labels=tagset, zero_division=0)[0], '\n')
# print("Recall: ", var[1], '\t', 'Weighted Avg: ', metrics.precision_recall_fscore_support(tag, total_pred_nn, average='weighted', labels=tagset, zero_division=0)[1], '\n')

# f100 = metrics.fbeta_score(tag, total_pred_nn, average=None, beta=1, labels=tagset, zero_division=0)
# print("F1-score: ", f100, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred_nn, average='weighted', beta=1, labels=tagset, zero_division=0), '\n')
# f50 = metrics.fbeta_score(tag, total_pred_nn, average=None, beta=0.5, labels=tagset, zero_division=0)
# print("F0.5-score: ", f50, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred_nn, average='weighted', beta=0.5, labels=tagset, zero_division=0), '\n')
# f200 = metrics.fbeta_score(tag, total_pred_nn, average=None, beta=2, labels=tagset, zero_division=0)
# print("F2-score: ", f200, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred_nn, average='weighted', beta=0.5, labels=tagset, zero_division=0), '\n')

# confusion_matrix = np.transpose(metrics.confusion_matrix(tag, total_pred_nn))
# np.set_printoptions(precision=2)
# confusion_matrix = np.round_(np.transpose(confusion_matrix/np.sum(confusion_matrix,0)),decimals=2)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = tagset)
# fig, ax = plt.subplots(figsize=(10,10))
# cm_display.plot(xticks_rotation='vertical',ax=ax)
# plt.savefig("img_FFNN.png")

# print("---------------------------------------- Viterbi Evaluation ----------------------------------------------------")

# print("Scores list: ", score_list, '\t', 'Avg: ', np.average(score_list), '\n')
# print("Per POS Accuracy: ")
# print(per_pos[0])
# print(per_pos[1])
# print(per_pos[2])
# print(per_pos[3])
# print(per_pos[4], '\n')

# var = metrics.precision_recall_fscore_support(tag, total_pred, average=None, labels=tagset, zero_division=0)
# print("Precision: ", var[0], '\t', 'Weighted Avg: ', metrics.precision_recall_fscore_support(tag, total_pred, average='weighted', labels=tagset, zero_division=0)[0], '\n')
# print("Recall: ", var[1], '\t', 'Weighted Avg: ', metrics.precision_recall_fscore_support(tag, total_pred, average='weighted', labels=tagset, zero_division=0)[1], '\n')

# f100 = metrics.fbeta_score(tag, total_pred, average=None, beta=1, labels=tagset, zero_division=0)
# print("F1-score: ", f100, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred, average='weighted', beta=1, labels=tagset, zero_division=0), '\n')
# f50 = metrics.fbeta_score(tag, total_pred, average=None, beta=0.5, labels=tagset, zero_division=0)
# print("F0.5-score: ", f50, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred, average='weighted', beta=0.5, labels=tagset, zero_division=0), '\n')
# f200 = metrics.fbeta_score(tag, total_pred, average=None, beta=2, labels=tagset, zero_division=0)
# print("F2-score: ", f200, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred, average='weighted', beta=0.5, labels=tagset, zero_division=0), '\n')

# confusion_matrix = np.transpose(metrics.confusion_matrix(tag, total_pred))
# np.set_printoptions(precision=2)
# confusion_matrix = np.round_(np.transpose(confusion_matrix/np.sum(confusion_matrix,0)),decimals=2)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = tagset)
# fig, ax = plt.subplots(figsize=(10,10))
# cm_display.plot(xticks_rotation='vertical',ax=ax)
# plt.savefig("img_Viterbi.png")


# etime = time.time()

# print(etime-stime)

