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
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# nltk.download('brown')
# nltk.download('universal_tagset')
# nltk.download('punkt')

stime = time.time()

tagset = ['NOUN', 'VERB', '.', 'ADP', 'DET', 'ADJ', 'ADV', 'PRON', 'CONJ', 'PRT', 'NUM', 'X']
identity = torch.eye(12, dtype=torch.double)
tagset_dict = {'NOUN':identity[0,:], 'VERB':identity[1,:], '.':identity[2,:], 'ADP':identity[3,:], 'DET':identity[4,:], 'ADJ':identity[5,:], 'ADV':identity[6,:], 'PRON':identity[7,:], 'CONJ':identity[8,:], 'PRT':identity[9,:], 'NUM':identity[10,:], 'X':identity[11,:]}

words = list(brown.tagged_words(tagset='universal'))

v_size = 128 # Word Vector Size

def tag_frequency(tag, words):
    total = len(words)
    tagcount = Counter(tag) 
    tagfreq = {}
    tagcount_v= torch.tensor((12,1),dtype=torch.int64)
    # tagfreq = torch.tensor((12,1),dtype=torch.float64)

    for i in tagcount.keys():
        tagcount_v = tagcount[i]
        tagfreq[i] = tagcount[i]/total
    # tagfreq = tagcount_v/total
    return tagcount, tagfreq

def emission_probs(tokens, words, tagcount):
    tokenTags = defaultdict(Counter)
    # tT = torch.tensor(())
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

def FFNN(words, test,data_w, data_t, taglist, taglist_t):
    tokens, tag = zip(*words)
    tag_count, tag_freq = tag_frequency(tag, words)
    t_tokens,t_tag = zip(*test)
    
    model_train = gensim.models.Word2Vec(data_w, min_count = 1, vector_size = v_size, window = 5, workers = 4)
    n_t = len(test)

    model_test = gensim.models.Word2Vec(data_t, min_count = 1, vector_size = v_size, window = 5, workers = 4)

    class net(nn.Module):
        def __init__(self):
            super(net,self).__init__()
            self.l1 = nn.Linear(v_size,256)
            # self.l2 = nn.Linear(256,512)
            self.l3 = nn.Linear(256,256)
            self.l4 = nn.Linear(256,12)
            self.relu = nn.ReLU()
            self.activation = nn.ReLU()
        
        def forward(self,x):
            x = self.l1(x) 
            x = self.relu(x)
            # x = self.l2(x) 
            # x = self.relu(x)
            x = self.l3(x) 
            x = self.relu(x)
            x = self.l4(x) 
            x = self.relu(x)
            output = x
            return output.double()

    model = net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 5
    model.zero_grad()
    n_train = len(tokens)//1000

    for epo in range(epochs):
        model.train()
        for i in range(1000):
          y_pred = model(torch.from_numpy(model_train.wv[tokens[i*n_train:(i+1)*n_train]]))
          cost = criterion(y_pred, taglist[i*n_train:(i+1)*n_train])
          optimizer.zero_grad()
          cost.backward()
          optimizer.step()
        y_pred = model(torch.from_numpy(model_train.wv[tokens]))
        cost = criterion(y_pred, taglist)
        print("cost", cost, "epoch", epo)


    y_pred = model(torch.from_numpy(model_train.wv[tokens]))
    cost = criterion(y_pred, taglist)
    rando, y_pred_tags = torch.max(y_pred, dim = 1) 
    y_pred_tags = torch.eye(12)[y_pred_tags,:]
    
    correct_pred = ((y_pred_tags == taglist)*(y_pred_tags == 1)).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    print("Accuracy : ", acc)
    print("Cost : ", cost)


    per_pos = torch.zeros((12,1),dtype=torch.float64)
    per_pos_count = torch.zeros((12,1),dtype=torch.int64)
    pred = []
    score = 0.0

    im_tired = []

    for tok in t_tokens:
      try:
        model_train.wv[tok]
      except:
        im_tired.append(model_test.wv[tok])
      else:
        im_tired.append(model_train.wv[tok])

    print("opop")
    i_am_tired = np.array(im_tired)
    predict = model(torch.from_numpy(i_am_tired))

    index = torch.argmax(predict,1)
    for ind in index:
      pred.append(tagset[ind])

    per_pos_count = torch.count_nonzero(taglist_t,dim=0)
    for i in range(n_t):
      per_pos[index[i]] += taglist_t[i,index[i]]
      score = score + taglist_t[i,index[i]]
      
    per_pos = per_pos/per_pos_count

    # return pred
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
taglist = torch.empty((total_length,12),dtype=torch.float64)
for i in range(len(words)):
  taglist[i] = tagset_dict[words[i][1]].unsqueeze(0)

data_t = [[],[],[],[],[]]
data_w = [[],[],[],[],[]]

# kFold_f = KFold(n_splits=5, shuffle=True, random_state=1)
# tSents_f = np.array(words, dtype=object)
# yPreds_f = []
# yTrues_f = []

# for fold, (train, test) in enumerate(kFold_f.split(words)):
#     trainSent = words[train]
#     testSent = words[test]
    
#     tokens_w, tag_w = zip(*trainSent)
#     tokens_t, tag_t = zip(*trainSent)
#     data_w = []
#     temp_w = []
#     data_t = []
#     temp_t = []
# #     taglist_w = []
# #     taglist_t = []

#     taglist_w = torch.empty((len(trainSent),12),dtype=torch.float64)
#     taglist_t = torch.empty((len(testSent),12),dtype=torch.float64)
    
#     for i in range(len(trainSent)):
#       taglist_w[i] = tagset_dict[trainSent[i][1]].unsqueeze(0)
    
#     for i in range(len(testSent)):
#       taglist_t[i] = tagset_dict[testSent[i][1]].unsqueeze(0)
    
#     for j in range(len(tokens_w)):
#         temp_w.append(tokens_w[j])
#         if tokens_w[j] == ".":
#             data_w.append(temp_w)
#             temp_w = []
#     data_w.append(temp_w)
    
#     for j in range(len(tokens_t)):
#         temp_t.append(tokens_t[j])
#         if tokens_t[j] == ".":
#             data_t.append(temp_t)
#             temp_t = []
#     data_t.append(temp_t)
    
#     print("yp")


#     yPred, yTrue = [],[]
    
#     yPred = FFNN(trainSent, testSent, data_w, data_t, taglist_w, taglist_t)
#     print("ffnn finished")
    
#     for w in (testSent):
#         yTrue.append(w[1])
    
#     yPreds_f.append(np.array(yPred))
#     yTrues_f.append(np.array(yTrue))
    
#     acc = np.sum(yTrues_f[-1] == yPreds_f[-1])/len(yTrues_f[-1])
#     print(f'Fold {fold + 1} Accuracy : {acc}')

for i in range(5):
    temp_w = []
    temp_t=[]
    x = int(i*total_length/5)
    y = int((i+1)*total_length/5)
    tokens_w = [l.lower() for l in tokens[:x]+tokens[y:]]
    tokens_t = [l.lower() for l in tokens[x:y]]
    for j in range(len(tokens_w)):
        temp_w.append(tokens_w[j])
        if tokens_w[j] == ".":
            data_w[i].append(temp_w)
            temp_w = []
    data_w[i].append(temp_w)
    for j in range(len(tokens_t)):
        temp_t.append(tokens_t[j])
        if tokens_t[j] == ".":
            data_t[i].append(temp_t)
            temp_t = []
    data_t[i].append(temp_t)



stime = time.time()
k = int(4*total_length/5)
a = [(x.lower(), y) for x, y in words[:k]]
b = [(key.lower(), val) for key, val in words[k:]]
yTrues = tag[x:y]
score_list_nn[i], per_pos_nn[i], pred_nn = FFNN(a, b,data_w[i],data_t[i],torch.cat((taglist[:x,:],taglist[y:,:])),taglist[x:y])
total_pred_nn = np.append(total_pred_nn, np.array(pred_nn))
etime = time.time()
print("Time Taken : ", etime-stime)

def getScores(yTrue, yPred, tags): # tags is tag list 
    right, wrong = {}, {}
    for tag in tags:
        right[tag] = 0
        wrong[tag] = 0
        
    for tag, pred in zip(yTrue, yPred):
        if tag in right and tag == pred:
            right[tag] += 1
        elif tag in wrong:
            wrong[tag] += 1
            
    scores = []
    total = len(yTrue)
    for tag in tags:
        cur = np.array([right[tag], wrong[tag]])
        scores.append(cur / (right[tag] + wrong[tag]))
    return np.array(scores)

def plotConfusionMatrix(classes, mat, normalize=True, cmap=plt.cm.Blues):
    cm = np.copy(mat)
    title = 'Confusion Matrix (without normalization)'
    if normalize:
        cm = cm.astype('float') / np.sum(cm, axis=1, keepdims=True)
        title = title.replace('without', 'with')
    plt.clf()    
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title(title, y=-0.06, fontsize=22)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = np.max(cm) / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if (cm[i, j] > thresh) else "black"
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color=color)
    plt.ylabel('True label',fontsize=22)
    plt.xlabel('Predicted label', fontsize=22)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', bbox_inches="tight", transparent=True)

def plotCLFReport(classes, plotMat, support, cmap=plt.cm.Blues):
    title = 'Classification Report'
    xticklabels = ['Precision', 'Recall', 'F1-score', 'F0.5-Score','F2-Score']
    yticklabels = ['{0} ({1})'.format(classes[idx], sup) for idx, sup in enumerate(support)]
    plt.clf()
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title(title, y=-0.06, fontsize=22)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.colorbar()
    plt.xticks(np.arange(5), xticklabels, rotation=0)
    plt.yticks(np.arange(len(classes)), yticklabels)

    thresh = np.max(plotMat) / 2
    for i in range(plotMat.shape[0]):
        for j in range(plotMat.shape[1]):
            color = "white" if (plotMat[i, j] > thresh) else "black"
            plt.text(j, i, format(plotMat[i, j], '.2f'), horizontalalignment="center", color=color, fontsize=14)

    plt.xlabel('Metrics',fontsize=22)
    plt.ylabel('Classes',fontsize=22)
    plt.tight_layout()
    plt.savefig('classification_report.png')

def plotTagScores(classes, scores, normalize=True):
    plt.clf()
    width = 0.45
    fig, ax = plt.subplots(figsize=(20,10))
    ax.xaxis.set_tick_params(labelsize=18, rotation=25)
    ax.yaxis.set_tick_params(labelsize=18)
    range_bar1 = np.arange(len(classes))
    rects1 = ax.bar(range_bar1, tuple(scores[:, 0]), width, color='b')
    rects2 = ax.bar(range_bar1 + width, tuple(scores[:, 1]), width, color='r')

    ax.set_ylabel('Scores',fontsize=22)
    ax.set_title('Tag scores', fontsize=22)
    ax.set_xticks(range_bar1 + width / 2)
    ax.set_xticklabels(classes)

    ax.legend((rects1[0], rects2[0]), ('Correct', 'Wrong'), fontsize=20)
    plt.legend()
    plt.savefig('tag_scores.png')
    plt.show()

def getReport(y_true, y_pred, classes):
    clf_report = classification_report(y_true, y_pred, labels=classes, zero_division=0)
    clf_report = clf_report.replace('\n\n', '\n')
    clf_report = clf_report.replace('macro avg', 'macro_avg')
    clf_report = clf_report.replace('micro avg', 'micro_avg')
    clf_report = clf_report.replace('weighted avg', 'weighted_avg')
    clf_report = clf_report.replace(' / ', '/')
    lines = clf_report.split('\n')
    
    class_names, plotMat, support = [], [], []
    for line in lines[1:]:
        t = line.strip().split()
        if len(t) < 2:
            continue
        v = [float(x) for x in t[1: len(t) - 1]]
        
        #print(v)
        
        
        if len(v) == 1 : v = v * 3
        x,y = v[0],v[1]
        # f 0.5
        if x!=0  and y!=0:
            v += [round((1.25*x*y)/((0.25*x) + y),2)]
            # f 2
            v += [round((5*x*y)/((4*x) + y),2)]
            #print(v)
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)
    plotMat = np.array(plotMat)
    support = np.array(support)
    
    return class_names, plotMat, support


tags = tagset
class_names = None
report = None
support = None
cm, scores = None, None
cnt = 0
yPreds = pred_nn
# yTrues = 
# print(len(yPreds))
# print(len(yTrues))
# print(yPreds[0])
# print(yTrues[0])
yPred = yPreds
yTrue = yTrues
# for yTrue, yPred in zip(yTrues, yPreds):

class_names, report_, support_ = getReport(yTrue, yPred, tags)
#print(report_)
cm_ = confusion_matrix(yTrue, yPred, labels=tags)
scores_ = getScores(yTrue, yPred, tags)

# if report is None : report = np.zeros_like(report_, dtype=np.float64)
# report += report_

# if support is None : support = np.zeros_like(support_, dtype=np.float64)
# support += support_

# if cm is None : cm = np.zeros_like(cm_, dtype=np.float64)
# cm += cm_

# if scores is None : scores = np.zeros_like(scores_, dtype=np.float64)
# scores += scores_

# cnt += 1
    
# report /= cnt
# support /= cnt
# cm /= cnt
# scores /= cnt

plotCLFReport(class_names, report_, support_)
plotConfusionMatrix(tags, cm_)
plotTagScores(tags, scores_)
'''score_list_v[i], per_pos_v[i], pred_v = Viterbi_vec(a, b,data_w[i],data_t[i])
total_pred_v = np.append(total_pred_v, np.array(pred_v))
score_list[i], per_pos[i], pred = Viterbi(a, b)
total_pred = np.append(total_pred, np.array(pred))'''

# confusion_matrix = np.transpose(metrics.confusion_matrix(tag, total_pred_nn))
# np.set_printoptions(precision=2)
# confusion_matrix = np.round_(np.transpose(confusion_matrix/np.sum(confusion_matrix,0)),decimals=2)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = tagset)
# fig, ax = plt.subplots(figsize=(10,10))
# cm_display.plot(xticks_rotation='vertical',ax=ax)
# plt.savefig("img_FFNN.png")