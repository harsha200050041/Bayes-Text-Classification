import collections
import math
from time import sleep
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
import glob
import torch
import json
import time


#files     --- names of all files not acc to order
#counts    --- counts of all files acc to order
#tags      --- tags of all files acc to order (eg alt.atheism)

#--------------------------------------------test files--------------------------------------------------

test_files = glob.glob('20news-bydate/20news-bydate-train/*', recursive=True)

#--------------------------------------------------------------------------------------------------------
    

# Read all the text from the files into a nparray----------------------------------------------------------------------
start_time = time.time()
print('Reading files...')
li = ["./20news-bydate/20news-bydate-train/alt.atheism/*","./20news-bydate/20news-bydate-train/comp.graphics/*","./20news-bydate/20news-bydate-train/comp.os.ms-windows.misc/*","./20news-bydate/20news-bydate-train/comp.sys.ibm.pc.hardware/*","./20news-bydate/20news-bydate-train/comp.sys.mac.hardware/*","./20news-bydate/20news-bydate-train/comp.windows.x/*","./20news-bydate/20news-bydate-train/misc.forsale/*","./20news-bydate/20news-bydate-train/rec.autos/*","./20news-bydate/20news-bydate-train/rec.motorcycles/*","./20news-bydate/20news-bydate-train/rec.sport.baseball/*","./20news-bydate/20news-bydate-train/rec.sport.hockey/*","./20news-bydate/20news-bydate-train/sci.crypt/*","./20news-bydate/20news-bydate-train/sci.electronics/*","./20news-bydate/20news-bydate-train/sci.med/*","./20news-bydate/20news-bydate-train/sci.space/*","./20news-bydate/20news-bydate-train/soc.religion.christian/*","./20news-bydate/20news-bydate-train/talk.politics.guns/*","./20news-bydate/20news-bydate-train/talk.politics.mideast/*","./20news-bydate/20news-bydate-train/talk.politics.misc/*","./20news-bydate/20news-bydate-train/talk.religion.misc/*"]
counts=[]
for elem in li:
    files = glob.glob(elem, recursive=True)
    counts.append(files.__len__())
    files = np.array(files)


print('Reading files done')
files = glob.glob("./20news-bydate/20news-bydate-train/**/*", recursive=True)
files = np.array(files)
#---------------------------------------------------------------------------------------------------------------------------

# Read the labels from the file names
print('Reading labels...')
labels = {}
count =0
#--------------------------------labels are sorted according to the order of files------------------------------------------
# for file in files:

#     if count>=20:
#         temp = file.split('/')
        
#         file = temp[3]+'/'+temp[4]
#         if(file.__contains__("alt.atheism")):
#             labels[file] = 0  
#         elif(file.__contains__("comp.graphics")):
#             labels[file] = 1
#         elif(file.__contains__("comp.os.ms-windows.misc")):
#             labels[file] = 2
#         elif(file.__contains__("comp.sys.ibm.pc.hardware")):
#             labels[file] = 3
#         elif(file.__contains__("comp.sys.mac.hardware")):
#             labels[file] = 4
#         elif(file.__contains__("comp.windows.x")):
#             labels[file] = 5
#         elif(file.__contains__("misc.forsale")):
#             labels[file] = 6
#         elif(file.__contains__("rec.autos")):
#             labels[file] = 7
#         elif(file.__contains__("rec.motorcycles")):
#             labels[file] = 8
#         elif(file.__contains__("rec.sport.baseball")):
#             labels[file] = 9
#         elif(file.__contains__("rec.sport.hockey")):
#             labels[file] = 10
#         elif(file.__contains__("sci.crypt")):
#             labels[file] = 11
#         elif(file.__contains__("sci.electronics")):
#             labels[file] = 12
#         elif(file.__contains__("sci.med")):
#             labels[file] = 13
#         elif(file.__contains__("sci.space")):
#             labels[file] = 14
#         elif(file.__contains__("soc.religion.christian")):
#             labels[file] = 15
#         elif(file.__contains__("talk.politics.guns")):
#             labels[file] = 16
#         elif(file.__contains__("talk.politics.mideast")):
#             labels[file] = 17
#         elif(file.__contains__("talk.politics.misc")):
#             labels[file] = 18
#         elif(file.__contains__("talk.religion.misc")):
#             labels[file] = 19
#     else:
#         count+=1
# print('Reading labels done')

text=[]
tags = files[0:20]
tags = sorted(tags)
for i in range(20):
    baaaa = tags[i].split('/')
    tags[i]= baaaa[3]

class_count = {}
ind = 0
for elem in tags:
    class_count[elem] = counts[ind]
    ind+=1

files = files[20:]
files = sorted(files)
count=0
class_file_lists = collections.defaultdict(list)
#creating a list of files of a particular class
cnt = 0
for elem in tags:
    for file in files:
        if file.__contains__(elem):
            haa = file.split('/')
            class_file_lists[elem].append(haa[3]+'/'+haa[4])



file_names = []
print('Reading text...')
for file in files:
        # print('extracting text from', file)
        with open(file, 'r',errors='ignore') as f:
            count+=1
            text.append(f.read())
            ha = file.split("/")
            file_names.append(ha[3]+'/'+ha[4])

wow = np.array(text)
print('Reading text done')
print('fitting vectorizer...')
vectorizer = CountVectorizer()
doc_vectors = vectorizer.fit_transform(wow)
vocab = vectorizer.get_feature_names_out()


del vectorizer
del wow
del files 


x = doc_vectors.toarray()
print('fitting vectorizer done')
print('converting sparse matrix into dictionary of dictionaries...')
cbow_vecs = collections.defaultdict(dict)
for i in range(0,len(x)):
    for j in range(0,len(x[i])):
        if(x[i][j]!=0):
            cbow_vecs[file_names[i]][j] = int(x[i][j])

# #load cbow_vecs
# with open('cbow_vecs.json', 'r') as fp:
#     cbow_vecs = json.load(fp)

print('cbow processing done')
with open('cbow_vecs.json', 'w') as fp:
    json.dump(cbow_vecs, fp,indent = 0)
print('cbow_vecs.json saved')
del cbow_vecs
x[x>1] = 1
bbow_vecs = collections.defaultdict(dict)
count =0
print('bbow processing...')
for i in range(0, len(x)):
    for elem in range(0,len(x[i])):
        if x[i][elem]!=0:
            bbow_vecs[file_names[i]][elem]=int(x[i][elem])

# with open('bbow_vecs.json', 'r') as fp:
#     bbow_vecs = json.load(fp)

print('processed', len(bbow_vecs), 'documents')

#convert a dictionary into a json file
with open('bbow_vecs.json', 'w') as fp:
    json.dump(bbow_vecs,fp,indent = 0)
print('bbow_vecs.json saved')

print('elapsed time:', time.time()-start_time)

#index of a word in vocab

# def word_to_index(word):
#     vocab.index(word)

# print(type(bbow_vecs))

#write a laplace smoothing function
# def laplace_smoothing(word_index,k,label,type):
#     len_vocab = len(x[0])    
#     a = 0
#     for elem in class_file_lists[label]:
#         if word_index in bbow_vecs[elem]:
#             if type == 'cbow':
#                 a+=cbow_vecs[elem][word_index]
#             else:
#                 a+=bbow_vecs[elem][word_index]
#     return math.log((a+k)/(class_count[label]+k*len_vocab))

# #write a function to calculate the probclass
# def prob_doc_class(doc,label):
#     prob = 0
#     for word in cbow_vecs[doc]:
#         prob+=laplace_smoothing(word,1,label,'cbow')
#     return prob

# #write a function to calculate the probability of a document belonging to a class
# def prob_doc_class_bbow(doc,label):
#     prob = 0
#     print(doc)
#     print(bbow_vecs[doc])
#     for word in map(int,bbow_vecs[doc].keys()):
#         prob+=laplace_smoothing(word,1,label,'bbow')
#     return prob

# #calculate the probability of a test document belonging to a class

# def predict_class(doc):

#     prob = []
#     for elem in tags:
#         prob.append(prob_doc_class(doc,elem))
#     return tags[np.argmax(prob)]

# def predict_class_bbow(doc):
    
#         prob = []
#         for elem in tags:
#             prob.append(prob_doc_class_bbow(doc,elem))
#         print(prob)
#         return tags[np.argmax(prob)]


# #write a function to calculate the accuracy of the model
# # def accuracy():
# #     count = 0
# #     for elem in test_files:
# #         if predict_class(elem) == elem.split('/')[3]+'/'+elem.split('/')[4]:
# #             count+=1
# #     return count/len(test_files)

# # def accuracy_bbow():
# #     count = 0
# #     for elem in test_files:
# #         if predict_class_bbow(elem) == elem.split('/')[3]+'/'+elem.split('/')[4]:
# #             count+=1
# #     return count/len(test_files)

# #read test files
# with open('./20news-bydate/20news-bydate-test/alt.atheism/53068','r') as f:
#     test = f.read()

# #predict the class of the test document
# print(predict_class_bbow('comp.graphics/37261'))







