import collections
import json
import glob
from pydoc import doc
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import math
import time

start_time = time.time()
files = glob.glob("./20news-bydate/20news-bydate-train/**/*", recursive=True)
files = np.array(files)
files = files[20:]
files = sorted(files)


#----------------------------division of files into lists---------------------------

conven = []

alt_atheism = files[0:480]
conven.append(alt_atheism)
comp_graphics = files[480:1064]
conven.append(comp_graphics)
comp_os_ms_windows_misc = files[1064:1655]
conven.append(comp_os_ms_windows_misc)
comp_sys_ibm_pc_hardware = files[1655:2245]
conven.append(comp_sys_ibm_pc_hardware)
comp_sys_mac_hardware = files[2245:2823]
conven.append(comp_sys_mac_hardware)
comp_windows_x = files[2823:3416]
conven.append(comp_windows_x)
misc_forsale = files[3416:4001]
conven.append(misc_forsale)
rec_autos = files[4001:4595]
conven.append(rec_autos)
rec_motorcycles = files[4595:5193]
conven.append(rec_motorcycles)
rec_sport_baseball = files[5193:5790]
conven.append(rec_sport_baseball)
rec_sport_hockey = files[5790:6390]
conven.append(rec_sport_hockey)
sci_crypt = files[6390:6985]
conven.append(sci_crypt)
sci_electronics = files[6985:7576]
conven.append(sci_electronics)
sci_med = files[7576:8170]
conven.append(sci_med)
sci_space = files[8170:8763]
conven.append(sci_space)
soc_religion_christian = files[8763:9362]
conven.append(soc_religion_christian)
talk_politics_guns = files[9362:9908]
conven.append(talk_politics_guns)
talk_politics_mideast = files[9908:10472]
conven.append(talk_politics_mideast)
talk_politics_misc = files[10472:10937]
conven.append(talk_politics_misc)
talk_religion_misc = files[10937:11314]
conven.append(talk_religion_misc)

#-------------------------------------------------------------------------------------


text = []
file_names = []
for file in files:
        # print('extracting text from', file)
        with open(file, 'r',errors='ignore') as f:
            text.append(f.read())
            ha = file.split("/")
            file_names.append(ha[3]+'/'+ha[4])

for i in range(len(files)):
    ha = files[i].split('/')
    files[i] = ha[3] + '/' + ha[4]

wow = np.array(text)
print('Reading text done')
print('fitting vectorizer...')
vectorizer = CountVectorizer()
doc_vectors = vectorizer.fit_transform(wow)
vocab = vectorizer.get_feature_names_out()

# Read the labels from the file names

labels = {}


bbow_vecs = collections.defaultdict(dict)
with open('bbow_vecs.json', 'r') as fp:
    bbow_vecs = json.load(fp)

print('processed', len(bbow_vecs), 'documents')

cbow_vecs = collections.defaultdict(dict)
with open('cbow_vecs.json', 'r') as fp:
    cbow_vecs = json.load(fp)

print('processed', len(cbow_vecs), 'documents')

#---------------------------testfiles-----------------------------------




def words_in_testfile(file_name, vocab):
    text = []
    with open(file_name, 'r',errors='ignore') as f:
        text.append(f.read())
    words = []  # contains indices wrt to vocab
    vectorizer = CountVectorizer(vocabulary=vocab)
    doc_vec = vectorizer.fit_transform(text)
    doc_veci = doc_vec[0].toarray()
    doc_veci[doc_veci > 1] = 1
    
    for i in range(len(vocab)):
        if doc_veci[0][i]!=0:
            words.append(i)

    return words


# test_files = glob.glob("./20news-bydate/20news-bydate-test/**/*", recursive=True)
# words_in_test = []
# for i in range(20,len(test_files)):
#     words_in_test.append(words_in_testfile(test_files[i],vocab))
# print('test files processing into lists of words done')

a = input("Enter the path of the test file : ")
words_in_test = words_in_testfile(a,vocab)
# print(words_in_test)
print('test file processing into lists of words done')


def pruning(labels):
    wow = []
    for elem in labels :
        elem = elem.split('/')
        wow.append(elem[3]+'/'+elem[4])
    return wow


util1_bbow = collections.defaultdict(dict)
util2_cbow = collections.defaultdict(dict)
bbow_probs = {}
cbow_probs = {}
hits = 0


# #write a laplace smoothing function
def laplace_smoothing(word_index,label,k,type):
    a=0
    
    count = 0
    for elem in conven[label]:
        ha = elem.split('/')
        elem = ha[3] + '/' + ha[4]

        if type == 'bbow':
            count += np.sum(np.array(list((bbow_vecs[elem].values()))))
            if(bbow_vecs[elem].get(str(word_index)) != None):
                a += bbow_vecs[elem][str(word_index)]

            # for j in bbow_vecs[elem].keys():
                
            #     count+=bbow_vecs[elem][j]
            #     if int(j) == word_index:
            #         a+=bbow_vecs[elem][j]
            # if j not in bbow_vecs[elem].keys():
            #     a+=0
    

        else:
            count += np.sum(np.array(list(cbow_vecs[elem].values())))
            if(cbow_vecs[elem].get(str(word_index)) != None):
                a += cbow_vecs[elem][str(word_index)]
            # for j in cbow_vecs[elem].keys():
                
            #     count+=cbow_vecs[elem][j]
            #     if int(j) == word_index:
            #         a+=cbow_vecs[elem][j]
            # if j not in cbow_vecs[elem].keys():
            #     a+=0
            

    # print('laplace smoothing done')
    return math.log((a+k)/(count+k*len(vocab)))

# #write a function to calculate the probability of a document belonging to a class
def prob_of_doc(label,words_in_test,type):
    global hits
    if(type == "bbow"):
        prob = 0
        for i in words_in_test:
            if bbow_probs.get(i) == None or bbow_probs.get(i).get(label) == None :
                pr = laplace_smoothing(i,label,1,type)
                prob+=pr
                if(bbow_probs.get(i) == None): bbow_probs[i] = {}
                bbow_probs[i][label] = pr
            else:
                hits+=1
                prob+=bbow_probs[i][label]
        # print('prob_of_doc done')
        return prob
    else:
        prob = 0
        
        for i in words_in_test:
            if cbow_probs.get(i) == None or cbow_probs.get(i).get(label) == None :
                pr = laplace_smoothing(i,label,1,type)
                prob+=pr
                if(cbow_probs.get(i) == None): cbow_probs[i] = {}
                cbow_probs[i][label] = pr
            else: 
                hits+=1
                prob+=cbow_probs[i][label]
        # print('prob_of_doc done')
        return prob


def predict(words_in_test,type):
    global hits
    hits=0
    prob = np.zeros(20)
    for i in range(20):
        prob[i] = prob_of_doc(i,words_in_test,type) + math.log(len(conven[i])/11314)
    print('prediction - done')
    print("hits ",hits)
    return np.argmax(prob)   

def accuracy(type):
    count = 0
    i = 0
    for i in range(0,len(conven[0])):
        if predict(words_in_test[i],type) == 0:
            count+=1
        i+=1
    for i in range(i,i+len(conven[1])):
        if predict(words_in_test[i],type) == 1:
            count+=1
        i+=1
    for i in range(i,i+len(conven[2])):
        if predict(words_in_test[i],type) == 2:
            count+=1
        i+=1
    for i in range(i,i+len(conven[3])):
        if predict(words_in_test[i],type) == 3:
            count+=1
        i+=1
    for i in range(i,i+len(conven[4])):
        if predict(words_in_test[i],type) == 4:
            count+=1
        i+=1
    for i in range(i,i+len(conven[5])):
        if predict(words_in_test[i],type) == 5:
            count+=1
        i+=1
    for i in range(i,i+len(conven[6])):
        if predict(words_in_test[i],type) == 6:
            count+=1
        i+=1
    for i in range(i,i+len(conven[7])):
        if predict(words_in_test[i],type) == 7:
            count+=1
        i+=1
    for i in range(i,i+len(conven[8])):
        if predict(words_in_test[i],type) == 8:
            count+=1
        i+=1
    for i in range(i,i+len(conven[9])):
        if predict(words_in_test[i],type) == 9:
            count+=1
        i+=1
    for i in range(i,i+len(conven[10])):
        if predict(words_in_test[i],type) == 10:
            count+=1
        i+=1
    for i in range(i,i+len(conven[11])):
        if predict(words_in_test[i],type) == 11:
            count+=1
        i+=1
    for i in range(i,i+len(conven[12])):
        if predict(words_in_test[i],type) == 12:
            count+=1
        i+=1
    for i in range(i,i+len(conven[13])):
        if predict(words_in_test[i],type) == 13:
            count+=1
        i+=1
    for i in range(i,i+len(conven[14])):
        if predict(words_in_test[i],type) == 14:
            count+=1
        i+=1
    for i in range(i,i+len(conven[15])):
        if predict(words_in_test[i],type) == 15:
            count+=1
        i+=1
    for i in range(i,i+len(conven[16])):
        if predict(words_in_test[i],type) == 16:
            count+=1
        i+=1
    for i in range(i,i+len(conven[17])):
        if predict(words_in_test[i],type) == 17:
            count+=1
        i+=1
    for i in range(i,i+len(conven[18])):
        if predict(words_in_test[i],type) == 18:
            count+=1
        i+=1
    for i in range(i,i+len(conven[19])):
        if predict(words_in_test[i],type) == 19:
            count+=1
        i+=1
    print('accuracy done')
    return count/len(words_in_test)


label=[]
label.append("alt.atheism")
label.append("comp.graphics")
label.append("comp.os.ms-windows.misc")
label.append("comp.sys.ibm.pc.hardware")
label.append("comp.sys.mac.hardware")
label.append("comp.windows.x")
label.append("misc.forsale")
label.append("rec.autos")
label.append("rec.motorcycles")
label.append("rec.sport.baseball")
label.append("rec.sport.hockey")
label.append("sci.crypt")
label.append("sci.electronics")
label.append("sci.med")
label.append("sci.space")
label.append("soc.religion.christian")
label.append("talk.politics.guns")
label.append("talk.politics.mideast")
label.append("talk.politics.misc")
label.append("talk.religion.misc")
print("cbow -- model prediction")
print(label[predict(words_in_test,"cbow")])
print('Elapsed time: ', time.time() - start_time)
start_time = time.time()
print("bbow -- model prediction")
print(label[predict(words_in_test,"bbow")])
print('Elapsed time: ', time.time() - start_time)
# print(accuracy('bbow'))  

# print('Elapsed time: ', time.time() - start_time)
    



#-------------------------------------------------------------------------