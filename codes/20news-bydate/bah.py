import collections
import json
import glob
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

files = glob.glob("./20news-bydate/20news-bydate-train/**/*", recursive=True)
files = np.array(files)
files = files[20:]
files = sorted(files)
#write a list into a txt file
with open('hah.txt','w') as fp:
    for elem in files:
        fp.write(elem+'\n')


print('finished')

text = []
file_names = []
for file in files:
        # print('extracting text from', file)
        with open(file, 'r',errors='ignore') as f:
            text.append(f.read())
            ha = file.split("/")
            file_names.append(ha[3]+'/'+ha[4])

wow = np.array(text)
print('Reading text done')
print('fitting vectorizer...')
vectorizer = CountVectorizer()
doc_vectors = vectorizer.fit_transform(wow)
vocab = vectorizer.get_feature_names()

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
    with open(file_name, 'r') as f:
        text = f.read()
    words = []  # contains indices wrt to vocab
    vectorizer = CountVectorizer(vocabulary=vocab)
    doc_vec = vectorizer.fit_transform([text])
    for i in range(len(doc_vec)):
        if bbow_vecs[i]!=0:
            words.append(i)

    return words

words_in_test = words_in_testfile('./20news-bydate/20news-bydate-test/alt.atheism/',vocab)



# def get_list(label):
#     if label == 0 :



# #write a laplace smoothing function
# def laplace_smoothing(word_index,label):
    



#-------------------------------------------------------------------------