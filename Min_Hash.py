import sklearn.feature_extraction.text as textextract
from bs4 import BeautifulSoup
import random
import numpy as np
import time

#Change here for the size of the shingles
ngram_size = 3
#Change here to increase K
num_hash_fun = 16

#For 3 grams
#prime number for 1000 documents
prime = 67103
#prime number for 5000 documents
#prime = 299171
#prime number for 10000 documents
#prime = 586787
#prime number for 20000 documents
#prime = 1169789
#Countvectorizer used for tokenizing the documents. It will store the results of the documents in a sparse matrix to save space.
vect = textextract.CountVectorizer(lowercase=True, stop_words='english', ngram_range=(ngram_size, ngram_size), analyzer=u'word', binary=True, dtype=bool)
start = time.time()
i = 100
bodyText = []
#Change here to increase the number of documents considered, upto 122 for all documents. Warning: it will take really long time for all documents. We ran for 5 hrs and the code did not finish yet.
while i < 101:
    xml_filename = 'reut2-0' + str(i)[1:] + '.sgm'
    i += 1
    xml_file = open(xml_filename, 'r')
    soup = BeautifulSoup(xml_file, 'html.parser')
    bodyTag = soup.findAll('body')

    for body in bodyTag:
        if body.getText():
            bodyText.append(body.getText())

transformedVect = vect.fit_transform(bodyText)

jaccard = [x[:] for x in [[0] * transformedVect.shape[0]] * transformedVect.shape[0]]

#Calculating jaccard similarity using the formula n11/n00, where n11 is the intersection of 1's and n00 is the union of 1's.
print 'Started calculating jaccard similarity'
for i in range(transformedVect.shape[0]):
    for j in range(i+1,transformedVect.shape[0]):
        x = transformedVect[i].nonzero()[1]
        y = transformedVect[j].nonzero()[1]
        n11 = len(set(x).intersection(set(y)))
        n00 = len(x) + len(y) - n11
        jaccard[i][j] = n11/float(n00)

print 'Time taken for Jaccard :', time.time() - start

maxShingleID = transformedVect.shape[1]
total_no_docs = transformedVect.shape[0]


def pickRandomCoeffs(k):
    randList = random.sample(range(0, 20000), k)

    return randList


coeff_a = pickRandomCoeffs(num_hash_fun)
coeff_b = pickRandomCoeffs(num_hash_fun)

corpus_signatures = []
print 'Started creating minhash'
for doc_id in xrange(total_no_docs):

    doc_signature = []

    for i in range(0, num_hash_fun):
        min_Hash = total_no_docs + 1

        for index in transformedVect[doc_id].nonzero()[1]:
            hash_value = (((coeff_a[i] * index) + coeff_b[i]) % prime) % total_no_docs

            if hash_value < min_Hash:
                min_Hash = hash_value

        doc_signature.append(min_Hash)

    corpus_signatures.append(doc_signature)

print 'About to start sim'
mse = 0
for i in range(transformedVect.shape[0]):
    print i
    for j in range(i + 1, transformedVect.shape[0]):

        minx = corpus_signatures[i]
        miny = corpus_signatures[j]

        c = len(set(minx).intersection(set(miny)))

        minsim = c/float(num_hash_fun)
	mse += (jaccard[i][j] - c)) ** 2

base = (total_no_docs*(total_no_docs-1))/float(2)
print 'Root mean squared error:',(mse/base)**0.5
print 'Time taken :', time.time() - start
