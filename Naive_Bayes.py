import csv
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
import time
import numpy as np


def writePredictionsToCSV(mlb,neigh,testx):
    bodyclassnb = open('bodyclassnb.csv','w+')
    writer1 = csv.writer(bodyclassnb)
    writer1.writerow(mlb.classes_)
    writer1.writerows(neigh.predict(testx))

start1 = time.time()
X = []
Y = []

bodywords = open('/home/9/mandadapu.1/dmassignment1/bodywordvector.csv','r')
reader = csv.reader(bodywords)

header = next(reader)
headerlen = len(header)

for row in reader:
    X.append(map(int,row[0:len(header)]))
    Y.append(row[len(header):])

#lenx = (3 * len(X))/5  #For 60-40 split
lenx = (4 * len(X))/5   #For 80-20 split
X=np.array(X)
Y=np.array(Y)
mlb = MultiLabelBinarizer()
transformedY = mlb.fit_transform(Y)

trainingx = X[0:lenx]
trainingy = transformedY[0:lenx]
testx = X[lenx:]
testy = transformedY[lenx:]

neigh = OneVsRestClassifier(MultinomialNB())
neigh.fit(trainingx,trainingy)
print 'Total offline cost(in seconds) :', time.time()-start1
start2 = time.time()
#writePredictionsToCSV(mlb,neigh,testx)     #Write predictions to a CSV file
#print neigh.predict(testx)                 #Print the predictions to the system output
print neigh.score(testx,testy)              #Print the accuracy of predictions to the system output

print 'Total online cost (for the entire test set, online cost for a single tuple is written in the report) :', time.time() - start2




