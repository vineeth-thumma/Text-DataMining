import csv
from sklearn import tree
from sklearn.preprocessing import MultiLabelBinarizer
import time

def writePredictionsToCSV(mlb,neigh,testx):
    bodyclassdectree = open('bodyclassdectree.csv','w+')
    writer1 = csv.writer(bodyclassdectree)
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
    X.append(row[0:len(header)])
    Y.append(row[len(header):])

#lenx = (3 * len(X))/5  #For 60-40 split
lenx = (4 * len(X))/5   #For 80-20 split
mlb = MultiLabelBinarizer()
transformedY = mlb.fit_transform(Y)

trainingx = X[0:lenx]
trainingy = transformedY[0:lenx]
testx = X[lenx:]

neigh = tree.DecisionTreeClassifier()
neigh.fit(trainingx,trainingy)
print 'Total offline cost (in seconds) :', time.time() - start1
start2 = time.time()
#writePredictionsToCSV(mlb,neigh,testx)             #write the predictions to a CSV file
#print neigh.predict(testx)                         #print the predictions in the system output
print neigh.score(testx,transformedY[lenx:])        #print the accuracy of preditions in the system output
#print neigh.decision_path(testx)                   #print the decision tree path into system output

print 'Total online cost (for the entire test set, online cost for a single tuple is written in the report) :', time.time() - start2




