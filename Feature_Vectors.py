from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from collections import Counter
from collections import OrderedDict
import time
import csv
import math

stmr = SnowballStemmer('english', ignore_stopwords=True)
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stops = set(stopwords.words('english')) | set('d')

body_words = OrderedDict() # It will contain all the uniques words in the entire 22,000 documents and the place at which they will occur. This will be of the type {'word' : when it occured}
c = 0
total_docs = 0
IDF = OrderedDict()  #It is used to store total number of times a word appears in the document. It will be of the type {'word' : freq}
TF = OrderedDict()   #Used for storing total word count in each document . It will be of the type {'newid of document' : total number of words}
ClassTag = OrderedDict() #Used for storing topics in the article which are used as class tags. If there are no tpics then the class tag will be empty.

#Creates a global list of words and also an intermediate file which is used to save memory. Along that it will also count the total words in document to fill TF, number of documents
# in which the word appears for IDF and also ClassTag.
def totalWordCreator(xml_filename,writer):
    xml_file = open(xml_filename, 'r')
    soup = BeautifulSoup(xml_file, 'html.parser')
    bodyTag = soup.findAll('reuters')
    global body_words
    global IDF
    global TF
    global c
    global ClassTag

    for pTag in bodyTag:
        if pTag.body and pTag.body.getText() and pTag.topics and pTag.topics.getText():
            words = Counter(list(str(stmr.stem(i)) for i in tokenizer.tokenize(pTag.body.getText()) if i.lower() not in stops))
            id = str(pTag['newid'])

            classwords = list(str(stmr.stem(k)) for k in tokenizer.tokenize(str(pTag.topics.findChildren())) if k.lower() not in stops)
            ClassTag[id] = ",".join(classwords)
            total_words = 0
            for k,v in words.items():
                total_words += v
                if k not in body_words:
                    body_words[k] = c
                    c += 1
                    IDF[k] = 1
                else:
                    IDF[k] += 1
            TF[id] = total_words
            writer.writerow([id] + words.items())
    xml_file.close()
   
start_time = time.time()
print "Creating Body Word Vector"
i = 100
bodylists = open('bodylist.csv','w+') #Opening the intermediate file.
writer = csv.writer(bodylists)
while i < 122:
    xml_filename = 'reut2-0'+str(i)[1:]+'.sgm'
    i += 1
    totalWordCreator(xml_filename,writer)
bodylists.close()

#Now The code will iterate over each row of the intermediate file(Each row is each document, so there will be aroung 20000 rows), then it will generate the Word Vector and TF IDF document.
bodywords = open('bodywordvector.csv','w+')
tfidf = open('tfidfbody.csv','w+')
writer1 = csv.writer(bodywords)
writer2 = csv.writer(tfidf)
writer1.writerow(body_words.keys()) #removed 'Document'
writer2.writerow(body_words.keys()) #removed 'Document'
bodylists = open('bodylist.csv','r')
reader = csv.reader(bodylists)
for row in reader:
    body_count = [0]*(len(body_words))
    tf_idf = [0]*(len(body_words))
    for i in row[1:]:
        word = ((i.split(',')[0])[2:]).replace('\'','')
        count = int((i.split(',')[1]).replace(')',''))
        body_count[body_words[word]] = count
        tf_idf[body_words[word]] = ((count/float((TF[row[0]])))*(math.log((len(TF))/(float(IDF[word])))))
    writer1.writerow(body_count + ClassTag[row[0]].split(",")) #removed row[0]
    writer2.writerow(tf_idf + ClassTag[row[0]].split(",")) #removed row[0]

bodywords.close()
bodylists.close()
tfidf.close()
print
print ("Time taken for creating Body Words Vector:", time.time() - start_time)

