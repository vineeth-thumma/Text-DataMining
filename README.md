# Classification-of-Documents-based-on-similarity
It tries to classify a set of documents based on the words in text and how similar they are to other documents.

It creates a feature vector from all bag of words of the documents. It creates both 1-grams and 3-grams. 3-grams are  
better because they try to capture particular phrases and order of words in the documents, but with 3-grams the 
feature vector becomes really large so we have to make a decision between efficiency and precision.

Once the feature vectors are created then I am splitting the documents into training and testing to check accuracy of 
various classifiers.

Some of the Classifiers that I used are:
  Navie Bayes Classifier.
  Decision Tree Classifier.
  K-Nearest Neighbors Classifier.
  Min-Hash based Classifier.
  Apriori Classifier.
  
Of all the above classifiers the fastest was decision tree based classiier, but with decision tree classifier 
there is the problem of overfitting. Apriori and Naive Bayes were also closer in time takn to Decision tree but Naive Bayes
accuracy was hogher. 

Min Hash gave the highest accuracy among all the classifier but it was also one of the very slow in comparison to above three
classifiers. 
K-nearst neighbors has a very long training period and the accuracy was less than all other classifiers. 
Also for K-nearest neighbors we have decide on 'k' very carefully as it has risk of both overfitting and underfitting, 
also it effects the efficiency of the classifier. 
