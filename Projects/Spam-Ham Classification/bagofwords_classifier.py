#to classify whether a sms is spam or ham

#dependencies
import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize as st
from nltk.stem import WordNetLemmatizer as wordnet
import re

#reading the file
df=pd.read_csv('spam.csv',encoding = 'ISO-8859-1',usecols=['v1','v2'])
corpus=[]
wordnet=wordnet()
length=len(df['v2'])
for i in range(length):
	rev=re.sub('[^a-zA-Z]',' ',df['v2'][i])
	rev=rev.lower()
	rev=rev.split()
	rev=[wordnet.lemmatize(word) for word in rev if word not in stopwords.words('english')]
	rev=' '.join(rev)
	corpus.append(rev)	

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)
x=cv.fit_transform(corpus).toarray()
y=df['v1'] #dependent variable

#y is a categorical variable so will encode it
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

#now splittin the model into train and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#training the model
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train,y_train)

#predicting the values
y_pred=model.predict(x_test)

#score of the model
model.score(x_test,y_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)