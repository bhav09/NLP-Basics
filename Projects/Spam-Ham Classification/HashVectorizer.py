#HashingVectorizer

import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize as st
from nltk.stem import WordNetLemmatizer as wordnet
import re
from sklearn.metrics import classification_report

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

	
from sklearn.feature_extraction.text import HashingVectorizer as hv

hv=hv(n_features=5000)
x=hv.fit_transform(corpus).toarray()
y=df['v1'] #dependent variable

#y is a categorical variable so will encode it
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

						#now splittin the model into train and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
						#training the model
from sklearn.linear_model import PassiveAggressiveClassifier 
model=PassiveAggressiveClassifier()
model.fit(x_train,y_train)
						#predicting the values
y_pred=model.predict(x_test)
						#score of the model
model.score(x_test,y_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(f"Classification Report : \n\n{classification_report(y_test, y_pred)}")
'''Classification Report : 

              precision    recall  f1-score   support

           0       0.98      0.99      0.99       965
           1       0.96      0.88      0.92       150

    accuracy                           0.98      1115
   macro avg       0.97      0.94      0.95      1115
weighted avg       0.98      0.98      0.98      1115
'''