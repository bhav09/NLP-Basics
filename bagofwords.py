#dependency
import nltk
from nltk.tokenize import sent_tokenize as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as ps, WordNetLemmatizer as wl

para='''An atom is the smallest unit of ordinary matter that forms a chemical element.
		Every solid, liquid, gas, and plasma is composed of neutral or ionized atoms.
		Atoms are extremely small, typically around 100 picometers across. 
		They are so small that accurately predicting their behavior using classical physics—as 
		if they were tennis balls, for example—is not possible due to quantum effects.

		Every atom is composed of a nucleus and one or more electrons bound to the nucleus. 
		The nucleus is made of one or more protons and a number of neutrons. 
		Only the most common variety of hydrogen has no neutrons. 
		More than 99.94% of an atom's mass is in the nucleus. 
		The protons have a positive electric charge, the electrons have a negative electric charge, 
		and the neutrons have no electric charge. If the number of protons and electrons are equal, 
		then the atom is electrically neutral. If an atom has more or fewer electrons than protons, 
		then it has an overall negative or positive charge, respectively – such atoms are called ions.

		The electrons of an atom are attracted to the protons in an atomic nucleus by the electromagnetic force. 
		The protons and neutrons in the nucleus are attracted to each other by the nuclear force. 
		This force is usually stronger than the electromagnetic force that repels the positively 
		charged protons from one another. Under certain circumstances, the repelling electromagnetic 
		force becomes stronger than the nuclear force. In this case, the nucleus splits and leaves 
		behind different elements. This is a form of nuclear decay.'''

#clearning the texts
import re

ps=ps() #object creation porter stemmer
wl=wl() #object creation word net lemmatizer
sentences=st(para) #tokenizing to sentences
corpus=[]

for i in range(len(sentences)):
	rev=re.sub('[^a-zA-Z]',' ',sentences[i]) #everything other than alphabets would be replaced by space
	rev=rev.lower() #lowers the letters in the sentences
	rev=rev.split() #splits them word wise into elements of a list
	rev=[wl.lemmatize(word) for word in rev if word not in set(stopwords.words('english'))]
	rev=' '.join(rev)
	corpus.append(rev) #appending to list
	
#bag of words
from sklearn.feature_extraction.text import CountVectorizer #importing countervectorizer
cv=CountVectorizer()
x=cv.fit_transform(corpus).toarray() #transforming it to an array

