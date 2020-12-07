#tf idf
import nltk

para = '''An atom is the smallest unit of ordinary matter that forms a chemical element.
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

#dependencies
import re #regular expression
from nltk.tokenize import sent_tokenize as st, word_tokenize as wt #for tokenization
from nltk.corpus import stopwords #stop words
from nltk.stem import WordNetLemmatizer as wl #for lemmatization
 
wordnet=wl() #object creation for lemmatization
corpus=[] #empty list
sentences=st(para) #tokenizing the paragraph to sentences

for i in range(len(sentences)):
	rev=re.sub('[^a-zA-Z]',' ',sentences[i]) #replace all the letters by space except the alphabets
	rev=rev.lower() #lower the senteces
	rev=rev.split() #each word gets converted to an element of a list
	rev=[wordnet.lemmatize(word) for word in rev if word not in stopwords.words('english')]
	rev=' '.join(rev)
	corpus.append(rev)
	
#creating TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
cv=tfidf() #object creation
x=cv.fit_transform(corpus).toarray() #transforming