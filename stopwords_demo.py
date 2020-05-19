'''
stopwords can be understood as : to exclude or stop at a point where a certain word among the list of words occur
in a particular para/ sentence
'''
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text='It is an example of showing the stop words filteration.'
stop_words=stopwords.words('english')
#print(stop_words)

filtered_list=[]
#now filtering our sentence
words=word_tokenize(text)
for w in words:
	if w not in stop_words:
		filtered_list.append(w)
print(filtered_list)  # ['It', 'example', 'showing', 'stop', 'words', 'filteration', '.']