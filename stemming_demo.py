'''Stemming is the process of reducing a word to its word stem that affixes to 
suffixes and prefixes or to the roots of words known as a lemma
'''
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

text='He played football every Tuesday.	He plays football every Tuesday. He is going to play football every Tuesday.'
words=word_tokenize(text)
#print(words)

ps=PorterStemmer()

for w in words:
	print(ps.stem(w))
