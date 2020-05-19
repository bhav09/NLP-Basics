'''
speech tagging - what is basically is doing , is tagging the words into various articulates of english grammar 
makes a tuple which is of the format : (word,tag)


POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent\'s
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when

'''

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
import nltk

train_text=state_union.raw('2005-GWBush.txt')
test_text=state_union.raw('2006-GWBush.txt')
#print(text)

custom_tokenizer=PunktSentenceTokenizer(train_text)
test_tokenizer=custom_tokenizer.tokenize(test_text)

#print(test_tokenizer)

def our_content():
	try:
		for i in test_tokenizer:
			words=nltk.word_tokenize(i)
			tag=nltk.pos_tag(words)
			print(tag)
	except Exception as e:
		print(str(e))
our_content()