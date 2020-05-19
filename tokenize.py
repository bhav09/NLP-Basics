'''
Tokenizing is the process in which huge sentence/ paragraphs are divided into smaller segments called tokens.
Here we will be seeing two tokenizers : word_tokenizer, sent_tokenizer 
word_tokenizer=it actually divides a group of sentence where the delimiter is the word
sent_tokenizer= it delimits the para/sentences on sentences.
'''

from nltk.tokenize import sent_tokenize, word_tokenize


text='Hello Mr. Bhavishya Pandit. How are doing? Hope everything is going smooth.'
print('Sentence Tokenize:',sent_tokenize(text))
print()
print('Word Tokenize:',word_tokenize(text))