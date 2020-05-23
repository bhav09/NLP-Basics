'''Lemmatization is the process of grouping together the different inflected forms of a word so they can be analysed as 
a single item. Lemmatization is similar to stemming but it brings context to the words. 
So it links words with similar meaning to one word.

Text preprocessing includes both Stemming as well as Lemmatization. 
Many times people find these two terms confusing. Some treat these two as same. 
Actually, lemmatization is preferred over Stemming because lemmatization does morphological analysis of the words.
The word resulting would have the same meaning but would be a synonym of the actual word
'''

#dependency : nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemm=WordNetLemmatizer()
print(lemm.lemmatize('dogs')) #prints dog

print(lemm.lemmatize('mosquitoes')) #prints mosquito

print(lemm.lemmatize('better',pos="a")) #prints good   (a) stands for adjective 
#also the default parameter for lemmatizer is noun (n)

