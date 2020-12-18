import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
import re
from gensim.models import Word2Vec

paragraph="""Before you discuss the resolution, let me place before you one or two things,
	 I want you to understand two things very clearly and to consider them from the same
	  point of view from which I am placing them before you. I ask you to consider it from
	   my point of view, because if you approve of it, you will be enjoined to carry out 
	    all I say. It will be a great responsibility. There are people who ask me whether 
		 I am the same man that I was in 1920, or whether there has been any change in me 
		  or you. You are right in asking that question.

	Let me, however, hasten to assure that I am the same Gandhi as I was in 1920. 
	I have not changed in any fundamental respect. I attach the same importance 
	to non-violence that I did then. If at all, my emphasis on it has grown stronger. 
	There is no real contradiction between the present resolution and my previous writings and utterances.

	Occasions like the present do not occur in everybody’s and rarely in anybody’s life. 
	I want you to know and feel that there is nothing but purest Ahimsa in all that I 
	am saying and doing today. The draft resolution of the Working Committee is based on 
	Ahimsa, the contemplated struggle similarly has its roots in Ahimsa. If, therefore, 
	there is any among you who has lost faith in Ahimsa or is wearied of it, let him not 
	vote for this resolution. Let me explain my position clearly. God has vouchsafed to 
	me a priceless gift in the weapon of Ahimsa. I and my Ahimsa are on our trail today. 
	If in the present crisis, when the earth is being scorched by the flames of Himsa 
	and crying for deliverance, I failed to make use of the God given talent, God will 
	not forgive me and I shall be judged unworthy of the great gift. I must act now. 
	I may not hesitate and merely look on, when Russia and China are threatened."""

#para='He is a very good man and everyone loves him!'
para=re.sub('[^a-zA-Z.]',' ',paragraph) 
para=re.sub('\s{2,10}',' ',para) #removed extra spaces
para=para.lower()

sentences=sent_tokenize(para)

for i in range(len(sentences)):
	sentences[i]=sentences[i].split()
	sentences[i]=[word for word in sentences[i] if word not in stopwords.words('english')]

model=Word2Vec(sentences, min_count=1)

words=model.wv.vocab #vocab of the paragraph

#finding the vectors of the word
vector=model.wv['assure'] #here we see 100 dimensions of the word

#finding the word which is similar to another word
similar=model.wv.most_similar('faith')

