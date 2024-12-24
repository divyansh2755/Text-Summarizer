import json
import nltk
import math
import collections
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer


training_file = "/Users/Dstrip/Desktop/NLP_Final/wiki_json_data/AA/wikiall.json"
stopwords_file = "stopwords.txt"

punctuations = ['!','"','#','$','%','&','\'','(',')','*','+','-','.','/','\\',',',':',';','<','=','>','?','@','[','\'',']','^','_','`','{','|','}','~']

stopwords = {}
for sw in open(stopwords_file,"r").readlines():
	sw = sw.replace("\n","")
	stopwords[sw] = 1

for punc in punctuations:
	stopwords[punc] = 1


name = "doc"
index = 1
idfs = collections.defaultdict(int)
exist = {}

with open(training_file) as wiki_file:

	for json_doc in wiki_file:

		exist.clear()
		data = json.loads(json_doc)
		fname = name + str(index)
		fout = open("/Users/Dstrip/Desktop/NLP_Final/wiki_training_docs/" + fname,"w+")
		text = data['text']

		lines = text.splitlines()
		for line in lines[1:]:
			tokenized = word_tokenize(line)
			pos_tags = nltk.pos_tag(tokenized)
			for pointer,token in enumerate(tokenized):
				token = token.lower()
				#if (token not in stopwords) and (token.isalpha()):
				if (token not in stopwords):
					word_tag = token + "::" + pos_tags[pointer][1]
					if word_tag not in exist:
						idfs[word_tag] += 1
						exist[word_tag] = True

		fout.close()
		index += 1


N = index -1

for term in idfs:
	idfs[term] = math.log(float(idfs[term]/N))

fout = open("idfs.model.txt","w+")
for term in sorted(idfs.keys()):
	fout.write(term + " " + str(idfs[term]) + "\n")
