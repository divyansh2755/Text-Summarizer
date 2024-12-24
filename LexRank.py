import re
import math
import numpy as np
from nltk import word_tokenize
from collections import Counter,defaultdict
from itertools import combinations

original_sentences = {}

def ReadIDFs(fidf):
	idf = {}
	for line in open(fidf,"r").readlines():
		tokens = line.split()
		idf[tokens[0]] = float(tokens[1])
	return idf

def CreateStopDict(fstopwords):
	punctuations = ['!','"','#','$','%','&','\'','(',')','*','+','-','.','/','\\',',',':',';','<','=','>','?','@','[','\'',']','^','_','`','{','|','}','~','\'\'','``']
	stopwords = {}
	for sw in open(fstopwords,"r").readlines():
		sw = sw.replace("\n","")
		stopwords[sw] = 1
	for punc in punctuations:
		stopwords[punc] = 1
	return stopwords

def Similarity_Matrix(idfs,stopwords,document):

	with open(document,"r") as doc:

		global original_sentences
		original_sentences = {}
		sentence_tfidf = {}

		index = 1
		for line in doc.readlines():
			original_sentences[index] = line
			tokenized = word_tokenize(line.rstrip("\n"))
			words = [w.lower() for w in tokenized if (w not in stopwords)]
			sentence_tfidf[index] = { word : tf * idfs[word] if word in idfs else 0 for word,tf in Counter(words).items() }
			index += 1


		cosines = defaultdict(dict)
		sentences = sorted(sentence_tfidf.keys())
		pairs = list(combinations(sentences,2))

		for pair in pairs:
			Sx = pair[0]
			Sy = pair[1]
			nom = sum([sentence_tfidf[Sx][word]*sentence_tfidf[Sy][word] if word in sentence_tfidf[Sy] else 0 for word in sentence_tfidf[Sx]])
			denomSx = [sentence_tfidf[Sx][word]**2 for word in sentence_tfidf[Sx]]
			denomSx = math.sqrt(sum(denomSx))
			denomSy = [sentence_tfidf[Sy][word]**2 for word in sentence_tfidf[Sy]]
			denomSy = math.sqrt(sum(denomSy))
			denom = denomSx * denomSy
			cosines[Sx][Sy] = nom/denom

		for sentence in sentences:
			cosines[sentence][sentence] = 1.0

		rows = []
		for index1 in sentences:
			row = []
			for index2 in range(1,len(sentences)+1):
				if index2 in cosines[index1]:
					row.append(cosines[index1][index2])
				else:
					row.append(cosines[index2][index1])
			rows.append(row)

		similarity_matrix = np.array(rows)

		return similarity_matrix

def Compute_Lexrank(Similar_matrix,threshold):
	m, n = Similar_matrix.shape
	Damping_factor=0.85

	for i in range(0,m):
		row_sum=0
		for j in range(0,n):
			row_sum+=Similar_matrix[i][j]
		for j in range(0,n):
			if row_sum!=0:
				Similar_matrix[i][j]=(1-Damping_factor)/n+Damping_factor*(Similar_matrix[i][j]/row_sum)
			else:
				Similar_matrix[i][j]=(1-Damping_factor)/n
	return Similar_matrix


def PowerMethod(Similar_matrix,threshold):
	m, n = Similar_matrix.shape
	P=np.array([1/n]*n)
	Pn=np.array([1/n]*n)
	error=float('inf')
	St=np.transpose(Similar_matrix)
	while error>threshold:
		Pn=np.dot(St,P)
		error = np.linalg.norm((Pn-P))
		P=Pn
	return P


if __name__ == "__main__" :

	idf = ReadIDFs("idfs.model.lower.txt")
	stopwords = CreateStopDict("stopwords.txt")
	document = "doc2"

	Similar_matrix = Similarity_Matrix(idf,stopwords,document)
	#print(Similar_matrix)
	Similar_matrix = Compute_Lexrank(Similar_matrix,0.1)
	P = PowerMethod(Similar_matrix,0.1)
	#print(P)
	summary = np.argsort(P)[::-1][:5]
	# increase indexes by 1 -- to resemble sentences indexes
	summary = [num+1 for num in summary]

	for index in sorted(summary):
		print(original_sentences[index].rstrip("\n"))



