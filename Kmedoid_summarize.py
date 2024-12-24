import nltk
import nltk.data
import random
from nltk import pos_tag
from nltk import word_tokenize
import sys
import subprocess
from stemming.porter2 import stem
import itertools
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import numpy.random as rnd
files = sys.argv

# Enter the test document for summarization
def arrange():
    f = open(files[1],'r')
    content = f.read()
    sentences = content.split("\n")
    # print(sentences)
    f.close()
    return sentences

# Remove the stop words
def stopword():
    list_stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
    for key in dict_sentence.keys():
        tokens = nltk.word_tokenize(dict_sentence[key])
        dict_sentence[key] = ' '.join([word for word in tokens if word not in list_stopwords])
    return dict_sentence

# Bring all the wors to their stem
def stem_words():
    lancaster_stemmer = LancasterStemmer()
    for key in dict_sentence.keys():
        tokens = nltk.word_tokenize(dict_sentence[key])
        dict_sentence[key] = ' '.join([lancaster_stemmer.stem(word) for word in tokens])
    return dict_sentence

# Find postag of all words
def postag():
    pos_dict = {}
    for key in dict_sentence.keys():
        tokens = nltk.word_tokenize(dict_sentence[key])
        tagged = nltk.pos_tag(tokens)
        pos_dict[key] = tagged
    return pos_dict

# Extract all words that are nouns
def find_nouns():
    for key in dict_sentence.keys():
        tokens = nltk.word_tokenize(dict_sentence[key])
        dict_sentence[key] = ' '.join([word.lower() for index,word in enumerate(tokens) if pos_dict[key][index][0] == word and pos_dict[key][index][1] in  nouns])
    return dict_sentence

# For each word of each sentence, use DISCO API to find 10 most similar words to them
def word_similarity():
    similar_words = {}
    for key in dict_sentence.keys():
        similar_words[key] = []
        tokens = nltk.word_tokenize(dict_sentence[key])
        for word in tokens:
            output = subprocess.check_output(['java', '-jar', 'disco-1.3.jar', 'DATA-DIRECTORY', '-bn', word.lower(), '10'])
            output = output.split()
            x = '"' + word.lower() + '"'
            if x in output:
                continue
            outputnew = [x for i, x in enumerate(output) if i % 2 == 0]
            similar_words[key] += outputnew
            similar_words[key].append(word)
        similar_words[key] = set(similar_words[key])
        similar_words[key] = list(similar_words[key])
    return similar_words

# Find the similarity between each pair of sentences
def jaccard_similarity():
    indexes = dict_sentence.keys()
    pairs = itertools.combinations(indexes, 2)
    sent_similarity = {}
    for pair in pairs:
        common_words = [word for word in similar_words[pair[0]] for word1 in similar_words[pair[1]] if word == word1]
        try:
            sent_similarity[pair] = (2 * len(common_words))/float((len(similar_words[pair[0]]) + len(similar_words[pair[1]])))
        except ZeroDivisionError:
            sent_similarity[pair] = 0
    return sent_similarity


# use k-medoid clustering to find medoid of each cluster
def kmedoid():

    medoids = {}
    k = 5
    ini = []

    while len(ini) != k:
        r = random.randint(0, len(sentence_list)-1)
        if r not in ini:
            ini.append(r)
    for i in range(k):
        medoids[i] = ini[i]

    for i in range(100):
        clusters = {}
        for x in medoids.values():
            clusters[x] = []
        for j in range(len(sentence_list)):
            if j not in medoids.values():
                mindistance = float('inf')
                for num, l in medoids.items():
                    if medoids[num] != j:
                        index = tuple(sorted([medoids[num], j]))
                        if distance_pairs[index] < mindistance:
                            mindistance = distance_pairs[index]
                            cluster = l
                clusters[cluster].append(j)
        for index, points in enumerate(clusters.values()):
            minsum = float('inf')
            for sentence in points:
                summ = 0
                for p in points:
                    if sentence != p:
                        summ += distance_pairs[tuple(sorted([sentence, p]))]
                if summ < minsum:
                    minsum = summ
                    cluster = sentence
                    medoids[index] = cluster

    return medoids


sentence_list = arrange()
dict_sentence = {}
for index,sent in enumerate(sentence_list):
    dict_sentence[index] = sent
dict_sentence = stopword()
dict_sentence = stem_words()
pos_dict = postag()
nouns = ['NN', 'NNP', 'NNS', 'NNPS']
dict_sentence = find_nouns()
similar_words = word_similarity()
sent_similarity = jaccard_similarity()
distance_pairs = {}
for pair in sent_similarity:
    distance_pairs[pair] = 1 - sent_similarity[pair]
medoids = kmedoid()
for i in medoids.values():
    print(sentence_list[i])
