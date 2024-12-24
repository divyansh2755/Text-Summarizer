from nltk.corpus import stopwords
import codecs
import sys
from collections import Counter
import json
from math import log
import math
import os
import random

stop_dict={}
IDF_dict={}
total_no_doc=2
def Make_stop_dict():
    for w in stopwords.words():
        stop_dict[w]=None

def Filter_stopwords(text):
    if(text in stop_dict):
        return False
    return True

def Cal_IDF_Assign_ID(text):
    words=text.split()
    words=[x.lower() for x in words]
    freqs=Counter(words)
    for word in freqs:
        if(Filter_stopwords(word)):
            if word in IDF_dict:
                IDF_dict[word][1]=IDF_dict[word][1]+1
            else:
                IDF_dict[word]=[len(IDF_dict),1]


def tf_idf_doc(text):
    words=text.split()
    words=[x.lower() for x in words]

    temp_log_total_doc=log(total_no_doc)
    freqs=Counter(words)
    max_tf=max(freqs.iteritems(), key=lambda key:freqs[key])[1]

    tf_idf_list=[]
    for word in freqs:
        if word in IDF_dict:
            tf_idf_list.append((IDF_dict[word][0],(freqs[word]/max_tf)*(temp_log_total_doc-log(IDF_dict[word][1]))))
    sorted(tf_idf_list, key=lambda t: t[0])

    return tf_idf_list



def cosine_similarity_between_two_vector(doc_tf_list1,doc_tf_list2,centroid_magnitude):
    cosine_score=0
    i=0
    j=0
    len_1=len(doc_tf_list1)
    len_2=len(doc_tf_list2)
    magnitude_1=0
    magnitude_2=0
    while(i<len_1 or j<len_2):
        if i<len_1 and j<len_2:
            if(doc_tf_list1[i][0]==doc_tf_list2[j][0]):
                cosine_score+=doc_tf_list1[i][1]*doc_tf_list2[j][1]
                magnitude_1+=doc_tf_list1[i][1]*doc_tf_list1[i][1]
                magnitude_2+=doc_tf_list2[j][1]*doc_tf_list2[j][1]
                i+=1
                j+=1
            elif(doc_tf_list1[i][0]<doc_tf_list2[j][0]):
                magnitude_1+=doc_tf_list1[i][1]*doc_tf_list1[i][1]
                i+=1
            else:
                magnitude_2+=doc_tf_list2[j][1]*doc_tf_list2[j][1]
                j+=1
        else:
            if i<len_1:
                magnitude_1+=doc_tf_list1[i][1]*doc_tf_list1[i][1]
                i+=1
            if j<len_2:
                magnitude_2+=doc_tf_list2[j][1]*doc_tf_list2[j][1]
                j+=1

    if cosine_score!=0:
        return cosine_score/(math.sqrt(magnitude_1)*math.sqrt(magnitude_2))
    return cosine_score


def cosine_similarity_vec_cen(doc_tf_list1,centroid,centroid_magnitude):
    cosine_score=0
    i=0
    j=0
    len_1=len(doc_tf_list1)
    magnitude_1=0
    while(i<len_1):
        cosine_score+=doc_tf_list1[i][1]*centroid[doc_tf_list1[i][0]]
        magnitude_1+=doc_tf_list1[i][1]*doc_tf_list1[i][1]
        i+=1
    if cosine_score!=0:
        return cosine_score/(math.sqrt(magnitude_1)*centroid_magnitude)
    return cosine_score


def cal_magnitude(vector_doc):
    magnitude=0
    for no in vector_doc:
        magnitude+=no*no
    return math.sqrt(magnitude)


def Generate_initial_centroids(no_of_cluster):
    Centroids=[]
    centroid_magnitude=[]
    for i in range(0,no_of_cluster):
        Centroids.append(random.sample(range(i*25)),len(IDF_dict))
        centroid_magnitude.append(cal_magnitude(Centroids[-1]))
    return [Centroids,centroid_magnitude]


def cal_change(old_centroid,new_centroid,no_of_cluster,centroid_length):
    total_change=0
    for i in range(0,no_of_cluster):
        for j in range(0,centroid_length):
            total_change+=math.fabs(old_centroid[i][j]-new_centroid[i][j])
    return total_change/no_of_cluster


def K_means(no_of_cluster,doc_vectors,threshold,max_iteration):
    [New_centroids,centroid_magnitude]=Generate_initial_centroids(no_of_cluster)
    centroid_length=len(IDF_dict)
    Centroids=[[0]*centroid_length]*no_of_cluster

    No_element_centroid=[1]*no_of_cluster
    change=int('inf')
    max_similarity=float('-inf')
    max_cluster_no=0
    c_s=0
    no_iteration=0
    while(cal_change(Centroids,New_centroids)>threshold and no_iteration<max_iteration):
        Centroids=New_centroids
        New_centroids=[[0]*centroid_length]*no_of_cluster
        No_element_centroid=[1]*no_of_cluster
        for doc in doc_vectors:
            for i in range(0,no_of_cluster):
                c_s=cosine_similarity_vec_cen(doc,Centroids[i],centroid_magnitude[i])
                if(c_s>max_similarity):
                    max_similarity=c_s
                    max_cluster_no=i
            ##calculating new cluster centroid
            for word in doc:
                New_centroids[max_cluster_no][word[0]]+=word[1]
                No_element_centroid[max_cluster_no]+=1
        ##Averaging each centroid
            centroid_magnitude=[0]*no_of_cluster
            for i in range(0,no_of_cluster):
                for j in range(0,centroid_length):
                    New_centroids[i][j]/=No_element_centroid[i]
                    centroid_magnitude[i]+=New_centroids[i][j]
                centroid_magnitude[i]=math.sqrt(centroid_magnitude[i])
        no_iteration+=1
    return [New_centroids,centroid_magnitude]





if __name__=="__main__":
    IDF_dict_filename='Idf_dict.txt'
    tf_idf_doc_filename='Tf_idf_docs.txt'
    centroid_filename='Centroids.txt'
    sys.stdout=codecs.getwriter('utf8')(sys.stdout.buffer)

    if(sys.argv[1]=='-IDF'):
        mode=input('Do you want to proceed with IDF calculation?(y/n)')
        if(mode!='y'):
            exit()
        Make_stop_dict()
        ##########################

        files=os.listdir('./docs')
        #####
        total_no_doc=len(files)
        #####
        for file in files:
            ifile=open(file,'r',errors='ignore')
            Cal_IDF_Assign_ID(ifile.read())
            ifile.close()
        ofile=open(IDF_dict_filename,'w',errors='ignore')
        json.dump(IDF_dict,ofile)
        ofile.close()
    elif(sys.argv[1]=='-tfidf_doc'):
        mode=input('Do you want to proceed tfidf?(y/n)')
        if(mode!='y'):
            exit()
        ##########################
        IDF_dict=json.load(open(IDF_dict_filename,'r'))
        ofile=open(tf_idf_doc_filename,'w',errors='ignore')
        Docs_vector=[]
        #########################
        files=os.listdir('./docs')
        for file in files:
            ifile=open(file,'r',errors='ignore')
            tf_idf_list=tf_idf_doc(ifile.read())

            if(len(tf_idf_list)!=0):
                Docs_vector.append(tf_idf_list)

            ifile.close()
        json.dump(Docs_vector)
        ofile.close()
    else:
        no_of_cluster=input('Enter no. of cluster = ')
        max_iteration=input('Enter max_iteration = ')
        IDF_dict=json.load(open(IDF_dict_filename,'r'))

        Docs_vector=json.load(open(tf_idf_doc_filename,'r',errors='ignore'))
        Centroids=K_means(no_of_cluster,Docs_vector,0,max_iteration)
        json.dump(Centroids,open(centroid_filename,'w'))
