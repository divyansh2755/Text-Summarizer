import numpy, scipy.sparse
from sparsesvd import sparsesvd
import sys
import re
from collections import Counter



def tokenize():

    data=sys.stdin.read()
    sentence_dict={}

    pattern='.*?\.\s*[A-Z]'
    p=re.compile(pattern)
    for m in p.finditer(data):
        if m.start()!=0 and m.end()!=len(data):

            sentence_dict[len(sentence_dict)]=data[m.start()-1:m.end()-1]
        elif m.start()==0 and m.end()!=len(data):
            sentence_dict[len(sentence_dict)]=(data[m.start():m.end()-1])
        elif m.start==0 and m.end()==len(data):
            sentence_dict[len(sentence_dict)]=(data)
        else:
            sentence_dict[len(sentence_dict)]=(data[m.start()-1:m.end()])
    sentence_dict[len(sentence_dict)]=data[m.end()-1:]
    Word_dict={}
    for sen in sentence_dict:
        words=sentence_dict[sen].split()
        for word in words:
            word=word.lower()
            if word not in Word_dict:
                Word_dict[word]=len(Word_dict)

    return [sentence_dict,Word_dict]

def Compute_TF_IDF_matrix_of_sentence(Sentence_dict,Word_dict):

    M = numpy.zeros([len(Word_dict),len(Sentence_dict)])
    for sen in Sentence_dict:
        words=Sentence_dict[sen].split()
        words=[word.lower() for word in words]
        freqs=Counter(words)
        for key in freqs:
            M[Word_dict[key]][sen]=freqs[key]
            ##using just frequency of word in sentence as IDF is not available
            ##also one can weight each individual word based on if it is noun or verb etc.
    return M


def heapify(L,index,k):

    while(2*index+1<k):
        t=1
        if(L[index][0]<L[2*index+1][0]):
            if 2*index+2>k or L[index][0]<L[2*index+2][0]:
                return L
            else:
                t=2
        else:
            if (t==1 and (2*index+2>=k) or L[2*index+1][0]<L[2*index+2][0]):
                t=1
            else:
                t=2

        temp=L[index]
        L[index]=L[2*index+t]
        L[2*index+t]=temp
        index=2*index+t
    return L



def LSA(M,k):  ##will return top k sentences
    SM = scipy.sparse.csc_matrix(M) # convert to sparse CSC format
    u, s, vt = sparsesvd(SM,k+10) #
    ##SVD calculated at this stage, concept matrix vt, from now we can apply various approaches
    ##to filter out top k sentences.
    ##We are using OzSoy's approach
    ##Using Cross Method
    m,n=M.shape

    Avg=numpy.average(M,1)
    for i in range(0,m):
        for j in range(0,n):
            if M[i][j]<Avg[i]:
                M[i][j]=0
    Length=numpy.dot(s,vt)
    L=[]
    ##returning top k sentences
    for i in range(0,n):
        L.append(tuple([Length[i],i]))

    if k>=len(L):
        return L
    #building min heap

    count= int(k/2-1)

    while(count>=0):
        L=heapify(L,count,k)
        count-=1
    for i in range(k,len(L)):
        if L[0][0]<L[i][0]:
            L[0]=L[i]
            L=heapify(L,0,k)
    return L[:k]


if __name__=="__main__":
    [Sentence_dict,Word_dict]=tokenize()
    M=Compute_TF_IDF_matrix_of_sentence(Sentence_dict,Word_dict)
    L = LSA(M,5)
    L=sorted(L,key=lambda s : s[1])
    print(L)

    for i in L:
        print(str(i[1])+':::'+str(Sentence_dict[i[1]]).strip())

