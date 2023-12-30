import nltk
import os
import math

nltk.download("stopwords")
MotsVides = nltk.corpus.stopwords.words('english')

def saveDesc(docs,tfidf,fname):
    with open(fname,"w") as f:
        for name,FreqDist in docs.items():
            ndoc =name[1] # get the number of doc 
            for i in dict(FreqDist).keys():
                f.write(f"{ndoc} {i} {FreqDist[i]} {tfidf[name][i]}\n")

def saveinvers(docs,tfidf,fname):
    with open(fname,"w") as f:
        for name,FreqDist in docs.items():
            ndoc =name[1] # get the number of doc 
            for i in dict(FreqDist).keys():
                f.write(f"{i} {ndoc} {FreqDist[i]} {tfidf[name][i]}\n")
def cleanerSplitPorter(text):
    Termes = text.split()
    TermesSansMotsVides = [terme for terme in Termes if terme.lower() not in MotsVides]
    Porter = nltk.PorterStemmer()
    TermesNormalisation = [Porter.stem(terme) for terme in TermesSansMotsVides]
    TermesFrequence = nltk.FreqDist(TermesNormalisation)
    return TermesFrequence

def cleanerRegPorter(text):
    ExpReg = nltk.RegexpTokenizer('(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*') 
    Termes = ExpReg.tokenize(text)
    TermesSansMotsVides = [terme for terme in Termes if terme.lower() not in MotsVides]
    Porter = nltk.PorterStemmer()
    TermesNormalisation = [Porter.stem(terme) for terme in TermesSansMotsVides]
    TermesFrequence = nltk.FreqDist(TermesNormalisation)
    return TermesFrequence

def cleanerSplitLancaster(text):
    Termes = text.split()
    TermesSansMotsVides = [terme for terme in Termes if terme.lower() not in MotsVides]
    Lancaster = nltk.LancasterStemmer()
    TermesNormalisation = [Lancaster.stem(terme) for terme in TermesSansMotsVides]
    TermesFrequence = nltk.FreqDist(TermesNormalisation)
    return TermesFrequence

def cleanerRegLancaster(text):
    ExpReg = nltk.RegexpTokenizer('(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*')  
    Termes = ExpReg.tokenize(text)
    TermesSansMotsVides = [terme for terme in Termes if terme.lower() not in MotsVides]
    Lancaster = nltk.LancasterStemmer()
    TermesNormalisation = [Lancaster.stem(terme) for terme in TermesSansMotsVides]
    TermesFrequence = nltk.FreqDist(TermesNormalisation)
    return TermesFrequence

def TFIDF(data,wordcounter):
    tfIDF = dict()
    for doc_name,FreqDist in data.items():
        tfIDF[doc_name] = dict()
        for i,j in dict(FreqDist).items():
            tfIDF[doc_name][i] =(j/ max(dict(FreqDist).values()))*math.log10(len(data)/wordcounter[i]+1)
    return tfIDF

def wordcounter(docs):
    dictOfWord = dict()
    for _,FreqDist in docs.items():
        for i in dict(FreqDist).keys():
            if not dictOfWord.get(i):
                dictOfWord[i] = 0
                for _,d in docs.items():
                    if d.get(i):
                        dictOfWord[i] +=1
    return dictOfWord

def build_files(path="Collection"):
    docs = dict()

    for i in os.listdir(path):
        with open(path+"/"+i,"r") as f:
            docs[i] = "".join(f.readlines())

    N = len(docs)

    
    SplitPorterDict = dict()
    RegPorterDict = dict()
    SplitLancasterDict = dict()
    RegLancasterDict = dict()

    for i in docs.keys():
        SplitPorterDict[i]= cleanerSplitPorter(docs[i])
        RegPorterDict[i]= cleanerRegPorter(docs[i])
        SplitLancasterDict[i]= cleanerSplitLancaster(docs[i])
        RegLancasterDict[i]= cleanerRegLancaster(docs[i])

    counterSP = wordcounter(SplitPorterDict)
    counterRP = wordcounter(RegPorterDict)
    counterSL = wordcounter(SplitLancasterDict)
    counterRL = wordcounter(RegLancasterDict)

    tfidfSP = TFIDF(SplitPorterDict,counterSP)
    tfidfRP = TFIDF(RegPorterDict,counterRP)
    tfidfSL = TFIDF(SplitLancasterDict,counterSL)
    tfidfRL = TFIDF(RegLancasterDict,counterRL)

    saveDesc(SplitPorterDict,tfidfSP,"output/DescripteursSplitPorter.txt")
    saveDesc(RegPorterDict,tfidfRP,"output/DescripteursTokenPorter.txt")
    saveDesc(SplitLancasterDict,tfidfSL,"output/DescripteursSplitLancaster.txt")
    saveDesc(RegLancasterDict,tfidfRL,"output/DescripteursTokenLancaster.txt")
    saveinvers(SplitPorterDict,tfidfSP,"output/InverseSplitPorter.txt")
    saveinvers(RegPorterDict,tfidfRP,"output/InverseTokenPorter.txt")
    saveinvers(SplitLancasterDict,tfidfSL,"output/InverseSplitLancaster.txt")
    saveinvers(RegLancasterDict,tfidfRL,"output/InverseTokenLancaster.txt")

    return N

def remove_opp(q):
    i = 0
    bool_not = False
    nots = []
    opps = []
    values = []
    while i<len(q):
        if q[i].upper() =="NOT":
            bool_not=not bool_not
        elif q[i].upper() in ["OR","AND"]:
            opps.append(q[i])
        else:
            nots.append(True if bool_not else False)
            values.append(q[i])
            bool_not = False
        i+=1
    return nots, opps, values

def evaluate(nots, opps,values):
    if len(values)==0:
        return True
    with_not = [not values[i] if nots[i] else values[i] for i in range(len(values))]
    global_bool = with_not[0]
    for i in range(len(opps)):
        if opps[i]=="OR":
            global_bool = global_bool or with_not[i+1] 
        else:
            global_bool = global_bool and with_not[i+1]
    return global_bool


def check_query(query):

    if len(query)>1:
        if query[0].upper() in ["AND","OR"]:
            return False
        if query[-1].upper() in ["NOT","AND","OR"]:
            return False
        for i,j in zip(query[:-1],query[1:]):
            if i.upper()=="NOT":
                if j.upper() in ["NOT" ,"AND","OR"]:
                    return False
        if len(query)>2:
            for j,k in zip(query[1:-1],query[2:]):
                if j.upper() in ["AND","OR"]:
                    if k.upper() in ["AND","OR"]:
                        return False
    else:
        if query[0].upper() in ["NOT","AND","OR"]:
            return False
    return True         


