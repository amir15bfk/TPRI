import streamlit as st
import pandas as pd
import numpy as np
import nltk
import math
# nltk.download("stopwords")
# MotsVides = nltk.corpus.stopwords.words('english')
Porter = nltk.PorterStemmer()
Lancaster = nltk.LancasterStemmer()
ExpReg = nltk.RegexpTokenizer('(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*') 

def read_data(labels,ty,file):
    data = []
    with open(file,"r") as f:
        for i in f.readlines():
            line = i.split()
            data.append([ ty[j](line[j]) for j in range(len(line))])
    out = pd.DataFrame(data)
    out.columns = labels
    return out
dataset = read_data(["N° document","Terme","Fréquence","Poids"],[int , str,int,float],"output/DescripteursSplitLancaster.txt")

dataset ={
    "***TERM per DOCS***" :{
        "***Split***":{
            "***lemmatization***":read_data(["N° document","Terme","Fréquence","Poids"],[int , str,int,float],"output/DescripteursSplitLancaster.txt"),
            "***stemming***":read_data(["N° document","Terme","Fréquence","Poids"],[int , str,int,float],"output/DescripteursSplitPorter.txt")
        },
        "***Tokenization***":{
            "***lemmatization***":read_data(["N° document","Terme","Fréquence","Poids"],[int , str,int,float],"output/DescripteursTokenLancaster.txt"),
            "***stemming***":read_data(["N° document","Terme","Fréquence","Poids"],[int , str,int,float],"output/DescripteursTokenPorter.txt")
        }
    },
    "***DOCS par TERM***" :{
        "***Split***":{
            "***lemmatization***":read_data(["Terme","N° document","Fréquence","Poids"],[str , int,int,float],"output/InverseSplitLancaster.txt"),
            "***stemming***":read_data(["Terme","N° document","Fréquence","Poids"],[str , int,int,float],"output/InverseSplitPorter.txt")
        },
        "***Tokenization***":{
            "***lemmatization***":read_data(["Terme","N° document","Fréquence","Poids"],[str , int,int,float],"output/InverseTokenLancaster.txt"),
            "***stemming***":read_data(["Terme","N° document","Fréquence","Poids"],[str , int,int,float],"output/InverseTokenPorter.txt")
        }
    }
}

st.toast("welcome to tp RI",icon='👋')
st.toast("by BENBACHIR Mohamed Amir",icon="❤️")
st.title("Search engine")
st.text("tp RI")

# Create a search bar
search_query = st.text_input("Query")

# Create toggle boxes
check = st.toggle(
    "check")
if check:
    query = search_query.split()
    
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
    if check_query(query):
        st.text("valide query")
    else:
        st.text("wrong query")

# Create toggle boxes
chunking = st.radio(
    "chunking type",
    ["***Split***", "***Tokenization***"],
    captions = ["String.split()", "nltk.RegexpTokenizer()"])
preprocess = st.radio(
    "preprocess",
    ["***lemmatization***", "***stemming***"],
    captions = ["nltk.LancasterStemmer()", "nltk.PorterStemmer()"])

index = st.radio(
    "index",
    ["***DOCS par TERM***", "***TERM per DOCS***"],
    captions = ["Inverse","Descripteurs"])

matching = st.radio(
    "Matching",
    ["Vector Space Model", "Probabilistic Model(BM25)","Boolean Model","Data Mining Model"])

if matching == "Vector Space Model":
    typeVSM = st.selectbox("type",["Scaler Product","Cosine Measure","Jaccard Measure"])

if matching == "Probabilistic Model(BM25)":
    K = st.number_input("K",value=1.5)
    B = st.number_input("B",value=0.75)

data = dataset[index][chunking][preprocess]



# Create check buttons for filtering
options = st.multiselect("Filter by Category", data["Terme"].unique())


if matching == "Probabilistic Model(BM25)":
    filtered_data = data.copy()
    q = ExpReg.tokenize(search_query)
    # q = [terme for terme in q if terme.lower() not in MotsVides]
    out = [[i,0] for i in range(1,7)]
    N = 6
    avdl = np.sum(filtered_data["Fréquence"])/N
    for i  in range(1,7):
        temp = filtered_data[filtered_data['N° document'] == i]
        dl = np.sum(temp["Fréquence"])
        for v in q:
            if preprocess=="***lemmatization***":
                v = Lancaster.stem(v)
            else:
                v = Porter.stem(v)
            temp2 = temp[temp["Terme"] == v]["Fréquence"]
            print(temp2)
            if len(temp2)>0:
                freq = np.sum(temp2)
            else:
                freq= 0
            print("freq =",freq)
            temp3 = filtered_data[filtered_data["Terme"] == v]
            ni = len(temp3)
            out[i-1][1]+=((freq/(freq+K*((1-B)+((B*dl)/avdl))))*(math.log10((N-ni+0.5)/(ni+0.5))))
    out.sort(key = lambda x:x[1],reverse=True)
    st.dataframe(pd.DataFrame(out,columns=['N° document',"RSV"]))
elif matching == "Vector Space Model":
    filtered_data = data.copy()
    q = ExpReg.tokenize(search_query)
    #q = [terme for terme in q if terme.lower() not in MotsVides]
    out = [[i,0] for i in range(1,7)]
    for i  in range(1,7):
        temp = filtered_data[filtered_data['N° document'] == i]
        if typeVSM == "Cosine Measure":

            out[i-1].append(np.sqrt(np.sum(temp["Poids"]**2)))
        elif typeVSM == "Jaccard Measure":
            out[i-1].append(np.sum(temp["Poids"]**2))
        
        for v in q:
            if preprocess=="***lemmatization***":
                v = Lancaster.stem(v)
            else:
                v = Porter.stem(v)
            
            temp2 = temp[temp["Terme"] == v]["Poids"]
            
            out[i-1][1]+=np.sum(temp2)
    if typeVSM == "Scaler Product":
        pass
    elif typeVSM == "Cosine Measure":
        out =[[i[0],i[1]/(np.sqrt(len(q))*i[2])] for i in out]
    elif typeVSM == "Jaccard Measure":
        out =[[i[0],i[1]/(len(q)+i[2]-i[1])] for i in out]
    out.sort(key = lambda x:x[1],reverse=True)
    st.dataframe(pd.DataFrame(out,columns=['N° document',"RSV"]))
elif matching == "Boolean Model":
    filtered_data = data.copy()
    q = ExpReg.tokenize(search_query)
    q = [i for i in q if str(i).upper() is not "AND"]
    print(len(q))
    out = [[i,0] for i in range(1,7)]
    
    for i  in range(1,7):
        temp = filtered_data[filtered_data['N° document'] == i]
        
        for v in q:
            if preprocess=="***lemmatization***":
                v = Lancaster.stem(v)
            else:
                v = Porter.stem(v)
            print(v)
            st.dataframe(temp["Terme"])
            if v in temp["Terme"]:
                out[i-1][1]+=1
                print(out[i-1][1])
        out[i-1][1] = 1 if len(q)==out[i-1][1] else 0
    out.sort(key = lambda x:x[1],reverse=True)
    st.dataframe(pd.DataFrame(out,columns=['N° document',"RSV"]))

filtered_data = data.copy()
if preprocess=="***lemmatization***":
    search_query = Lancaster.stem(search_query)
else:
    search_query = Porter.stem(search_query)

if ("(" in search_query) or (")" in search_query):
    search_query = search_query.replace('(','\(')
    search_query = search_query.replace(')','\)')

if index == "***DOCS par TERM***":
    filtered_data = filtered_data[filtered_data["Terme"].str.contains(search_query, case=False)]
else:
    filtered_data = filtered_data[filtered_data["N° document"].astype(str).str.contains(search_query, case=False)]


if options:
    filtered_data = filtered_data[filtered_data["Terme"].isin(options)]

st.dataframe(filtered_data,use_container_width=True)

