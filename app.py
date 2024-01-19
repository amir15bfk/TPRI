import streamlit as st
import pandas as pd
import numpy as np
import nltk
import math
import matplotlib.pyplot as plt
from utils import *
# nltk.download("stopwords")
# MotsVides = nltk.corpus.stopwords.words('english')
if 'stage' not in st.session_state:
    st.session_state.stage = 0
Porter = nltk.PorterStemmer()
Lancaster = nltk.LancasterStemmer()
ExpReg = nltk.RegexpTokenizer('(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*') 
if st.button("rebuild"):
    path = "lisa2"
    build_files(path)
def read_data(labels,ty,file):
    data = []
    freq_by_doc_dict = dict() 
    poids_seq_by_doc_dict = dict() 
    poids_by_term_by_doc_dict = dict()
    freq_by_term_by_doc_dict = dict()
    ni_by_term = dict()
    if labels[0][0]=="N":
        doc_idx = 0
        term_idx =1
    else:
        doc_idx = 1
        term_idx =0
    with open(file,"r") as f:
        for i in f.readlines():
            line = i.split()
            to_add = [ ty[j](line[j]) for j in range(len(line))]
            data.append(to_add)
            if freq_by_doc_dict.get(to_add[doc_idx]):
                freq_by_doc_dict[to_add[doc_idx]]+=to_add[2]
            else:
                freq_by_doc_dict[to_add[doc_idx]]=to_add[2]
            if poids_seq_by_doc_dict.get(to_add[doc_idx]):
                poids_seq_by_doc_dict[to_add[doc_idx]]+=to_add[3]**2
            else:
                poids_seq_by_doc_dict[to_add[doc_idx]]=to_add[3]**2
            
            if freq_by_term_by_doc_dict.get(to_add[doc_idx]):
                if freq_by_term_by_doc_dict.get(to_add[doc_idx]).get(to_add[term_idx]):
                    freq_by_term_by_doc_dict[to_add[doc_idx]][to_add[term_idx]]+=to_add[2]
                else:
                    freq_by_term_by_doc_dict[to_add[doc_idx]][to_add[term_idx]]=to_add[2]
            else:
                freq_by_term_by_doc_dict[to_add[doc_idx]]={to_add[term_idx]:to_add[2]}

            if poids_by_term_by_doc_dict.get(to_add[doc_idx]):
                if poids_by_term_by_doc_dict.get(to_add[doc_idx]).get(to_add[term_idx]):
                    poids_by_term_by_doc_dict[to_add[doc_idx]][to_add[term_idx]]+=to_add[3]
                else:
                    poids_by_term_by_doc_dict[to_add[doc_idx]][to_add[term_idx]]=to_add[3]
            else:
                poids_by_term_by_doc_dict[to_add[doc_idx]]={to_add[term_idx]:to_add[3]}

            if ni_by_term.get(to_add[term_idx]):
                ni_by_term[to_add[term_idx]][to_add[doc_idx]]=1
            else:
                ni_by_term[to_add[term_idx]]={to_add[doc_idx]:1}
    

                

    out = pd.DataFrame(data)
    out.columns = labels
    print("done")
    for i in ni_by_term.keys():
        ni_by_term[i]= len(ni_by_term[i].values())
    return out,freq_by_doc_dict,freq_by_term_by_doc_dict,poids_by_term_by_doc_dict,ni_by_term,poids_seq_by_doc_dict

if st.session_state.stage == 0:
    
    with st.spinner('data loading...'):
        st.session_state.dataset ={
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
    st.session_state.stage = 1


st.toast("welcome to tp RI",icon='👋')
st.toast("by BENBACHIR Mohamed Amir",icon="❤️")
st.title("Search engine")
st.text("tp RI")

# Create a search bar

tp8_querys = st.toggle("tp 8 querys") 
if tp8_querys:
    with open("Queries") as f :
        queries = f.readlines()
    with open("Judgements") as f:
        judgements = [[] for i in queries]
        for i in f:
            x,y = map(int,i.split())
            judgements[x-1].append(y)
    qry_num = st.number_input("query number",1,len(queries),value=1)
    search_query = st.text_input("Query",value=queries[qry_num-1])
else:
    search_query = st.text_input("Query")


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
with st.spinner('data loading...'):
    data,freq_by_doc_dict,freq_by_term_by_doc_dict,poids_by_term_by_doc_dict,ni_by_term,poids_seq_by_doc_dict = st.session_state.dataset[index][chunking][preprocess]
N = np.max(data["N° document"])


# Create check buttons for filtering
options = st.multiselect("Filter by Category", data["Terme"].unique())


if matching == "Probabilistic Model(BM25)":
    filtered_data = data.copy()
    q = ExpReg.tokenize(search_query)
    if preprocess=="***lemmatization***":
        q = [Lancaster.stem(v) for v in q]
    else:
        q = [Porter.stem(v) for v in q]
    # q = [terme for terme in q if terme.lower() not in MotsVides]
    out = [[i,0] for i in range(1,N+1)]

    avdl = np.sum(filtered_data["Fréquence"])/N
    # print("avdl :",avdl)
    for i  in range(1,N+1):
        if freq_by_doc_dict.get(i):
            # temp = filtered_data[filtered_data['N° document'] == i]
            # dl = np.sum(temp["Fréquence"])
            dl = freq_by_doc_dict[i]
            temp = freq_by_term_by_doc_dict[i]
            # print(f"dl {i}  :{dl}")
            for v in q:

                if temp.get(v):
                    freq = temp[v]
                else:
                    freq= 0
                if ni_by_term.get(v):
                    ni = ni_by_term[v]
                else:
                    ni = 0
                
                out[i-1][1]+=((freq/(freq+K*((1-B)+((B*dl)/avdl))))*(math.log10((N-ni+0.5)/(ni+0.5))))
    out.sort(key = lambda x:x[1],reverse=True)
    st.session_state.out = pd.DataFrame(out,columns=['N° document',"RSV"])
    st.session_state.out = st.session_state.out[st.session_state.out["RSV"]!=0]
elif matching == "Vector Space Model":
    filtered_data = data.copy()
    q = ExpReg.tokenize(search_query)
    q = [terme for terme in q if terme.lower() not in MotsVides]
    if preprocess=="***lemmatization***":
        q = [Lancaster.stem(v) for v in q]
    else:
        q = [Porter.stem(v) for v in q]
    out = []
    for i  in range(1,N+1):
        if freq_by_doc_dict.get(i):
            out.append([i,0])
            temp = poids_by_term_by_doc_dict[i]
            if typeVSM == "Cosine Measure":
                out[-1].append(np.sqrt(poids_seq_by_doc_dict[i]))
            elif typeVSM == "Jaccard Measure":
                out[-1].append(poids_seq_by_doc_dict[i])
            
            for v in q:
                if temp.get(v):
                    out[-1][1]+=temp[v]
    if typeVSM == "Scaler Product":
        pass
    elif typeVSM == "Cosine Measure":
        out =[[i[0],i[1]/(np.sqrt(len(q))*i[2])] for i in out]
    elif typeVSM == "Jaccard Measure":
        out =[[i[0],i[1]/(len(q)+i[2]-i[1])] for i in out]
    out.sort(key = lambda x:x[1],reverse=True)
    st.session_state.out = pd.DataFrame(out,columns=['N° document',"RSV"])
    st.session_state.out = st.session_state.out[st.session_state.out["RSV"]!=0]
elif matching == "Boolean Model":
    filtered_data = data.copy()
    q = ExpReg.tokenize(search_query)
    # if preprocess=="***lemmatization***":
    #     q = [Lancaster.stem(v) for v in q]
    # else:
    #     q = [Porter.stem(v) for v in q]
    if check_query(q):
        nots, opps, values = remove_opp(q)
        if preprocess=="***lemmatization***":
            values = [Lancaster.stem(v) for v in values]
        else:
            values = [Porter.stem(v) for v in values]
        out = []
        
        for i  in range(1,N+1):
            if freq_by_doc_dict.get(i):
                out.append([i,[]])
                temp = poids_by_term_by_doc_dict[i]
                
                for v in values:
                    if temp.get(v):
                        out[-1][1].append(True)
                    else:
                        out[-1][1].append(False)
                out[-1][1] = evaluate(nots, opps,out[-1][1])
        out.sort(key = lambda x:x[1],reverse=True)
        st.session_state.out = pd.DataFrame(out,columns=['N° document',"RSV"])
        st.session_state.out = st.session_state.out[st.session_state.out["RSV"]]
    else:
        st.error("wrong query")

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

st.dataframe(st.session_state.out)
st.dataframe(filtered_data,use_container_width=True)

if tp8_querys:
    partinate = 0
    for i in st.session_state.out['N° document']:
        if i in judgements[qry_num-1]:
            partinate+=1
    precision = partinate/len(st.session_state.out['N° document'])
    st.write(f"p = {precision*100:0.2f} %")
    partinate_5 = 0
    out = st.session_state.out['N° document'][:5] if len(st.session_state.out['N° document'])>=5 else st.session_state.out['N° document']
    for i in out:
        if i in judgements[qry_num-1]:
            partinate_5+=1
    st.write(f"p@5 = {partinate_5/5*100:0.2f} %")
    partinate_10 = 0
    out = st.session_state.out['N° document'][:10] if len(st.session_state.out['N° document'])>=10 else st.session_state.out['N° document']
    for i in out:
        if i in judgements[qry_num-1]:
            partinate_10+=1
    st.write(f"p@10 = {partinate_10/10*100:0.2f} %")
    recall = partinate /len(judgements[qry_num-1])
    st.write(f"recall = {recall*100:0.2f} %")
    f1 = (2*recall*precision)/(recall+precision)
    st.write(f"f1_score = {f1*100:0.2f} %")
    # curbe-recall-precision
    recalls = []
    precisions = []
    ris = [i/10 for i in range(11)]
    partinate = 0
    size = 0
    for i in range(10):
        size += 1
        if len(st.session_state.out['N° document'])>i:
            if st.session_state.out['N° document'][i] in judgements[qry_num-1]:
                partinate+=1
        precision = partinate/size
        recall = partinate /len(judgements[qry_num-1])
        precisions.append(precision)
        recalls.append(recall)
    precisions_ri =[]
    for i in range(11):
        precisions_ri.append(max([precisions[j] for j in range(10) if recalls[j]>=ris[i]]+[0]))
    fig,ax = plt.subplots()
    ax.plot(ris,precisions_ri)
    ax.set_title('Recall-Precision Curve')
    ax.set_xlabel('Recall (ri)')
    ax.set_ylabel('Precision')
    ax.set_ylim([-0.1, 1.1])
    ax.grid(True)
    st.pyplot(fig)