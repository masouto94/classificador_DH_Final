import streamlit as st
import numpy as np
import pandas as pd
import pickle
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import re
import unidecode



st.image('patagonia2.png') 
st.subheader('Clasificador de Comentarios en Redes Sociales')

comentario = st.text_input('Comentario:')

# Load classification model
with open('./modelo.pkl', 'rb') as modelo:
        classifier = pickle.load(modelo)

with open('./vectorizer.pkl', 'rb') as vectorizador:
        vect = pickle.load(vectorizador)

# Función normalizadora de la frase
def normalizador_frase(x):
    #Minusculizador,saltos de línea y números
    x = unidecode.unidecode(x.lower())
    x = re.sub(r'\n',' ', x)
    x = re.sub(r'([^\s\w]|_|)+','',x)
    x = re.sub(r'\d+','', x)
    #Letras repetidas
    á=re.compile('(á+)')
    é=re.compile('(é+)')
    í=re.compile('(í+)')
    ó=re.compile('(ó+)')
    ú=re.compile('(ú+)')
    aa= re.compile('(aa+)')
    bb= re.compile('(bb+)')
    ccc= re.compile('(ccc+)')
    dd= re.compile('(dd+)')
    eee= re.compile('(eee+)')
    ff= re.compile('(ff+)')
    gg= re.compile('(gg+)')
    hh= re.compile('(hh+)')
    ii= re.compile('(ii+)')
    jj= re.compile('(jj+)')
    kk= re.compile('(kk+)')
    lll= re.compile('(lll+)')
    mm= re.compile('(mm+)')
    nn= re.compile('(nn+)')
    ññ= re.compile('(ññ+)')
    ooo= re.compile('(ooo+)')
    pp= re.compile('(pp+)')
    qq= re.compile('(qq+)')
    rrr= re.compile('(rrr+)')
    ss= re.compile('(ss+)')
    tt= re.compile('(tt+)')
    uu= re.compile('(uu+)')
    vv= re.compile('(vv+)')
    ww= re.compile('(ww+)')
    xx= re.compile('(xx+)')
    yy= re.compile('(yy+)')
    zzz= re.compile('(zzz+)')
    x = re.sub(á, 'a', x)
    x = re.sub(é, 'e', x)
    x = re.sub(í, 'i', x)
    x = re.sub(ó, 'o', x)
    x = re.sub(ú, 'u', x)
    x = re.sub(aa, 'a', x)
    x = re.sub(bb, 'b', x)
    x = re.sub(ccc, 'cc', x)
    x = re.sub(dd, 'd', x)
    x = re.sub(eee, 'ee', x)
    x = re.sub(ff, 'f', x)
    x = re.sub(gg, 'g', x)
    x = re.sub(hh, 'h', x)
    x = re.sub(ii, 'i', x)
    x = re.sub(jj, 'j', x)
    x = re.sub(kk, 'k', x)
    x = re.sub(lll, 'll', x)
    x = re.sub(mm, 'm', x)
    x = re.sub(nn, 'n', x)
    x = re.sub(ññ, 'ñ', x)
    x = re.sub(ooo, 'oo', x)
    x = re.sub(pp, 'p', x)
    x = re.sub(qq, 'q', x)
    x = re.sub(rrr, 'rr', x)
    x = re.sub(ss, 's', x)
    x = re.sub(tt, 't', x)
    x = re.sub(uu, 'u', x)
    x = re.sub(vv, 'v', x)
    x = re.sub(ww, 'w', x)
    x = re.sub(xx, 'x', x)
    x = re.sub(yy, 'y', x)
    x = re.sub(zzz, 'zz', x)
    return x

        
# Función de preprocesamiento
def stemfraseesp(x):    
    frase = normalizador_frase(x)
    token_words=word_tokenize(frase)
    #token_words
    stem_sentence=[]    
    spanishStemmer=SnowballStemmer("spanish",ignore_stopwords=True)
    for word in token_words:
        stem_sentence.append(spanishStemmer.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

        
if comentario != '':
    # Pre-process 
    #sentence = vect.transform(stemfraseesp(comentario)) 
    sentence=stemfraseesp(comentario)
    dato = [sentence]
    texto_vec=vect.transform(dato)
         
    # Make predictions
    with st.spinner('Predicting...'):
        clase=classifier.predict_proba(texto_vec)[:,0]>=0.3
        if clase == True:
            st.image('negativo.png', width=300) 
            #st.write(':anguished:') 
        else:
            st.image('positivo.png', width=300)
            #st.write(':sunglasses:') 
        
     
    
    
        
