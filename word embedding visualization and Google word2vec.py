
# coding: utf-8

# In[3]:


from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import numpy as np


# In[57]:


sentences=[['Hello' ,'everyone' ,'this', 'is', 'a', 'NLP','article','.'] ,          
 ['NotAll','Welcome' ,'to', 'Visualization', 'Word', 'embeding', 'visualization.'],
 ['You' ,'are' ,'reading' ,'NLP' ,'article']]


# In[58]:


sentences


# In[59]:


model=Word2Vec(sentences,min_count=1)


# In[60]:


model


# In[61]:


print(model)


# In[62]:


words=list(model.wv.vocab)
print(words)


# In[63]:


print(model.wv.vocab) #to show that each word  has different embedding 


# In[64]:


print(len(model['reading']))
print((model['reading']))


# In[65]:


vocabulary=model[model.wv.vocab]


# In[66]:


vocabulary


# In[67]:


vocabulary.shape #20 unique vectors each of size 100


# In[68]:


pca_model=PCA(n_components=2)
pca_result=pca_model.fit_transform(vocabulary)


# In[70]:


plt.scatter(pca_result[:,0],pca_result[:,1])
words=list(model.wv.vocab)
for i,word in enumerate(words):
    plt.annotate(word,xy=(pca_result[i,0],pca_result[i,1]))
plt.show()               


# In[120]:


from gensim.models import KeyedVectors
import os


# In[125]:


os.getcwd()


# In[155]:


filename=os.path.normpath('C://Users/gandixit/Desktop/python tutorial/implementation/GoogleNews-vectors-negative300.bin')


# In[156]:


filename


# In[157]:


model1=KeyedVectors.load_word2vec_format(filename,binary=True)


# In[158]:


model1


# In[170]:


model1_result=model1.similar_by_word('semantic',topn=5)


# In[171]:


model1_result


# In[174]:


model1_result_1=model1.most_similar(positive=['cry','stress'],negative=['laugh'],topn=2)


# In[175]:


model1_result_1

