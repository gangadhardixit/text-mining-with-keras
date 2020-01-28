
# coding: utf-8

# In[1]:


from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Embedding,Dropout,Flatten
from keras.datasets import imdb


# In[2]:


num_words=10000 #limit size of words 


# In[3]:


(X_train,y_train),(X_test,y_test) = imdb.load_data(num_words=num_words)


# In[13]:


print('train data points',len(X_train))
print('test data points',len(X_test))


# In[14]:


max_length=256 #sequence of tokens 
embedding_size=32
batch_size=128


# In[16]:


print('length of 1st vector ',len(X_train[0]))
print('length of 10th vector ',len(X_train[9]))  #Vectors are of variable length ,normalise each to same dimemnsion 


# In[17]:


pad='post'
X_train_pad=pad_sequences(X_train,maxlen=max_length,padding=pad,truncating=pad)
X_test_pad=pad_sequences(X_test,maxlen=max_length,padding=pad,truncating=pad)


# In[18]:


X_train_pad.shape #all vectors have 256 in length 


# In[19]:


model=Sequential()


# In[20]:


model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_length,
                   name='embedding_layer'))


# In[21]:


model.add(Flatten()) #flatten data before feeding 


# In[22]:


model.add(Dense(250,activation='relu'))


# In[23]:


model.add(Dropout(0.3))


# In[24]:


#Last layer to classify +ve and -ve 
model.add(Dense(1,activation='sigmoid'))


# In[25]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[26]:


model.summary()


# In[27]:


model.fit(X_train_pad,y_train,epochs=5,validation_data=(X_test_pad,y_test),batch_size=batch_size)


# In[31]:


evaluation=model.evaluate(X_test_pad,y_test)


# In[36]:


print("Validation loss :",(evaluation[0]*100))
print("Validation accuracy :",(evaluation[1]*100))

