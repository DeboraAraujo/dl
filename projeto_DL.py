
# coding: utf-8

# In[1]:


import numpy
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Merge
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout

# fix random seed for reproducibility
numpy.random.seed(7)


# In[2]:


# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)


# In[3]:


def dividir_classe(x, y):
    # divide os dados em DUAS classes
    x1_features = []
    y1_label = []
    x2_features = []
    y2_label = []
    for i in range(0,len(y)):
        if y[i]==1:
            x1_features.append(x[i])
            y1_label.append(y[i])
        else:
            x2_features.append(x[i])
            y2_label.append(y[i])
    return numpy.asarray(x1_features), numpy.asarray(y1_label), numpy.asarray(x2_features), numpy.asarray(y2_label)




# In[4]:


# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


# In[5]:


x_train1, y_train1, x_train2, y_train2 =  dividir_classe(X_train, y_train)
x_test1, y_test1, x_test2, y_test2 =  dividir_classe(X_test, y_test)


# In[6]:

optimize = "rmsprop"
num_neu = 50
func_activation = "sigmoid"

# create the MODEL 1
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(num_neu))
model.add(Dropout(0.2))
model.add(Dense(1, activation=func_activation))
model.compile(loss='binary_crossentropy', optimizer=optimize, metrics=['accuracy'])
#print(model.summary())
model.fit(x_train1, y_train1, epochs=1, batch_size=64)


# In[ ]:


# create the MODEL 2
#embedding_vecor_length = 32
model2 = Sequential()
model2.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model2.add(Dropout(0.2))
model2.add(LSTM(num_neu))
model2.add(Dropout(0.2))
model2.add(Dense(1, activation=func_activation))
model2.compile(loss='binary_crossentropy', optimizer=optimize, metrics=['accuracy'])
#print(model2.summary())
model2.fit(x_train2, y_train2, epochs=1, batch_size=64)


# In[ ]:


#MODEL 3
#embedding_vecor_length = 32
model3 = Sequential()
model3.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model3.add(Dropout(0.2))

model3.add(LSTM(num_neu, return_sequences=True))
model3.add(Dropout(0.2))
keras.initializers.RandomUniform(minval=-1, maxval=0.05, seed=None)
model3.add(Dense(1, activation=func_activation, kernel_initializer='random_uniform'))
#weights=model.layers[-1].get_weights()

model3.add(LSTM(num_neu))
model3.add(Dropout(0.2))
keras.initializers.RandomUniform(minval=0, maxval=0.222, seed=None)
model3.add(Dense(1, activation=func_activation, kernel_initializer='random_uniform'))
#weights=model2.layers[-1].get_weights()

model3.compile(loss='binary_crossentropy', optimizer=optimize, metrics=['accuracy'])
#print(model3.summary())
model3.fit(X_train, y_train, epochs=3, batch_size=64)


# In[ ]:


# model.evaluate TESTE COMPLETO
scores = model3.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

