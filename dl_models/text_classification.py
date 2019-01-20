import keras
import tensorflow
import pickle
from keras.datasets import imdb, reuters
from keras.utils import to_categorical
from keras import models, optimizers, preprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, Embedding, SimpleRNN
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline


# We use second method here
def vectorize_sequence(sequences, NUM_WORDS):
    vector_data = np.zeros((len(sequences), NUM_WORDS))

    for ind, val in enumerate(sequences):
        vector_data[ind, val] = 1

    return vector_data


    '''
	Train embeddings on imdb dataset 
	Feed trained embedding to dense layer 
	(We can also use pre-trained embedding) 
	but here we try with custom embeddings
    '''
def dense_with_embeddings_model(max_features, max_len):


	model = Sequential()
	# maximum input to embedding layer is 8 (kind of window size)
	model.add(Embedding(max_features, 8, input_length=max_len)) # outputs (samples, maxlen, 8)
	model.add(Flatten())
	model.add(Dense(64))
	model.add(Dense(1))
	model.add(Activation('sigmoid')) # since it is binary classifcation (sigmoid activation makes sense)

	return model

'''
Output of Embedding is already compatible with Convolution1D
'''

def cnn1d_with_embeddings_model(max_features, max_len):

	model = Sequential()

	model.add(Embedding(max_features, 8, input_length=max_len))
	model.add(Convolution1D(16, 5)) # filter, kernal size
	model.add(Flatten())
	model.add(Dense(64))
	model.add(Dense(1))
	model.add(Activation('sigmoid')) # since it is binary classifcation (sigmoid activation makes sense)

	return model


def simpleRNN_with_embeddings_model(max_features, max_len):


	model = Sequential()
	model.add(Embedding(max_features, 8, input_length=max_len))
	model.add(SimpleRNN(32))
	# model.add(Flatten())
	model.add(Dense(64))
	model.add(Dense(1))
	model.add(Activation('sigmoid')) # since it is binary classifcation (sigmoid activation makes sense)

	return model



def cnn1d_model(input_shape):

	model = Sequential()
	model.add(Convolution1D(16, 5, input_shape= input_shape))
	# model.add(pooling)
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(64))
	model.add(Dense(1))
	model.add(Activation('softmax')) 

	return model

if __name__ == '__main__':

	# INDEX_FROM=3   # word index offset 
	# number of words are considered as features for embedding
	NUM_WORDS = 10000 # keep top 10,000 most frequently words in train data

	# To train dense with embeddings:
		# 1. Load dataset
		# 2. pad sequences (so that length of each sentence is same)
	# load dataset
	(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=NUM_WORDS)

	maxlen = 20

	#pad sequence turns the list of integers into a 2D integer tensor of shape (samples, maxlen)
	# which is useful for Embedding layer (that returns 3D tensor due to which we have to flatten it 
	# before applying to Dense layer)
	x_train = preprocessing.sequence.pad_sequences(train_data, maxlen=maxlen)
	x_test = preprocessing.sequence.pad_sequences(test_data, maxlen=maxlen)

	print (x_train.shape)
	'''
	To print decoded review (convert numbers to words uncomment this)
	(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)
	# word_to_id = keras.datasets.imdb.get_word_index()
	# # print (word_to_id)


	# # word_index is a dictionary mapping words to an integer index
	# word_to_id = keras.datasets.imdb.get_word_index()
	# # reverse mapping integer indices to words
	# reverse_word_index = dict((value, key) for key, value in word_to_id.items())
	# decode review. Note indices are offset by 3 because 0,1,and 2 are resereved for ndicing, 
	#                         start of sequence and unknown words
	# if key is not present, return '?'
	# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]]) 
	# print (decoded_review)
	
	# v_train_data = vectorize_sequence(train_data, NUM_WORDS)
	# v_test_data = vectorize_sequence(test_data, NUM_WORDS)

	# v_train_labels = np.asarray(train_labels).astype('float32')
	# v_test_labels = np.asarray(test_labels).astype('float32')

	'''



	# input_shape = (10000,50,)
	
	# model = dense_with_embeddings_model(NUM_WORDS, maxlen)
	model = simpleRNN_with_embeddings_model(NUM_WORDS, maxlen)

	model.compile(optimizer=optimizers.rmsprop(), loss='binary_crossentropy',
		         metrics=['accuracy'])

	model.summary()
	history = model.fit(x_train, train_labels, epochs=10, batch_size=32, validation_split=0.2)
	# train_results = model.fit(v_train_data, v_train_labels, batch_size=512, epochs=3, validation_split=0.2)
