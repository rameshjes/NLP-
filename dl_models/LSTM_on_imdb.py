import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models, optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Embedding, LSTM
import numpy as np
from imdb_with_Glove import tokenize_data, create_embedding_matrix, \
						collect_reviews_and_label, tokenize_test_data, parse_glove

# LSTM has ability to fight with vanishing gradient problem (they suffers but very less as compare to simple RNN)

def lstn_with_embeddings_model(max_words, max_len, embedding_dim):

	model = Sequential()

	model.add(Embedding(max_words, embedding_dim, input_length=max_len))
	model.add(LSTM(16)) # filter, kernal size
	# model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid')) # since it is binary classifcation (sigmoid activation makes sense)

	return model
if __name__ == '__main__':
	maxlen = 100
	training_samples = 15000 # we can take more (for demo it is fine)
	validation_samples = 10000
	max_words = 10000 # top 10k words in dataset
	imdb_dir = './aclImdb'
	train_data_path = os.path.join(imdb_dir, 'train')
	test_data_path = os.path.join(imdb_dir, 'test')
	texts, labels = collect_reviews_and_label(train_data_path)
	print ("total reviews ", len(texts))
	print ("total labels ", len(labels))

	(word_index, tokenizer, x_train, y_train, x_val, y_val) = tokenize_data(texts, labels, max_words, 
													maxlen, training_samples, validation_samples)

	embeddings_index = parse_glove()
	(embedding_matrix, embedding_dim) = create_embedding_matrix(max_words, embeddings_index, word_index)

	model = lstn_with_embeddings_model(max_words, maxlen, embedding_dim)

	# add pre-trained embeddings
	model.layers[0].set_weights([embedding_matrix])
	model.layers[0].trainable = False # freeze embedding layer
	model.compile(optimizer=optimizers.rmsprop(), loss='binary_crossentropy',
		         metrics=['accuracy'])

	model.summary()
	# print( "x train shape ",x_train.shape)
	# print("y train shape ", y_train.shape)
	history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
	model.save_weights('lstm_using_pre_trained_glove_model.h5') 
	# We can also learn embeddings on imdb dataset(rather than glove)
	# Then we can just skip create_embedding matrix function 

	# Evalute on test dataset  
	texts, labels = collect_reviews_and_label(test_data_path)
	test_data, test_labels = tokenize_test_data(tokenizer, texts, labels, maxlen)
	model.load_weights('cnn1d_using_pre_trained_glove_model.h5')
	loss, accuracy = model.evaluate(test_data, test_labels)
	print ("loss ", loss)
	print ("accuracy ", accuracy)

	