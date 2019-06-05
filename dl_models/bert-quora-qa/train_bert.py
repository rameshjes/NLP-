import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import numpy as np
import argparse

import jsonlines
import tensorflow as tf
import pandas as pd
import gzip
import keras
from keras import models, optimizers, Input, regularizers
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Flatten, Embedding, concatenate, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, CSVLogger
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.utils import np_utils

from bert-quora-qa.test_model import testModel 

config = tf.ConfigProto( device_count = {'GPU': 1} ) 
#config = tf.ConfigProto(log_device_placement=True)
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

class Metrics(Callback):

	def on_train_begin(self, logs={}):

		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
	 
	def on_epoch_end(self, epoch, logs={}):

		val_predict = (np.asarray(model.predict([claims_sents_vec]))).round()
		val_targ = labels
		_val_f1 = f1_score(val_targ, val_predict, average = 'weighted')
		_val_recall = recall_score(val_targ, val_predict, average ='weighted')
		_val_precision = precision_score(val_targ, val_predict,average = 'weighted')
		self.val_f1s.append(_val_f1)
		self.val_recalls.append(_val_recall)
		self.val_precisions.append(_val_precision)
		print (' — val_f1: %f — val_precision: %f — val_recall %f' %( _val_f1, _val_precision, _val_recall))
		return

class TrainModel:

    def lstm_model(self, claim_length, embedding_dim, nb_classes):


        claims_input = Input(shape=(claim_length, embedding_dim), dtype='float32', name='claims')
        encoded_claims = LSTM(512, recurrent_dropout=0.3, dropout=0.3)(claims_input)
        encoded_claims = Dropout(0.4)(encoded_claims)
        encoded_claims = Dense(512, kernel_regularizer=regularizers.l2(0.001), activation='relu')(encoded_claims)
        encoded_claims = Dropout(0.4)(encoded_claims)

        if nb_classes == 3:
            pred_label = Dense(nb_classes, activation='softmax')(encoded_claims)
        else:
            pred_label = Dense(1, activation='sigmoid')(encoded_claims)

        return claims_input, pred_label


if __name__ == '__main__':


    training = TrainModel()
    metrics = Metrics()
    
    max_seq_length = 30

    nb_classes = 2

    with gzip.open("/scratch/kkuma12s/quora_embeddings/sampled_quora_bert_embeddings_train"+".pgz", 'rb') as f:
            claims_sents_vec = pickle.load(f)

    # only top 80k to train faster 
    train_data = pd.read_csv("/home/kkuma12s/github/sentence_similarity/question-pairs-dataset/train_set.csv")
    labels = train_data["is_duplicate"]

    train_data = train_data[:80000]
    labels = labels[:80000]

    if nb_classes == 3:
        labels = np_utils.to_categorical(labels, nb_classes)
        loss = 'categorical_crossentropy'

    else:
        loss = 'binary_crossentropy'

    embedding_dim = 768
    claims_input, pred_label = training.lstm_model(max_seq_length, embedding_dim, nb_classes)

    model = Model([claims_input], pred_label)
    print (model.summary())

    # print ("claim sents ndim ", claims_sents_vec.shape)
    # # print (sents_data.ndim)
    print ("labels ndim ", labels.ndim)

    # print (claims_sents_vec.shape)
    # # print (labels.shape)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.compile(optimizer=optimizers.Adam(), loss=loss, metrics=['accuracy'])
    csv_logger = CSVLogger("bert_training.log")
    model_path = 'model_bert_q_classifier.h5'
    history = model.fit({'claims': claims_sents_vec}, labels, 
                                epochs=60, batch_size=64, validation_split=0.12, callbacks=[early_stopping, metrics, csv_logger,
                                                ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)])

    test_lstm_model = testModel(test_dataset_name, model_path, nb_classes)
    test_lstm_model.get_results_on_test_data()
