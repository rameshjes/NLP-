from keras.models import load_model
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from keras.utils import np_utils
import pickle
import numpy as np
import gzip
import jsonlines
import pandas as pd
from bert_serving.client import BertClient


def get_embeddings(s1, s2):

	count = 0
	bc = BertClient(check_length=False, check_version=False)

	sents_pair = [[str(claim)+' ||| '+str(sent)] for claim,sent in zip([s1],[s2])]
	vec = np.empty((len(sents_pair), 768))

	for sent in sents_pair:
		if count == 0:
			vec = bc.encode(sent)
		else:
			vec = np.vstack((vec, bc.encode(sent)))
		    
		if count % 100 == 0:
			print ("count ", count)
		count += 1

	return vec





if __name__ == '__main__':
	
	s1 = "I love to eat pizza"
	s2 = "pizza is amazing"
	model = load_model("../trained_models/model_bert_classifierAcc847.h5")
	claims_sents_vec = get_embeddings(s1, s2)
	y_pred = model.predict({'claims': claims_sents_vec})
	print ("y pred ", y_pred)