from keras.models import load_model
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from keras.utils import np_utils
import pickle
import numpy as np
import gzip
import jsonlines
import pandas as pd

class testModel:

	def __init__(self, dataset_name, model_path, num_classes):

		self.data_name = dataset_name
		self.model_path = model_path
		self.num_classes = num_classes

		# test_data = "/home/kkuma12s/thesis/Proof_Extraction/data/fever-full/claim_classification/"+dataset_name+".jsonl"


		# self.test_data = pickle.load(open("./bert/datasets/test_"+str(self.data_name)+".pkl", "rb"))
		test_data = pd.read_csv("/home/kkuma12s/github/sentence_similarity/question-pairs-dataset/test_set.csv")
		labels = test_data["is_duplicate"]

		self.test_data = test_data[:20000]
		self.labels = labels[:20000]

	def get_results_on_test_data(self):

		with gzip.open("/scratch/kkuma12s/quora_embeddings/sampled_quora_bert_embeddings_test.pgz", 'rb') as f:
			claims_sents_vec = pickle.load(f)

		labels = np.asarray(self.labels)

		model = load_model(self.model_path)
		batch = 128

		if self.num_classes == 3:
			labels = np_utils.to_categorical(labels, self.num_classes)
			avg = 'weighted'
		else:
			avg = 'binary'

		loss, accuracy = model.evaluate({'claims':claims_sents_vec},
		 									labels)
		print ("test loss ", loss)
		print ("test accuracy ", accuracy)
		y_pred = (np.asarray(model.predict({'claims': claims_sents_vec} , batch_size=batch))).round()
		print ("score of lstm ", precision_recall_fscore_support(labels, y_pred, average=avg))


		with open("bert_test_results.log", "w") as f:
			f.write("test loss "+str(loss))
			f.write("\r\n test accuracy "+ str(accuracy))
			f.write("\r\n score of lstm "+ str(precision_recall_fscore_support(labels, y_pred, average=avg)))


