import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from bert_serving.client import BertClient
import numpy as np
import gzip
import pickle

def load_data(data_path):
    
	return pd.read_csv(data_path)

train = load_data("../question-pairs-dataset/test_set.csv")

# train, test = train_test_split(data, test_size=0.3, random_state=42)



claims = train["question1"].tolist()
sents = train["question2"].tolist()

sampled_claims = claims[:20000]
sampled_sents = sents[:20000]

bc = BertClient(check_length=False, check_version=False)

# print (type(claims[0:10]))
sents_pair = [[str(claim)+' ||| '+str(sent)] for claim,sent in zip(sampled_claims,sampled_sents)]
# print (sents_pair[30])
print (len(sents_pair))
vec = np.empty((len(sents_pair), 768))
print (vec.shape)

count = 0
for sent in sents_pair:
    
	if count == 0:
		vec = bc.encode(sent)
	else:
		vec = np.vstack((vec, bc.encode(sent)))
	    
	if count % 100 == 0:
		print ("count ", count)
	count += 1


def save_dataset_and_compress(dataset_dict, name):
    with gzip.GzipFile(name + '.pgz', 'w') as f:
        pickle.dump(dataset_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
save_dataset_and_compress(vec, "/scratch/kkuma12s/quora_embeddings/sampled_quora_bert_embeddings_test")

