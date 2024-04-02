# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:50:04 2021

@author: a Kid
"""

from tabnanny import verbose
import numpy as np
from random import random, seed
import tensorflow as tf
import tensorflow_addons as tfa

X=np.load('./Data/D_3_X.npy')
Y=np.load('./Data/D_3_Y.npy')
digit_indices = [np.where(Y == i)[0] for i in np.unique(Y)]
calss=np.unique(Y)
same_in=digit_indices[np.where(calss == 43)[0][0]]
Y = np.delete(Y, same_in, 0)
X = np.delete(X, same_in, 0)

np.random.seed(999)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

X,Y=unison_shuffled_copies(X,Y)




def bmodels():
  activef="selu"
  batch_size = 64
  
  chn=32
  sn=513

  input = tf.keras.layers.Input((chn, sn, 1))
  #x = tf.keras.layers.BatchNormalization()(input)
  x = tf.keras.layers.Conv2D(128, (1, 15), activation=activef, kernel_initializer='lecun_normal')(input)
  x = tf.keras.layers.AveragePooling2D(pool_size=(1, 2))(x)
  x = tf.keras.layers.Dropout(0.3)(x)
  #x = keras.layers.MaxPooling2D(pool_size=(1, 4))(x)
  x = tf.keras.layers.Conv2D(32, (1, 15), activation=activef, kernel_initializer='lecun_normal')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=(1, 2))(x)
  x = tf.keras.layers.Dropout(0.3)(x)
  #x = keras.layers.AveragePooling2D(pool_size=(1, 5))(x)
  x = tf.keras.layers.Conv2D(16, (1, 15), activation=activef, kernel_initializer='lecun_normal')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=(1,2))(x)
  x = tf.keras.layers.Dropout(0.3)(x)
  # = keras.layers.AveragePooling2D(pool_size=(1,5))(x)

  x = tf.keras.layers.Conv2D(8, (1, 15), activation=activef, kernel_initializer='lecun_normal')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=(1,2))(x)
  x = tf.keras.layers.Dropout(0.3)(x)

  x = tf.keras.layers.Conv2D(4, (1, 15), activation=activef, kernel_initializer='lecun_normal')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=(1,2))(x)
  x = tf.keras.layers.Dropout(0.3)(x)


  x = tf.keras.layers.Flatten()(x)
  #x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras. layers.Dense(32, activation=None, kernel_initializer='lecun_normal')(x)
  #x = tf.keras.layers.BatchNormalization()(x)
  embedding_network = tf.keras.Model(input, x, name="Embedding")
  embedding_network.compile(
    optimizer=tf.keras.optimizers.Adam(),
	#0.001, clipnorm=1.
    loss=tfa.losses.TripletSemiHardLoss(margin=1.0))
  #siamese.summary()
  return embedding_network



def assessment_model_data():
	from sklearn.metrics.pairwise import cosine_similarity as cs


	resutls=[]
	resutls2=[]
	from collections import defaultdict
	resutls3=defaultdict(list)
	pair1=[]
	pair2=[]
	calss=np.unique(y_test)
	TP,FP,TN,FN=0,0,0,0
	digit_indices = [np.where(y_test == i)[0] for i in np.unique(y_test)]
	x_test_1 = x_test
	print(len(x_test))
	#anc_et=embedding_network(x_train_val)
	anc_e=embedding_network(x_test[0:min(500,len(x_test))])
	for c in range(len(x_test)//500):
		anc_e=tf.concat(axis=0, values = [anc_e, embedding_network(x_test[(c+1)*500:min((c+2)*500,len(x_test))])]) 	
	print(len(x_test))
	#anc_e=embedding_network(x_test_1[0:600])
	#anc_e2=embedding_network(x_test_1[600:1100])
	#anc_e3=embedding_network(x_test_1[1100:])
	#anc_e=tf.concat(axis=0, values = [anc_e, anc_e2]) 
	#anc_e=tf.concat(axis=0, values = [anc_e, anc_e3]) 
	#print(type(anc_e))
	for i in range(len(x_test_1)):
		#x_test_t,y_test_t= x_test.copy(),y_test.copy()
		temp=np.where(calss == y_test[i])[0][0]
		#pairs_test, labels_test = autinitcation_same2(x_val2, y_val2,x_val3[i],y_val3[i])
		#pairs_test=scaler.transform(pairs_test.reshape(-1, pairs_test.shape[-1])).reshape(pairs_test.shape)
		#tm=[x_test[i]]*len(x_test)
		#tm=np.array(tm)
		#x_test_2 = tm
		prediction=[]
		#test_e=embedding_network(np.array([x_test[i]]))
		#prediction=cs(anc_e,tm2)
		#pair1.append(test_e)
		#pair2.append(anc_e)
		#cosine_similarity = metrics.CosineSimilarity()
		same_in=digit_indices[np.where(calss == y_test[i])[0][0]]
		
		for t in range(len(x_test_1)):
			#cosine_similarity = metrics.CosineSimilarity()
			#tempp=cosine_similarity(anc_e[t],test_e).numpy()
			#tempp=cs(anc_e[t].numpy().reshape(1, -1),test_e.numpy().reshape(1, -1))[0][0]
			#tempp=1000-euclidean_distance2(anc_e[t],test_e).numpy()[0][0]
			#print(euclidean_distance2(anc_e[t],anc_e[i]).numpy())
			tempp=-1*euclidean_distance2(anc_e[t],anc_e[i]).numpy()[0]
			#print(tempp)
	#resutls = np.array(resutls)


			if t in same_in:
				if t==i:
					pass
				else:
					resutls.append([tempp,1,y_test[i],y_test[t]])
			else:
				resutls.append([tempp,0,y_test[i],y_test[t]])
				
				
			prediction.append(tempp)
		
		
		prediction=np.array(prediction)
		
		for j in calss:
			same_in=digit_indices[np.where(calss == j)[0][0]]
			#print(np.where(calss == j)[0][0],"/n",same_in,"/n",digit_indices)
			
			same_in=np.setdiff1d(same_in,[i])
		
			#same_in=remove_outlier(same_in)
			#spredict=((sum(prediction[same_in]))/(len(same_in)))
			spredict=max(prediction[same_in])
			#print((y_test[i] ==j),j,y_test[i],spredict)
		
			
			if y_test[i] ==j:
				resutls2.append([spredict,1,y_test[i],j])
				resutls3[j].append([spredict,1,y_test[i],j])
			else:
				resutls2.append([spredict,0,y_test[i],j])
				resutls3[j].append([spredict,0,y_test[i],j])
				
				
			if spredict>0.85:
				if y_test[i] ==j:
					TP+=1
				else:
					FP+=1
			else:
				if y_test[i] == j:
					FN+=1
				else:
					TN+=1
		#print("TPR:",TP/(TP+FN)," FPR:",FP/(FP+TN))
		#print("i:",i,"Sum:",TP+FN+TN+FP)
	return resutls,resutls2,resutls3


from scipy.interpolate import interp1d
from scipy.optimize import brentq


def euclidean_distance2(x, y):
	sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=None, keepdims=True)
	return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def Find_EER2(far, tpr):
	eer = brentq(lambda x : 1. - x - interp1d(far, tpr)(x), 0., 1.)
   #print("index of min difference=", y)
	optimum = eer
	return optimum

def EERf(resutls):
  #print(resutls)
  resutls = np.array(resutls)
  y = np.array(resutls[:,1])
  scores = np.array(resutls[:,0])
  fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, scores, pos_label=1)
  ntpr = interp(base_fpr, fpr, tpr)
  ntpr[0] = 0.0
  return Find_EER2(fpr, tpr),1-ntpr[1]

import sklearn
from scipy import interp

def assessment_model(resutls):
	resutls2 = np.array(resutls)
	y = np.array(resutls2[:,1])
	scores = np.array(resutls2[:,0])
	fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, scores, pos_label=1)
	ntpr = interp(base_fpr, fpr, tpr)
	nfpr = interp(base_fpr, tpr, fpr)
	ntpr[0] = 0.0
	nfpr[0] = 0.0
	print(Find_EER2(fpr, tpr),1-ntpr[100])
	return Find_EER2(fpr, tpr),1-ntpr[100],nfpr[1],ntpr



base_fpr = np.linspace(0, 1, 10001)

import tensorflow_addons as tfa
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from sklearn.model_selection import GroupKFold, StratifiedKFold
dicr3={}
dicr2={}
dicr1={}
Y1=Y.copy()
X1=X.copy()

calss=np.unique(Y)
digit_indices = [np.where(Y == i)[0] for i in np.unique(Y)]
#same_in=digit_indices[np.where(calss == 43)[0][0]]
#Y = np.delete(Y, same_in, 0)
#X = np.delete(X, same_in, 0)

group_kfold = GroupKFold(n_splits=8)
#skf = StratifiedKFold(n_splits=8)
count_cv=0
flag=0
for train_index, test_index in group_kfold.split(X, Y, groups=Y):
#for train_index, test_index in skf.split(X, Y, groups=Y):
  flag+=1
  if flag ==400:
    continue
  print(flag)
  x_train_val, x_test, y_train_val, y_test =X[train_index],X[test_index],Y[train_index],Y[test_index]
  print(np.unique(y_train_val, return_counts=True),np.unique(y_test, return_counts=True),np.unique(Y, return_counts=True))
  x_test = x_test.astype("float32")
  tf.keras.backend.clear_session()
  model= bmodels()
  embedding_network=model
  #for i in range(30):
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train_val, y_train_val)).shuffle(1000).batch(128)
  history = embedding_network.fit(
      train_dataset,
      workers=5,
      epochs=250,
	  verbose=1)
  resutls,resutls2,resutls3=assessment_model_data()
  dicr1[count_cv] = resutls
  dicr2[count_cv] = resutls2
  dicr3.update(dict(resutls3))

  count_cv=count_cv+1

#x_train1 = x_train1.astype("float32")
import pickle
with open('./Similarity Scores/d3_dicr1.pkl', 'wb') as f:
    pickle.dump(dicr1, f)

with open('/Similarity Scores/d3_dicr2.pkl', 'wb') as f:
    pickle.dump(dicr2, f)

with open('./Similarity Scores/d3_dicr3.pkl', 'wb') as f:
    pickle.dump(dicr3, f)
