# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:02:40 2022

@author: Kid
"""

import numpy as np
#import tensorflow as tf

from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn import metrics
import matplotlib.pyplot as plt

base_fpr = np.linspace(0, 1, 100001)


def Find_EER2(far, tpr):
	eer = brentq(lambda x : 1. - x - interp1d(far, tpr)(x), 0., 1.)
	#fnr = 1 - tpr
	#eer=far[np.nanargmin(np.absolute((fnr - far)))]
	#eer = brentq(lambda x : 1. - x - far[int(x*100000)], 0., 1.)
   #print("index of min difference=", y)
	optimum = eer
	return optimum

def EERf(resutls):
  #print(resutls)
  resutls = np.array(resutls)
  y = np.array(resutls[:,1])
  scores = np.array(resutls[:,0])
  fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
  ntpr = np.interp(base_fpr, fpr, tpr)
  ntpr[0] = 0.0
  return Find_EER2(base_fpr, ntpr),1-ntpr[1000]



def assessment_model(resutls):
	resutls2 = np.array(resutls)
	y = np.array(resutls2[:,1])
	scores = np.array(resutls2[:,0])
	fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
	ntpr = interp1d(fpr, tpr)(base_fpr)
	nfpr = interp1d(fpr, tpr)(base_fpr)
	#ntpr = np.interp(base_fpr, fpr, tpr)
	#nfpr = np.interp(base_fpr, tpr, fpr)
	ntpr[0] = 0.0
	nfpr[0] = 0.0
	auc=metrics.auc(fpr, tpr)
	print(Find_EER2(base_fpr, ntpr),1-ntpr[1000])
	return Find_EER2(base_fpr, ntpr),1-ntpr[1000],nfpr[1],ntpr,auc
 

#Please change dataset offset (d1,d2,d3)


# d1 , d2 , d3
task="d1"




import pickle

with open('./Similarity Scores/'+task+'_dicr1.pkl', 'rb') as f:
    dicr1=pickle.load(f)

with open('./Similarity Scores/'+task+'_dicr2.pkl', 'rb') as f:
    dicr2=pickle.load(f)

with open('./Similarity Scores/'+task+'_dicr3.pkl', 'rb') as f:
    dicr3=pickle.load(f)


	
Y=np.load('./Similarity Scores/'+task+'Y.npy')





title=task+"t3"
plt.clf()
sum1=0
av_eer=0
av_eerb=0
eer3=[]
for i in dicr3.keys():
	temp,b=EERf(dicr3[i])
	av_eerb+=b
	eer3.append(temp)
	av_eer=temp+av_eer
	print(i,EERf(dicr3[i]))
print("EER avrage:",av_eer/len(dicr3),"FaF avrage:",av_eerb/len(dicr3))
font = {'size'   : 14}
plt.rc('font', **font)
plt.figure(figsize=(9, 5))
names = list(dicr3.keys())
values = eer3
plt.xticks(names,rotation='vertical')
plt.ylabel('EER')
plt.xlabel("Subjects")
plt.bar(names, values)
plt. savefig("./Plots/"+title+"_bar.pdf",bbox_inches='tight')







title=task+"t3"
plt.clf()
sum1=0
av_eer=0
av_eerb=0
eer3=[]


plt.clf()
tprs=[]
aucs=[]
aveer=0
aveerb=0
for i in dicr3.keys():
	a,b,c,d,auc=assessment_model(dicr3[i])
	aveer=aveer+a
	aveerb+=b
	tprs.append(d)
	aucs.append(auc)
print("EER avrage:",aveer/len(dicr3),"FaF avrage:",aveerb/len(dicr3))




plt.clf()
tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)

plt.figure(figsize=(8, 6))

plt.plot(base_fpr[0:5000]*100, (1-mean_tprs)[0:5000]*100, 'b')

newx=list(plt.xticks()[0]) + [base_fpr[1]*1000,0.5]
newxtemp=newx.copy()
i=0
for ii in newxtemp:
	#print(ii, ii<=0.0)
	if ii<=0.0 or ii>5:
		newx.remove(ii)
		
for i in range(len(newx)):
	if newx[i]>=1:
		newx[i]=int(newx[i])
#newx=newx*100
newxm = map(str, newx)
plt.xticks(newx,newxm)
#,rotation='vertical'


newy=list(plt.yticks()[0]) + [0]
newy=list(np.arange(0,15.5,0.5))



font = {'size'   : 20}
plt.rc('font', **font)

plt.plot([base_fpr[10]*100,0.50,1,5],[(1-mean_tprs)[10]*100,(1-mean_tprs)[500]*100,(1-mean_tprs)[1000]*100,(1-mean_tprs)[5000]*100], ls="", marker="o", label="points",color="red")
plt.ylabel('False Rejection Rate (%)')
plt.xlabel('False Acceptance Rate (%)')
plt.yticks(newy)
plt.ylim(ymin=0,ymax=3)
plt.xlim(xmin=-0.02,xmax=5.05)
plt.grid()
#plt.show()
plt. savefig("./Plots/"+title+"usability.pdf",bbox_inches='tight')






print("========================")

counti=[]
count_all=0
id_result=np.array([])
for i in dicr2.values():
    tempi=np.array(i.copy())
    at=0
    group=[]
    groupt=[]
    ttemp=tempi[0,2]
    for j in tempi:
        if ttemp==j[2]:
            groupt.append(j[0:2])
        else:
            group.append(groupt)
            groupt=[]
            groupt.append(j[0:2])
            ttemp=j[2]
    for c in group:
        c=np.array(c)
        temp=c[c[:, 0].argsort()]
        counti.append(temp[-1,1])

print("last iditification acurrecy top one from 5 s1: ", sum(counti)/len(counti))

print("========================")


print("========================")

counti=[]
count_all=0
id_result=np.array([])
for i in dicr1.values():
	tempi=np.array(i.copy())
	base=-1
	for c in np.unique(tempi[:,2]):
		base+=np.count_nonzero(Y==c)
	#print("make sure",len(i)/base)
	count_all=count_all+(len(i)//base)
	for j in range(len(i)//base):
		temp=tempi[0:base,:]
		tempi=tempi[base:,:]
		temp=temp[temp[:, 0].argsort()]
		if temp[-1,2]==temp[-1,3]:
			counti.append(1)
		else:
			#print("==================")
			#print(temp[base-5:base,:])
			pass
print("iditification acurrecy top one from 5 s1: ", sum(counti)/count_all)

print("========================")



print("========================")

TP=[]
TN=[]
FP=[]
FN=[]

count_all=0
id_result=np.array([])
aveer=0
for i in dicr1.values():
	f=[]
	tempi=np.array(i.copy())
	base=-1
	for c in np.unique(tempi[:,2]):
		base+=np.count_nonzero(Y==c)
	#print("make sure",len(i)/base)
	count_all=count_all+(len(i)//base)
	for j in range(len(i)//base):
		temp=tempi[0:base,:]
		tempi=tempi[base:,:]
		temp=temp[temp[:, 0].argsort()]
		tempv,tempv1,tempv2,counter3 =True,True,True,-1
		while(tempv):
			ttemp=temp[counter3]
			if temp[counter3,2]==temp[counter3,3] and tempv1:
				f.append([ttemp[0],1])
				tempv1=False

			if temp[counter3,2]!=temp[counter3,3] and tempv2:
				#print("I am here")
				f.append([ttemp[0],0])
				tempv2=False
			counter3=counter3-1
			if tempv1==False and tempv2==False:
				tempv=False
	a,b,c,d,auc=assessment_model(f)
	aveer=aveer+a
			#print("==================")
			#print(temp[base-5:base,:])
print("open-set iditification-verification acurrecy top one from 5 s1: ", aveer/8)


counti=[]
count_all=0
id_result=np.array([])
for i in dicr1.values():
	tempi=np.array(i.copy())
	base=-1
	for c in np.unique(tempi[:,2]):
		base+=np.count_nonzero(Y==c)
	#print("make sure",len(i)/base)
	count_all=count_all+(len(i)//base)
	for j in range(len(i)//base):
		temp=tempi[0:base,:]
		tempi=tempi[base:,:]
		temp=temp[temp[:, 0].argsort()]
		if temp[-1,2]==temp[-1,3] or temp[-2,2]==temp[-2,3]:
			counti.append(1)
		else:
			#print("==================")
			#print(temp)
			pass
print("iditification acurrecy top two from 5 s1: ", sum(counti)/count_all)


counti=[]
count_all=0
id_result=np.array([])
for i in dicr2.values():
	base=0
	#print("make sure",len(i)/5)
	tempi=np.array(i.copy())
	count_all=count_all+(len(i)//5)
	for j in range(len(i)//5):
		temp=tempi[0:5,:]
		tempi=tempi[5:,:]
		temp=temp[temp[:, 0].argsort()]
		if temp[4,2]==temp[4,3]:
			counti.append(1)
			
print("iditification acurrecy top one from 5 s2: ", sum(counti)/count_all)

counti=[]
count_all=0
id_result=np.array([])
for i in dicr2.values():
	base=0
	#print("make sure",len(i)/5)
	tempi=np.array(i.copy())
	count_all=count_all+(len(i)//5)
	for j in range(len(i)//5):
		temp=tempi[0:5,:]
		tempi=tempi[5:,:]
		temp=temp[temp[:, 0].argsort()]
		if temp[4,2]==temp[4,3] or temp[3,2]==temp[3,3]:
			counti.append(1)
		else:
			#print("==================")
			#print(temp)
			pass
print("iditification acurrecy top two from 5 s2: ", sum(counti)/count_all)





