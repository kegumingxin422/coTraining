#encoding=utf-8
import numpy as np
import scipy as sp
import re
from numpy import float64
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import linear_model as lm
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import StratifiedKFold
import jieba
from imp import reload
import os
import csv
import codecs
import sys
reload(sys)
import time
class_id = [1, -1]
lang = ["CN", "EN"]
fields = ["book", "dvd", "music"]
#存储测试集中text/summary/data/goal，有EN和CN两种
train_text = {"CN": [], "EN": []}
train_summary = {"CN": [], "EN": []}
train_data = {"CN": [], "EN": []}
train_goal = {"CN": [], "EN": []}
#存储未标注集中text/summary/data/goal，有EN和CN两种
unlabeled_text = {"CN": [], "EN": []}
unlabeled_summary = {"CN": [], "EN": []}
unlabeled_data = {"CN": [], "EN": []}
unlabeled_goal = {"CN": [], "EN": []}
unlabeled_id = {"CN": [], "EN": []}
#存储测试集中text/summary/data/goal，有EN和CN两种
test_text = {"CN": [], "EN": []}
test_summary = {"CN": [], "EN": []}
test_data = {"CN": [], "EN": []}
test_goal = {"CN": [], "EN": []}
test_id = {"CN": [], "EN": []}

rootPath = "E:/OneDrive/语义计算与知识检索/Homework2/新建文件夹/黄梓铭_1501214385_第二次报告"
dataPath = rootPath + "/processData"
trainPath = dataPath + "/Train_EN/"
testPath = dataPath + "/Test_CN/"
unlabeledPath = dataPath + "/Unlabel_CN/"
#向量化
vectorizer = {}
#构建中英两个分类器
clf = {}

def vectorize():
	start_time = time.time()
	for i in range(0, 2):
		vectorizer[lang[i]].fit(train_summary[lang[i]] + train_text[lang[i]])
		#训练数据向量化
		vec_summary = vectorizer[lang[i]].transform(train_summary[lang[i]])
		vec_text = vectorizer[lang[i]].transform(train_text[lang[i]])
		train_data[lang[i]] = sp.sparse.hstack((vec_summary, vec_text))
		#未标注数据向量化
		vec_summary = vectorizer[lang[i]].transform(unlabeled_summary[lang[i]])
		vec_text = vectorizer[lang[i]].transform(unlabeled_text[lang[i]])
		unlabeled_data[lang[i]] = sp.sparse.hstack((vec_summary, vec_text))
	end_time = time.time()
	print ("TfidfVectorize time:", end_time - start_time)

def tokenize(text):
	return [tok.strip() for tok in text.split(" ") if len(tok.strip()) > 1]
#Jieba分词
def jiebaTokenize(text):
	return jieba.cut(text)

#选出可信度最高的前n个正例和前p个负例，将其加入训练集并且从为标注集中删除
def deleteAndUpdateData(data_id, label, lang):
	index = -1
	j = 0
	for unlabel_id in unlabeled_id[lang]:
		if unlabel_id == data_id:
			index = j
			break
		j += 1

	train_text[lang].append(unlabeled_text[lang][index])
	train_summary[lang].append(unlabeled_summary[lang][index])
	train_goal[lang].append(label)

	del unlabeled_text[lang][index]
	del unlabeled_summary[lang][index]
	del unlabeled_id[lang][index]

def predict():
	for i in range(0, 2):
		print (lang[i] + " unlabel predicting ...")
		prob = clf[lang[i]].predict_proba(unlabeled_data[lang[i]])
		#正负例均选择出前5个最可信的样本
		updateData(5,5, prob, lang[i])

def readTrainData(file_name):
	summary = []
	goal = []
	text = []
	# print "reading " + file_name + " ..."
	with codecs.open(file_name, 'r', 'utf-8') as read_file:
		for line in read_file:
			line = line.split('+')
			summary.append(line[0])
			if line[1] == "P":
				goal.append(1)
			else:
				goal.append(-1)
			text.append(line[2])
	return summary, goal, text

def readUnlabeledData(file_name):
	test_id = []
	summary = []
	text = []
	# print "reading " + file_name + " ..."
	with codecs.open(file_name, 'r', 'utf-8') as read_file:
		for line in read_file:
			line = line.split('+')
			test_id.append(line[0])
			summary.append(line[1])
			text.append(line[2])
	return test_id, summary, text

def readTestData(file_name):
	test_id = []
	summary = []
	goal = []
	text = []
	# print "reading " + file_name + " ..."
	with codecs.open(file_name, 'r', 'utf-8') as read_file:
		for line in read_file:
			line = line.split('+')
			test_id.append(line[0])
			summary.append(line[1])
			if line[2] == "P":
				goal.append(1)
			else:
				goal.append(-1)
			text.append(line[3])
	return test_id, summary, goal, text


#构建LR分类器
def classifierTrain():
	for i in range(0, 2):
		print (lang[i] + " classifier training ...")
		# clf[lang[i]] = svm.SVC(probability = True)
		clf[lang[i]] = lm.LogisticRegression()
		clf[lang[i]].fit(train_data[lang[i]], train_goal[lang[i]])


#根据选出的可信度最高的前n个正例和前p个负例，更新训练集和为标注集
def updateData(p, n, prob_list, lang):
	pos_data = []
	neg_data = []
	j = 0
	for prob in prob_list:
		if prob[0] > prob[1]:
			pos_data.append([prob[0], -1, unlabeled_id[lang][j]])
		else:
			neg_data.append([prob[1], 1, unlabeled_id[lang][j]])
		j += 1
	pos_data.sort()
	neg_data.sort()
	for i in range(p):
		if len(pos_data) > i:
			deleteAndUpdateData(pos_data[i][2], pos_data[i][1], lang)
	for i in range(n):
		if len(neg_data) > i:
			deleteAndUpdateData(neg_data[i][2], neg_data[i][1], lang)

#计算正例和负例的precision/recall/F1
def result(prediction):
	k = 0
	right = 0
	t2t,t2f,f2t,f2f = 0,0,0,0
	result = []
	print(len(prediction))
	while k < len(prediction):
		if int(prediction[k]) == int(test_goal["CN"][k]):
			right += 1
			if(prediction[k]==1):
				t2t += 1
			else: 
				f2f += 1
		else:
			if(int(prediction[k])==1):
				t2f += 1
			else:
				f2t += 1
		k += 1
	# 正例计算
	precision1 = float(t2t) / float(t2t+t2f)
	recall1 = float(t2t) / float(t2t+f2t)
	F11 = 2.0 / float(1.0/precision1+1.0/recall1)
	#负例计算
	precision0 = float(f2f) / float(f2f+f2t)
	recall0 = float(f2f) / float(f2f+t2f)
	F10 = 2.0 / float(1.0/precision0+1.0/recall0)
	result.append(precision1)
	result.append(recall1)
	result.append(F11)
	result.append(precision0)
	result.append(recall0)
	result.append(F10)
	return result

def evaluate():
	prob = {}
	predictions = {"CN": [], "EN": []}
	res = []
	for i in range(2):
		vec_summary = vectorizer[lang[i]].transform(test_summary[lang[i]])
		vec_text = vectorizer[lang[i]].transform(test_text[lang[i]])
		test_data = sp.sparse.hstack((vec_summary, vec_text))

		prob[lang[i]] = clf[lang[i]].predict_proba(test_data)
		for proba in prob[lang[i]]:
			if proba[0] > proba[1]:
				predictions[lang[i]].append(-1 * proba[0])
			else:
				predictions[lang[i]].append(1 * proba[1])

	for i in range(len(prob["CN"])):
		#中文和英文分类器协同过滤
		temp_res = predictions["CN"][i] + predictions["EN"][i]
		if temp_res > 0:
			res.append(1)
		elif temp_res < 0:
			res.append(-1)
		else:
			res.append(0)
	return res, result(res)

#协同训练
def coTraining(stopword):
	vectorizer["EN"] = TfidfVectorizer(min_df = 4, ngram_range = (1, 2), stop_words = set(stopword), tokenizer = tokenize)
	vectorizer["CN"] = TfidfVectorizer(min_df = 4, ngram_range = (1, 2), stop_words = set(stopword), tokenizer = jiebaTokenize)
	vectorize()
	classifierTrain()
	predict()
	res, result = evaluate()
	return res,result


if __name__ == '__main__':
	# entk = tokenize("I love you!");
	# cntks = jiebaTokenize("独立音乐需要大家一起来推广，欢迎加入我们的行列！");
	# print(entk)
	# for cntk in cntks:
	# 	print(cntk)
	I = 40	#迭代次数
	stopword = [word.strip() for word in open("stopword.txt", "r")]
	out_file = open(rootPath + "/result.txt", 'w')

	for i in range(0, 3):
		print (fields[i] + " start: ")
		out_file.write(fields[i] + 'start:' + "\r\n")
		
		train_summary["EN"], train_goal["EN"], train_text["EN"] = readTrainData(trainPath + fields[i] + "/en_processData.data")
		train_summary["CN"], train_goal["CN"], train_text["CN"] = readTrainData(trainPath + fields[i] + "/cn_translateData")
		unlabeled_id["EN"], unlabeled_summary["EN"], unlabeled_text["EN"] = readUnlabeledData(unlabeledPath + fields[i] + "/en_translateData")
		unlabeled_id["CN"], unlabeled_summary["CN"], unlabeled_text["CN"] = readUnlabeledData(unlabeledPath + fields[i] + "/cn_processData.data")
		test_id["EN"], test_summary["EN"], test_goal["EN"], test_text["EN"] = readTestData(testPath + fields[i] + "/en_translateData")
		test_id["CN"], test_summary["CN"], test_goal["CN"], test_text["CN"] = readTestData(testPath + fields[i] + "/cn_processData")
		starttime = time.time()
		for j in range(I):
			print ('iterator' + str(j) + " : ")
			#协同训练
			res,acc = coTraining(stopword)
			out_file.write('iterator' + str(j) + " : " + "\r\n")
			print ("result: " + str(acc[0])+","+str(acc[1])+","+str(acc[2])+","+str(acc[3])+","+str(acc[4])+","+str(acc[5]))
			# out_file.write("正例: " + str(acc[0])+","+str(acc[1])+","+str(acc[2]) + "\r\n")
			# out_file.write("负例: " + str(acc[3])+","+str(acc[4])+","+str(acc[5]) + "\r\n")
			avg = (acc[0]+acc[3]) / 2.0
			out_file.write(str(avg)+"\r\n")
		endtime = time.time()
		print ("Every item time:", endtime - starttime)
	out_file.close()