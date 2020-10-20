import argparse
import time
import math
import os
import torch
import torch.utils.data
import numpy as np

parser = argparse.ArgumentParser(description='downstream link prediction task')
parser.add_argument('--inf', default='models/VANE_link_pred_n2v_model', help='folder for the tested model')
args = parser.parse_args()

MODE = 'VANE'

nodepair2labels_test = {}
with open('ground_truth/link_labels_test.txt','r',encoding='utf-8')as fin:
	for i in fin:
		i = i.strip().split('\t')
		nodepair2labels_test[(int(i[0]),int(i[1]))] = int(i[2])

nodepair2labels_val = {}
with open('ground_truth/link_labels_val.txt','r',encoding='utf-8')as fin:
	for i in fin:
		i = i.strip().split('\t')
		nodepair2labels_val[(int(i[0]),int(i[1]))] = int(i[2])

trainedNodes = set()
with open('walks/n2v(link_pred)_walks.txt','r',encoding='utf-8')as fin:
	for i in fin:
		trainedNodes.add(int(i.strip().split(' ')[1].split(',')[0]))

def cosineSimilarity(x1,x2):
	return x1.dot(x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))


if MODE == 'VANE':
	best_on_validation = 0.00
	best_filename = None
	for filename in os.listdir(args.inf):
		if 'Extractor' in filename:
			filename = args.inf+'/'+filename
			model = torch.load(filename)
			feature = model['encoder.weight'].data.numpy()
			total = 0
			hit = 0
			for i in nodepair2labels_val:
				label = nodepair2labels_val[i]
				if i[0] not in trainedNodes or i[1] not in trainedNodes: continue
				result = cosineSimilarity(feature[i[0]],feature[i[1]])

				if result > 0.5:
					result = 1
				else:
					result = 0

				total+=1
				if result == label:hit+=1
			acc = hit/total
			if acc > best_on_validation:
				best_on_validation = acc
				best_filename = filename
	print(best_filename)

	model = torch.load(best_filename)
	feature = model['encoder.weight'].data.numpy()
	total = 0
	hit = 0
	for i in nodepair2labels_val:
		label = nodepair2labels_val[i]
		if i[0] not in trainedNodes or i[1] not in trainedNodes: continue
		result = cosineSimilarity(feature[i[0]],feature[i[1]])

		if result > 0.5:
			result = 1
		else:
			result = 0

		total+=1
		if result == label:hit+=1
	acc = hit/total
	print("Accuracy:"+str(acc))
