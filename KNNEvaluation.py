import argparse
import time
import math
import os
import torch
import torch.utils.data
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

parser = argparse.ArgumentParser(description='downstream classification task')
parser.add_argument('--inf', default='models/VANE_node_pred_n2v_model', help='folder for the tested model')
args = parser.parse_args()


MODE = 'VANE'

id2domain_val = {}
with open('ground_truth/node_labels_val.txt','r',encoding='utf-8')as fin:
	for i in fin:
		i = i.strip().replace(',',':').split(':')
		id2domain_val[int(i[0])] = int(i[1])

id2domain_test = {}
with open('ground_truth/node_labels_test.txt','r',encoding='utf-8')as fin:
	for i in fin:
		i = i.strip().replace(',',':').split(':')
		id2domain_test[int(i[0])] = int(i[1])

if MODE == 'VANE':

	best_on_validation = 0.00
	best_filename = None
	for filename in os.listdir(args.inf):
		if 'Extractor' in filename:
			filename = args.inf+'/'+filename
			model = torch.load(filename)
			feature = model['encoder.weight'].data.numpy()

			final_results = []
			for _ in range(100):
				neigh = KNeighborsClassifier(n_neighbors=1)
				x = []
				y = []
				yy = list(range(850))

				for i in range(len(yy)):
					if yy[i] not in id2domain_val:continue
					x.append(feature[i])
					y.append(id2domain_val[yy[i]])

				permutation = np.random.permutation(len(x))
				x = [x[i] for i in permutation]
				y = [y[i] for i in permutation]


				numNodes = len(x)
				testingsetlength = numNodes//10
				neigh.fit(x[testingsetlength:], y[testingsetlength:])
				total = len(y[:testingsetlength])
				results = neigh.predict(x[:testingsetlength])
				hit = 0
				for i in range(total):
					if y[:testingsetlength][i]==results[i]:
						hit+=1
				final_results.append(hit/total)
			# print("mean: "+str(np.mean(final_results)))
			# print("str: "+str(np.std(final_results)))
			if np.mean(final_results)>best_on_validation:
				best_on_validation = np.mean(final_results)
				best_filename = filename

	print(best_filename)

	final_results = []
	for _ in range(50):
		neigh = KNeighborsClassifier(n_neighbors=1)
		model = torch.load(best_filename)
		feature = model['encoder.weight'].data.numpy()
		x = []
		y = []
		yy = list(range(850))

		for i in range(len(yy)):
			if yy[i] not in id2domain_test:continue
			x.append(feature[i])
			y.append(id2domain_test[yy[i]])

		permutation = np.random.permutation(len(x))
		x = [x[i] for i in permutation]
		y = [y[i] for i in permutation]


		numNodes = len(x)
		testingsetlength = numNodes//10
		neigh.fit(x[testingsetlength:], y[testingsetlength:])
		total = len(y[:testingsetlength])
		results = neigh.predict(x[:testingsetlength])
		hit = 0
		for i in range(total):
			if y[:testingsetlength][i]==results[i]:
				hit+=1
		final_results.append(hit/total)
	print("mean: "+str(np.mean(final_results)))
	print("str: "+str(np.std(final_results)))
