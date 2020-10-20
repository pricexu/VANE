import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Extractor(nn.Module):

	def __init__(self, rnn_type, ntoken, embedding_dim, nhid, nlayers, dropout):
		super(Extractor, self).__init__()
		self.drop = nn.Dropout(dropout)

		self.encoder = nn.Embedding(ntoken, embedding_dim)
		self.init_weights()

		if rnn_type in ['LSTM', 'GRU']:
			self.rnn = getattr(nn, rnn_type)(embedding_dim, nhid, nlayers)
		else:
			try:
				nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
			except KeyError:
				raise ValueError( """An invalid option for `--model` was supplied,
								 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
			self.rnn = nn.RNN(embedding_dim, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

		self.rnn_type = rnn_type
		self.nhid = nhid
		self.nlayers = nlayers

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, input,hidden):
		emb_o = self.drop(self.encoder(input))
		indexs = np.random.randint(int(emb_o.shape[1]/2), size=2)
		if (indexs[0] == indexs[1]):
			indexs = np.random.randint(int(emb_o.shape[1]/2), size=2)
		emb_1 = (emb_o[:,indexs[0],:], emb_o[:,indexs[1],:])

		#emb_o : batchsize, seqlen, embeddingSize
		#input of LSTM: seqlen, batchsize, embeddingSize
		emb = emb_o.transpose(0,1)
		output, hidden = self.rnn(emb, hidden)
		#output: seqlen, batchsize, hiddenSize
		output = self.drop(output)
		return output[-1,:,:], hidden, emb_1

	def init_hidden(self, bsz):
		weight = next(self.parameters())
		if self.rnn_type == 'LSTM':
			return (weight.new_zeros(self.nlayers, bsz, self.nhid),weight.new_zeros(self.nlayers, bsz, self.nhid))
		else:
			return weight.new_zeros(self.nlayers, bsz, self.nhid)

class Decoder(nn.Module):

	def __init__(self, nhid, ndomain, dropout=0.5):
		super(Decoder, self).__init__()
		self.drop = nn.Dropout(dropout)
		self.decoder = nn.Sequential(
			nn.Linear(nhid, nhid),
			nn.Dropout(dropout),
			nn.LeakyReLU(),
			nn.Linear(nhid, nhid),
			nn.Dropout(dropout),
			nn.LeakyReLU(),
			nn.Linear(nhid, ndomain),
		)

		self.nhid = nhid
		self.ndomain = ndomain

	def forward(self, input):
		decoded = self.decoder(input)
		return decoded

class Generator(nn.Module):

	def __init__(self, nhid, noiseDim, dropout=0.5):
		super(Generator, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(noiseDim, nhid),
			nn.Dropout(dropout),
			nn.LeakyReLU(True),
			nn.Linear(nhid, nhid),
			nn.Dropout(dropout),
			nn.LeakyReLU(True),
			nn.Linear(nhid, nhid),
			nn.Dropout(dropout),
			nn.LeakyReLU(True),
			nn.Linear(nhid, nhid)
		)

		self.nhid = nhid
		self.noiseDim = noiseDim

	def forward(self, input):
		output = self.main(input)
		return output

class Discriminator(nn.Module):

	def __init__(self, nhid, dropout=0.5):
		super(Discriminator, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(nhid, nhid),
			nn.Dropout(dropout),
			nn.LeakyReLU(),
			nn.Linear(nhid, nhid),
			nn.Dropout(dropout),
			nn.LeakyReLU(),
			nn.Linear(nhid, 1),
			nn.Sigmoid()
		)

		self.nhid = nhid

	def forward(self, input):
		output = self.main(input)
		return output
