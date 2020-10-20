# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
from torch.optim.lr_scheduler import StepLR

import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--model', type=str, default='LSTM',
					help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=64,
					help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=8,
					help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
					help='number of layers')
parser.add_argument('--nview', type=int, default=3,
					help='number of views')
parser.add_argument('--lr', type=float, default=0.001,
					help='initial learning rate')
parser.add_argument('--epochs', type=int, default=15,
					help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
					help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
					help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--save', type=str, default='model.pt',
					help='path to save the final model')
parser.add_argument('--noiseDim', type=int, default=10, help='size of the latent z vector')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--sqlen', default=10, help='length of the sequence')
parser.add_argument('--gamma', default=0.5, help='decay of lr')
# parser.add_argument('--pretrain', default=False, help='if use pretrained embedding')

parser.add_argument('--inputwalks', default='walks/n2v(link_pred)_walks.txt', help='path of input walks')
parser.add_argument('--outf', default='models/VANE_link_pred_n2v_model', help='folder to output models')

args = parser.parse_args()
if not os.path.exists(args.outf):
	os.makedirs(args.outf)

GAN = True
Locality = True

torch.manual_seed(123456)
if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

with open(args.inputwalks,'r',encoding='utf-8') as fin:
	data = list(fin)
	data = [x.strip().split(' ') for x in data]
	xs = torch.tensor([[int(y) for y in x[1].split(',')] for x in data])
	#ys = torch.tensor([int(x[0]) for x in data])
	ys = [int(x[0]) for x in data]
	yphi = []
	for i in ys:
		tmp = np.random.randint(0,args.nview)
		while tmp == i:
			tmp = np.random.randint(0,args.nview)
		yphi.append(tmp)
	ys = torch.tensor(ys)
	yphi = torch.tensor(yphi)


dataset = torch.utils.data.TensorDataset(xs, ys, yphi)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Build the model
###############################################################################

ntokens = 850
extractor = model.Extractor(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout).to(device)
decoder = model.Decoder(args.nhid, args.nview, args.dropout).to(device)
generator = model.Generator(args.emsize, args.noiseDim, args.dropout).to(device)
discriminator = model.Discriminator(args.emsize, args.dropout).to(device)

criterionD1 = nn.CrossEntropyLoss() # for D_S
criterionD2 = nn.BCELoss() # for D_N
LocalityConstrains = nn.CosineEmbeddingLoss()

noiseDim = int(args.noiseDim)

real_label = 1
fake_label = 0

lr = args.lr
optimizerExtractor = optim.Adam(extractor.parameters(), lr, betas=(args.beta1, 0.999))
optimizerDecoder = optim.Adam(decoder.parameters(), lr, betas=(args.beta1, 0.999)) # the D_S net
optimizerGenerator = optim.Adam(generator.parameters(), lr, betas=(args.beta1, 0.999))
optimizerDiscriminator = optim.Adam(discriminator.parameters(), lr, betas=(args.beta1, 0.999)) # the D_N net

schedulerExtractor = StepLR(optimizerExtractor, step_size=5, gamma=args.gamma)
schedulerDecoder = StepLR(optimizerDecoder, step_size=5, gamma=args.gamma)
schedulerGenerator = StepLR(optimizerGenerator, step_size=5, gamma=args.gamma)
schedulerDiscriminator = StepLR(optimizerDiscriminator, step_size=5, gamma=args.gamma)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
	"""Wraps hidden states in new Tensors, to detach them from their history."""

	if isinstance(h, torch.Tensor):
		return h.detach()
	else:
		return tuple(repackage_hidden(v) for v in h)

for epoch in range(args.epochs):
	for i, data in enumerate(dataloader, 0):

		x = data[0]
		y = data[1]
		yphi = data[2]

		batch_size = args.batch_size
		label = torch.full((batch_size,1), real_label, dtype=torch.float, device=device)

		###########################
		# (0) LocalityConstrains updates
		###########################
		if Locality:
			extractor.zero_grad()
			hidden = extractor.init_hidden(args.batch_size)
			# embedding_sequence: batchsize, seqlen, embeddingSize
			representation, _, emb1 = extractor(x, hidden)
			localitylabel = torch.full((batch_size,1), real_label, dtype=torch.float, device=device)
			localityloss1 = LocalityConstrains(emb1[0], emb1[1], localitylabel)
			localityloss1.backward(retain_graph=True)
			optimizerExtractor.step()
		else:
			extractor.zero_grad()
			hidden = extractor.init_hidden(args.batch_size)
			# embedding_sequence: batchsize, seqlen, embeddingSize
			representation, _, emb1 = extractor(x, hidden)
			localityloss1 = 0

		###########################
		# (1) Update D_S network
		###########################
		extractor.zero_grad()
		decoder.zero_grad()
		if GAN:
			output = decoder(representation.detach())
		else:
			output = decoder(representation)
		errDD1 = criterionD1(output, y)
		errDD1.backward()
		D_G_z1 = output.mean().item()
		optimizerDecoder.step()
		if not GAN:
			optimizerExtractor.step()

		############################
		# (2) Update D_N network
		###########################
		if GAN:
			# with real samples
			discriminator.zero_grad()
			label.fill_(real_label)
			output = discriminator(emb1[0].detach())
			errD2r = criterionD2(output, label)
			errD2r.backward()
			D2_real = output.mean().item()

			# with fake samples
			noise = torch.randn(batch_size, noiseDim, device=device)
			fake = generator(noise)
			label.fill_(fake_label)
			output = discriminator(fake.detach())
			errD2f = criterionD2(output, label)
			errD2f.backward()
			D2_fake = output.mean().item()
			optimizerDiscriminator.step()

		############################
		# (3) Update G network: maximize log(D(G(z)))
		###########################
		if GAN:
			generator.zero_grad()
			label.fill_(real_label)  # fake labels are real for generator cost
			output = discriminator(fake)
			errG = criterionD2(output, label)
			errG.backward()
			D_G = output.mean().item()
			optimizerGenerator.step()

		###########################
		# (4) Update Extractor
		###########################
		if GAN:
			extractor.zero_grad()
			# if not GAN:
			representation, _, emb1 = extractor(x, hidden)
			output = decoder(representation)
			errD1 = criterionD1(output, yphi)
			errD1.backward(retain_graph=True)

			# extractor.zero_grad()
			output = discriminator(emb1[0])
			label.fill_(real_label)
			errD2 = criterionD2(output, label)
			errD2.backward()
			D_E2 = output.mean().item()
			optimizerExtractor.step()
		if not GAN:
			pass # Extractor has been updated when updating D_S

		if GAN:
			print('[%d/%d][%d/%d] Loss_D1: %.4f Loss_D2real: %.4f Loss_D2fake: %.4f Loss_G: %.4f Loss_E: %.4f D(x): %.4f Loss_locality: %.4f'
				  % (epoch, args.epochs, i, len(dataloader),
					 errDD1.item(), errD2r, errD2f, errG, errD1+errD2, D2_real, localityloss1))

		else:
			print('[%d/%d][%d/%d] | Loss_D1: %.4f | Loss_E: %.4f | Loss_locality: %.4f' % (epoch, args.epochs, i, len(dataloader), errDD1.item(), errD1.item(), localityloss1))

	# if epoch%5 == 0:
	torch.save(extractor.state_dict(), '%s/Extractor_epoch_%d.pth' % (args.outf, epoch))
	torch.save(decoder.state_dict(), '%s/Decoder_epoch_%d.pth' % (args.outf, epoch))
	torch.save(generator.state_dict(), '%s/Generator_epoch_%d.pth' % (args.outf, epoch))
	torch.save(discriminator.state_dict(), '%s/Discriminator_epoch_%d.pth' % (args.outf, epoch))

	schedulerExtractor.step()
	schedulerDecoder.step()
	schedulerGenerator.step()
	schedulerDiscriminator.step()
