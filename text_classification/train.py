# train.py - initialize and train a model.

import os
import sys

sys.path.append(".")

from argparse import ArgumentParser
from datetime import datetime

import io
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np

import random
import time

from tqdm import tqdm

import wandb

wandb.init(project="mwml-tutorial-app")

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
import torch.optim as optim

from text_classificiation import config, data, models, utils, evaluate

from models import HierarchialAttentionNetwork
from utils import *
from data import HANDataset


data_folder = '../results'
word2vec_file = os.path.join(data_folder, 'word2vec_model')

with open(os.path.join(data_folder, 'word_map.json'), 'r') as j:
	word_map = json.load(j)

n_classes = len(label_map)
word_rnn_size = 50
sentence_rnn_size = 50

word_rnn_layers = 1
sentence_rnn_layers = 1
word_att_size = 100

sentence_att_size = 100
dropout = 0.3
fine_tune_word_embeddings = True

start_epoch = 0
batch_size = 64
lr = 1e-3

momentum = 0.9
workers = 4
epochs = 2

grad_clip = None
print_freq = 2000
checkpoint = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def main():

	global checkpoint, start_epoch, word_map

	if checkpoint is not None:
		checkpoint = torch.load(checkpoint)
		model = checkpoint['model']
		optimizer = checkpoint['optimizer']
		word_map = checkpoint['word_map']
		start_epoch = checkpoint['epoch'] + 1
		print('\nLoaded checkpoint from epoch %d.\n' % (start_epoch - 1))
	else:
		embeddings, emb_size = load_word2vec_embeddings(word2vec_file, word_map)

		model = HierarchialAttentionNetwork(n_classes=n_classes,
											vocab_size=len(word_map),
											emb_size=emb_size,
											word_rnn_size=word_rnn_size,
											sentence_rnn_size=sentence_rnn_size,
											word_rnn_layers=word_rnn_layers,
											sentence_rnn_layers=sentence_rnn_layers,
											word_att_size=word_att_size,
											sentence_att_size=sentence_att_size,
											dropout=dropout)

		model.sentence_attention.word_attention.init_embeddings(embeddings)
		model.sentence_attention.word_attention.fine_tune_embeddings(fine_tune_word_embeddings)

		optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

	train_criterion = nn.CrossEntropyLoss()

	model = model.to(device)

	wandb.watch(model)
	config.logger.info(
		"Model:\n"
		f"	{model.named_parameters}")

	train_criterion = train_criterion.to(device)

	train_loader = torch.utils.data.DataLoader(HANDataset(data_folder, 'train'), batch_size=batch_size,
											   shuffle=True, num_workers=workers, pin_memory=True)

	config.logger.info("Training:")
	
	for epoch in range(start_epoch, epochs):
		train(train_loader=train_loader,
			  model=model, 
			  criterion=train_criterion,
			  optimizer=optimizer,
			  epoch=epoch)

		adjust_learning_rate(optimizer, 0.1)

		best_val_loss = np.inf

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			save_checkpoint(epoch, model, optimizer, word_map)


def train(train_loader, model, criterion, optimizer, epoch):

	model.train()

	batch_time = AverageMeter()
	data_time = AverageMeter()

	losses = AverageMeter()
	accs = AverageMeter()

	start = time.time()

	for i, (train_documents, train_sentences_per_document, train_words_per_sentence, labels) in enumerate(train_loader):

		data_time.update(time.time() - start)

		train_documents = train_documents.to(device)
		train_sentences_per_document = train_sentences_per_document.squeeze(1).to(device)
		train_words_per_sentence = train_words_per_sentence.to(device)
		
		labels = labels.squeeze(1).to(device)

		train_scores, train_word_alphas, train_sentence_alphas = model(train_documents, train_sentences_per_document, 
			train_words_per_sentence)

		train_loss = criterion(train_scores, labels)

		optimizer.zero_grad()
		train_loss.backward()

		if grad_clip is not None:
			clip_gradient(optimizer, grad_clip)

		optimizer.step()

		_, train_predictions = train_scores.max(dim=1)
		correct_train_predictions = torch.eq(train_predictions, labels).sum().item()
		train_acc = correct_train_predictions / labels.size(0)

		losses.update(train_loss.item(), labels.size(0))
		batch_time.update(time.time() - start)
		accs.update(train_acc, labels.size(0))

		start = time.time()
		evaluate(model, word_map)

		if i % print_freq == 0:
			print('Epoch [{0}]{1}/{2}]\t'
				  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data Load Time {data_time.val:.3f} ({data_Time.avg:.3f})\t'
				  'Train Loss {train_loss.val:.4f} ({train_loss.avg:.4f})\t'
				  'Train Acc {train_acc.val:.3f} ({train_acc.avg:.3f})\t'
				  'Val Loss {val_loss.val:.4f} ({val_loss.avg:.4f})\t'
				  'Val Acc {val_acc.val:.3f} ({val_acc.avg:.3f})'.format(epoch, i, len(train_loader),
				  												  batch_time=batch_time,
				  												  data_time=data_time, loss=losses,
				  												  acc=accs))
			config.logger.info(
	            f"Epoch: {epoch+1} | "
	            f"train_loss: {train_loss:.2f}, train_acc: {train_acc:.1f}, "
	            f"val_loss: {val_loss:.2f}, val_acc: {val_acc:.1f}")
	        wandb.log({
	            "train_loss": train_loss,
	            "train_acc": train_acc,
	            "val_loss": val_loss,
	            "val_acc": val_acc})


if __name__ == '__main__':
	main()