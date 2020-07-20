# train.py - initialize and train a model.

import time

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from models import HierarchialAttentionNetwork
from utils import *
from data import HANDataset


data_folder = '/home/saams4u/mwml-tutorial-app/results'
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

	criterion = nn.CrossEntropyLoss()

	model = model.to(device)
	criterion = criterion.to(device)

	train_loader = torch.utils.data.DataLoader(HANDataset(data_folder, 'train'), batch_size=batch_size,
											   shuffle=True, num_workers=workers, pin_memory=True)

	for epoch in range(start_epoch, epochs):
		train(train_loader=train_loader,
			  model=model, 
			  criterion=criterion,
			  optimizer=optimizer,
			  epoch=epoch)

		adjust_learning_rate(optimizer, 0.1)

		save_checkpoint(epoch, model, optimizer, word_map)


def train(train_loader, model, criterion, optimizer, epoch):

	model.train()

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	accs = AverageMeter()

	start = time.time()

	for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(train_loader):

		data_time.update(time.time() - start)

		documents = documents.to(device)
		sentences_per_document = sentences_per_document.squeeze(1).to(device)
		words_per_sentence = words_per_sentence.to(device)
		labels = labels.squeeze(1).to(device)

		scores, word_alphas, sentence_alphas = model(documents, sentences_per_document, words_per_sentence)

		loss = criterion(scores, labels)

		optimizer.zero_grad()
		loss.backward()

		if grad_clip is not None:
			clip_gradient(optimizer, grad_clip)

		optimizer.step()

		_, predictions = scores.max(dim=1)
		correct_predictions = torch.eq(predictions, labels).sum().item()
		accuracy = correct_predictions / labels.size(0)

		losses.update(loss.item(), labels.size(0))
		batch_time.update(time.time() - start)
		accs.update(accuracy, labels.size(0))

		start = time.time()

		if i % print_freq == 0:
			print('Epoch [{0}]{1}/{2}]\t'
				  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data Load Time {data_time.val:.3f} ({data_Time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, i, len(train_loader),
				  												  batch_time=batch_time,
				  												  data_time=data_time, loss=losses,
				  												  acc=accs))

if __name__ == '__main__':
	main()