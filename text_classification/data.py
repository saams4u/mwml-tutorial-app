# data.py - load, preprocess, split, tokenize, etc. data.

from torch.utils.data import Dataset
import torch
import os


class HANDataset(Dataset):

	def __init__(self, data_folder, split):
		split = split.upper()
		assert split in {'TRAIN', 'TEST'}
		self.split = split

		self.data = torch.load(os.path.join(data_folder, split + '_data.pth.tar'))

	def __getitem__(self, i):
		return torch.LongTensor(self.data['docs'][i]), \
			   torch.LongTensor([self.data['sentences_per_document'][i]]), \
			   torch.LongTensor(self.data['words_per_sentence'][i]), \
			   torch.LongTensor([self.data['labels'][i]])

	def __len__(self):
		return len(self.data['labels'])
		