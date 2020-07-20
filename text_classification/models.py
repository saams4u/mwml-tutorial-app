# models.py - define model architectures.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pack_padded_sequence, PackedSequence


class HierarchialAttentionNetwork(nn.Module):

	def __init__(self, n_classes, vocab_size, emb_size, word_rnn_size, sentence_rnn_size,
				 word_rnn_layers, sentence_rnn_layers, word_att_size, sentence_att_size, dropout=0.5):
		
		super(HierarchialAttentionNetwork, self).__init__()

		self.sentence_attention = SentenceAttention(vocab_size, emb_size, word_rnn_size,
													sentence_rnn_size, word_rnn_layers,
													sentence_rnn_layers, word_att_size,
													sentence_att_size, dropout)

		self.fc = nn.Linear(2 * sentence_rnn_size, n_classes)

		self.dropout = nn.Dropout(dropout)


	def forward(self, documents, sentences_per_document, words_per_sentence):
		
		document_embeddings, word_alphas, sentence_alphas = self.sentence_attention(documents, sentences_per_document,
																					words_per_sentence)

		scores = self.fc(self.dropout(document_embeddings))

		return scores, word_alphas, sentence_alphas


class SentenceAttention(nn.Module):

	def __init__(self, vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers,
				 sentence_rnn_layers, word_att_size, sentence_att_size, dropout):

		super(SentenceAttention, self).__init__()

		self.word_attention = WordAttention(vocab_size, emb_size, word_rnn_size, word_rnn_layers,
											word_att_size, dropout)

		self.sentence_rnn = nn.GRU(2 * word_rnn_size, sentence_rnn_size, num_layers=sentence_rnn_layers,
								   bidirectional=True, dropout=dropout, batch_first=True)

		self.sentence_attention = nn.Linear(2 * sentence_rnn_size, sentence_att_size)

		self.sentence_context_vector = nn.Linear(sentence_att_size, 1, bias=False)

		self.dropout = nn.Dropout(dropout)


	def forward(self, documents, sentences_per_document, words_per_sentence):

		packed_sentences = pack_padded_sequence(documents, lengths=sentences_per_document.tolist(),
												batch_first=True, enforce_sorted=False)

		packed_words_per_sentence = pack_padded_sequence(words_per_sentence,
														 lengths=sentences_per_document.tolist(),
														 batch_first=True,
														 enforce_sorted=False)

		sentences, word_alphas = self.word_attention(packed_sentences.data,
													 packed_words_per_sentence.data)

		sentences = self.dropout(sentences)

		packed_sentences, _ = self.sentence_rnn(PackedSequence(data=sentences,
															   batch_sizes=packed_sentences.batch_sizes,
															   sorted_indices=packed_sentences.sorted_indices,
															   unsorted_indices=packed_sentences.unsorted_indices))

		att_s = self.sentence_attention(packed_sentences.data)
		att_s = torch.tanh(att_s)

		att_s = self.sentence_context_vector(att_s).squeeze(1)

		max_value = att_s.max()
		att_s = torch.exp(att_s - max_value)

		att_s, _ = pad_packed_sequence(PackedSequence(data=att_s, batch_sizes=packed_sentences.batch_sizes,
													  sorted_indices=packed_sentences.sorted_indices,
													  unsorted_indices=packed_sentences.unsorted_indices),
									   batch_first=True)

		sentence_alphas = att_s, torch.sum(att_s, dim=1, keepdim=True)

		documents, _ = pad_packed_sequence(packed_sentences, batch_first=True)

		documents = documents * sentence_alphas.unsqueeze(2)
		documents = documents.sum(dim=1)

		word_alphas, _ = pad_packed_sequence(PackedSequence(data=word_alphas,
															batch_sizes=packed_sentences.batch_sizes,
															sorted_indices=packed_sentences.sorted_indices,
															unsorted_indices=packed_sentences.unsorted_indices),
											 batch_first=True)

		return documents, word_alphas, sentence_alphas


class WordAttention(nn.Module):

	def __init__(self, vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size, dropout):

		super(WordAttention, self).__init__()

		self.embeddings = nn.Embedding(vocab_size, emb_size)

		self.word_rnn = nn.GRU(emb_size, word_rnn_size, num_layers=word_rnn_layers, bidirectional=True,
							   dropout=dropout, batch_first=True)

		self.word_attention = nn.Linear(2 * word_rnn_size, word_att_size)

		self.word_context_vector = nn.Linear(word_att_size, 1, bias=False)

		self.dropout = nn.Dropout(dropout)


	def init_embeddings(self, embeddings):

		self.embeddings.weight = nn.Parameter(embeddings)


	def fine_tune_embeddings(self, fine_tune=False):

		for p in self.embeddings.parameters():
			p.requires_grad = fine_tune


	def forward(self, sentences, words_per_sentence):

		sentences = self.dropout(self.embeddings(sentences))

		packed_words = pack_padded_sequence(sentences,
											lengths=words_per_sentence.tolist(),
											batch_first=True,
											enforce_sorted=False)

		packed_words, _ = self.word_rnn(packed_words)

		att_w = self.word_attention(packed_words.data)
		att_w = torch.tanh(att_w)

		att_w = self.word_context_vector(att_w).squeeze(1)

		max_value = att_w.max()
		att_w = torch.exp(att_w - max_value)

		att_w, _ = pack_padded_sequence(PackedSequence(data=att_w,
													   batch_sizes=packed_words.batch_sizes,
													   sorted_indices=packed_words.sorted_indices,
													   unsorted_indices=packed_words.unsorted_indices),
										batch_first=True)

		word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)

		sentences, _ = pad_packed_sequence(packed_words, batch_first=True)

		sentences = sentences * word_alphas.unsqueeze(2)
		sentences = sentences.sum(dim=1)

		return sentences, word_alphas

