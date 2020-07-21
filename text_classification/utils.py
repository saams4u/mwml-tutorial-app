# utils.py - utility functions to aid app operations.

import torch
from torch import nn

import numpy as np

from collections import Counter
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from tqdm import tqdm

import pandas as pd
import itertools
import os

import json
import gensim
import logging


classes = ['Society & Culture',
           'Science & Mathematics',
           'Health',
           'Education & Reference',
           'Computers & Internet',
           'Sports',
           'Business & Finance',
           'Entertainment & Music',
           'Family & Relationships',
           'Politics & Government']

label_map = {k: v for v, k in enumerate(classes)}
rev_label_map = {v: k for k, v in label_map.items()}

# Tokenizers
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()


def save_dict(d, filepath):

    with open(filepath, 'w') as fp:
        json.dump(d, indent=2, sort_keys=False, fp=fp)


def preprocess(text):

    if isinstance(text, float):
        return ''

    return text.lower().replace('<br />', '\n').replace('<br>', '\n').replace('\\n', '\n').replace('&#xd;', '\n')


def read_csv(csv_folder, split, sentence_limit, word_limit):

    assert split in {'train', 'test'}

    docs = []
    labels = []
    word_counter = Counter()

    data = pd.read_csv(os.path.join(csv_folder, split + '.csv'), header=None)

    for i in tqdm(range(data.shape[0])):
        row = list(data.loc[i, :])

        sentences = list()

        for text in row[1:]:
            for paragraph in preprocess(text).splitlines():
                sentences.extend([s for s in sent_tokenizer.tokenize(paragraph)])

        words = list()

        for s in sentences[:sentence_limit]:
            w = word_tokenizer.tokenize(s)[:word_limit]
            if len(w) == 0:
                continue
            words.append(w)
            word_counter.update(w)
        if len(words) == 0:
            continue

        labels.append(int(row[0]) - 1)
        docs.append(words)

    return docs, labels, word_counter


def create_input(csv_folder, output_folder, sentence_limit, 
                       word_limit, min_word_count=5, save_word2vec_data=True):
    
    print('\nReading and preprocessing training data...\n')
    train_docs, train_labels, word_counter = read_csv(csv_folder, 'train', sentence_limit, word_limit)

    if save_word2vec_data:
        torch.save(train_docs, os.path.join(output_folder, 'word2vec_data.pth.tar'))
        print('\nText data for word2vec saved to %s.\n' % os.path.abspath(output_folder))

    word_map = dict()
    word_map['<pad>'] = 0

    for word, count in word_counter.items():
        if count >= min_word_count:
            word_map[word] = len(word_map)

    word_map['<unk>'] = len(word_map)
    print('\n Discarding words with counts less than %d, the size of the vocab is %d.\n' 
        % (min_word_count, len(word_map)))

    with open(os.path.join(output_folder, 'word_map.json'), 'w') as j:
        json.dump(word_map, j)
    print('Word map saved to %s.\n' % os.path.abspath(output_folder))

    print('Encoding and padding training data...\n')
    encoded_train_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) 
            + [0] * (word_limit - len(s)), doc)) + [[0] * word_limit] * 
            (sentence_limit - len(doc)), train_docs))

    sentences_per_train_document = list(map(lambda doc: len(doc), train_docs))
    words_per_train_sentence = list(
        map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (sentence_limit - len(doc)), train_docs))

    print('Saving...\n')
    assert len(encoded_train_docs) == len(train_labels) == len(sentences_per_train_document) == len(
        words_per_train_sentence)

    torch.save({'docs': encoded_train_docs,
                'labels': train_labels,
                'sentences_per_document': sentences_per_train_document,
                'words_per_sentence': words_per_train_sentence},
                os.path.join(output_folder, 'TRAIN_data.pth.tar'))
    print('Encoded, padded training data saved to %s.\n' % os.path.abspath(output_folder))

    del train_docs, encoded_train_docs, train_labels, sentences_per_train_document, words_per_train_sentence

    print('Reading and preprocessing test data...\n')
    test_docs, test_labels, _ = read_csv(csv_folder, 'test', sentence_limit, word_limit)

    print('\nEncoding and padding test data...\n')
    encoded_test_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) +
            [0] * (word_limit - len(s)), doc)) + [[0] * word_limit] * (sentence_limit - len(doc)), test_docs))

    sentences_per_test_document = list(map(lambda doc: len(doc), test_docs))
    words_per_test_sentence = list(
        map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (sentence_limit - len(doc)), test_docs))

    print('Saving...\n')
    assert len(encoded_test_docs) == len(test_labels) == len(sentences_per_test_document) == len(
        words_per_test_sentence)

    torch.save({'docs': encoded_test_docs,
                'labels': test_labels,
                'sentences_per_document': sentences_per_test_document,
                'words_per_sentence': words_per_test_sentence},
                os.path.join(output_folder, 'TEST_data.pth.tar'))
    print('Encoded, padded test data saved to %s.\n' % os.path.abspath(output_folder))

    print('All done!\n')


def train_word2vec_model(data_folder, algorithm='skipgram'):

    assert algorithm in ['skipgram', 'cbow']
    sg = 1 if algorithm is 'skipgram' else 0

    sentences = torch.load(os.path.join(data_folder, 'word2vec_data.pth.tar'))
    sentences = list(itertools.chain.from_iterable(sentences))

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = gensim.models.word2vec.Word2Vec(sentences=sentences, size=200, workers=8,
                                            window=10, min_count=5, sg=sg)

    model.init_sims(True)
    model.wv.save(os.path.join(data_folder, 'word2vec_model'))


def init_embedding(input_embedding):

    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def load_word2vec_embeddings(word2vec_file, word_map):

    w2v = gensim.models.KeyedVectors.load(word2vec_file, mmap='r')

    print("\nEmbedding length is %d.\n" % w2v.vector_size)

    embeddings = torch.FloatTensor(len(word_map), w2v.vector_size)
    init_embedding(embeddings)

    print("Loaded embeddings...")

    for word in word_map:
        if word in w2v.vocab:
            embeddings[word_map[word]] = torch.FloatTensor(w2v[word])

    print("Done. \n Embedding vocabulary: %d.\n" % len(word_map))

    return embeddings, w2v.vector_size


def clip_gradient(optimizer, grad_clip):

    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, model, optimizer, word_map):

    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             'word_map': word_map}

    filename = 'checkpoint_han.pth.tar'

    torch.save(state, filename)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, scale_factor):

    print("\nDECAYING learning rate.")

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor

    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def get_best_run(project, metric, objective):
    # Get all runs
    api = wandb.Api()
    runs = api.runs(project)

    # Define objective
    if objective == 'maximize':
        best_metric_value = np.NINF
    elif objective == 'minimize':
        best_metric_value = np.inf

    # Get best run based on metric
    best_run = None
    for run in runs:
        if run.state == "finished":
            metric_value = run.summary[metric]
            if objective == 'maximize':
                if metric_value > best_metric_value:
                    best_run = run
                    best_metric_value = metric_value
            else:
                if metric_value < best_metric_value:
                    best_run = run
                    best_metric_value = metric_value

    return best_run


def load_run(run):
    run_dir = os.path.join(
        config.BASE_DIR, '/'.join(run.summary['run_dir'].split('/')[-2:]))

    # Create run dir if it doesn't exist
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    else:
        return run_dir

    # Load run files (if it exists, nothing happens)
    for file in run.files():
        file.download(replace=False, root=run_dir)

    return run_dir