import os
import sys

sys.path.append(".")

from argparse import ArgumentParser, Namespace

import collections
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from text_classification import config, data, models, utils
from text_classification.train import *


data_folder = './results'
word2vec_file = os.path.join(data_folder, 'word2vec_model')

with open(os.path.join(data_folder, 'word_map.json'), 'r') as j:
    word_map = json.load(j)


def get_run_components(run_dir):
    # Load args
    config = utils.load_json(os.path.join(run_dir, 'config.json'))

    # Load word_map
    _, emb_size = load_word2vec_embeddings(word2vec_file, word_map)

    model = models.HierarchialAttentionNetwork(n_classes=n_classes,
                                                vocab_size=len(word_map),
                                                emb_size=emb_size,
                                                word_rnn_size=word_rnn_size,
                                                sentence_rnn_size=sentence_rnn_size,
                                                word_rnn_layers=word_rnn_layers,
                                                sentence_rnn_layers=sentence_rnn_layers,
                                                word_att_size=word_att_size,
                                                sentence_att_size=sentence_att_size,
                                                dropout=dropout)

    # Load model
    model.load_state_dict(torch.load(os.path.join(run_dir, 'model.pt')))
    device = torch.device('cuda' if (
        torch.cuda.is_available() and args.cuda) else 'cpu')
    model = model.to(device)

    return model, word_map


def predict(inputs, args, model, word_map):

    checkpoint = 'checkpoint_han.pth.tar'
    checkpoint = torch.load(checkpoint)

    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    test_loader = torch.utils.data.DataLoader(HANDataset(data_folder, 'test'), batch_size=batch_size,
                                                shuffle=False, num_workers=workers, pin_memory=True)

    accs = AverageMeter()

    for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(
        tqdm(test_loader, desc='Evaluating')):

        documents = documents.to(device)
        sentences_per_document = sentences_per_document.squeeze(1).to(device)
        labels = labels.squeeze(1).to(device)

        scores, word_alphas, sentence_alphas = model(documents, sentences_per_document,
                                                     words_per_sentence)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(scores, labels)

        _, predictions = scores.max(dim=1)
        correct_predictions = torch.eq(predictions, labels).sum().item()
        accuracy = 100 * correct_predictions / labels.size(0)

        accs.update(accuracy, labels.size(0))
        start = time.time()

    performance = get_performance(predictions, labels, label_map)

    save_dict(performance, filepath=os.path.join(wandb.run.dir, 'performance.json'))
    config.logger.info(json.dumps(performance, indent=2, sort_keys=False))

    results = []
    results.append(performance)

    return results


if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    parser.add_argument('--text', type=str, required=True, help="text to predict")
    
    args = parser.parse_args()
    inputs = [{'text': args.text}]

    # Get best run
    best_run = utils.get_best_run(project="mahjouri-saamahn/mwml-tutorial-app",
                                  metric="test_loss", objective="minimize")

    # Load best run (if needed)
    best_run_dir = utils.load_run(run=best_run)

    # Get run components for prediction
    model, word_map = get_run_components(run_dir=best_run_dir)

    # Predict
    results = predict(inputs=inputs, args=args, model=model, word_map=word_map)
    config.logger.info(json.dumps(results, indent=4, sort_keys=False))

