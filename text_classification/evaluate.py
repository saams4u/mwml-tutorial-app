# evaluate.py - evaluate the testing accuracy results

import os
import time

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from text_classification.train import *
from text_classification.utils import *
from text_classification.data import HANDataset

import wandb
import json

import matplotlib.pyplot as plt


def plot_confusion_matrix(predictions, labels, classes, fp, cmap=plt.cm.Blues):

	cm = confusion_matrix(labels, predictions)
	cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(cm, cmap=plt.cm.Blues)
	fig.colorbar(cax)

	plt.title("Confusion matrix")
	plt.ylabel("True label")
	plt.xlabel("Predicted label")

	ax.set_xticklabels([''] + classes)
	ax.set_yticklabels([''] + classes)

	ax.xaxis.set_label_position('bottom')
	ax.xaxis.tick_bottom()

	thresh = cm.max() / 2.

	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, f"{cm[i, j]:d} ({cm_norm[i, j]*100:.1f}%)",
				 horizontalalignment="center", 
				 color="white" if cm[i, j] > thresh else "black")

	plt.rcParams["figure.figsize"] = (7, 7)
	plt.savefig(fp)


def get_performance(predictions, labels, classes):
    
	performance = {'overall': {}, 'class': {}}
	metrics = precision_recall_fscore_support(labels, predictions)

	performance['overall']['precision'] = np.mean(metrics[0])
	performance['overall']['recall'] = np.mean(metrics[1])
	performance['overall']['f1'] = np.mean(metrics[2])
	performance['overall']['num_samples'] = np.float64(np.sum(metrics[3]))

	for i in range(len(classes)):
		performance['class'][classes[i]] = {
			"precision": metrics[0][i],
			"recall": metrics[1][i],
			"f1": metrics[2][i],
			"num_samples": np.float64(metrics[3][i])
		}

	return performance


def evaluate(model):

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
    
    return loss, accuracy, predictions, labels

# plot_confusion_matrix(predictions, labels, label_map, 
    #     fp=os.path.join(wandb.run.dir, 'confusion_matrix.png'))