# eval.py - predict (infer) inputs (single/batch).

import time

from utils import *
from data import HANDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_folder = '/home/saams4u/mwml-tutorial-app/results'

batch_Size = 64
workers = 4
print_freq = 2000
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

	_, predictions = scores.max(dim=1)
	correct_predictions = torch.eq(predictions, labels).sum().item()
	accuracy = correct_predictions / labels.size(0)

	accs.update(accuracy, labels.size(0))

	start = time.time()


print('\n * TEST ACCURACY - %.1f per cent\n' % (accs.avg * 100))