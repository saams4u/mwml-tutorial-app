# create_input.py - generate input files to be processed for classification

from text_classification.utils import create_input, train_word2vec_model


if __name__ == '__main__':
	create_input(csv_folder='./dataset',
				 output_folder='./results',
				 sentence_limit=15,
				 word_limit=20,
				 min_word_count=5)

	train_word2vec_model(data_folder='./results',
						 algorithm='skipgram')