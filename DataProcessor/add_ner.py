'''
Convert TACRED data into json format
Author: Maosen Zhang
Email: zhangmaosen@pku.edu.cn
'''
__author__ = 'Maosen'
from tqdm import tqdm
import json
import argparse
import unicodedata
from stanza.nlp.corenlp import CoreNLPClient
# from nltk.tokenize import word_tokenize
relation_set = set()
ner_set = set()
pos_set = set()

# cd ~/maosen/stanford-corenlp-full-2018-02-27
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer

class NLPParser(object):
	"""
	NLP parse, including Part-Of-Speech tagging.
	Attributes
	==========
	parser: StanfordCoreNLP
		the Staford Core NLP parser
	"""
	def __init__(self):
		self.parser = CoreNLPClient(default_annotators=['ssplit', 'tokenize', 'ner'], server='http://localhost:9001')

	def get_ner(self, tokens):
		sent = ' '.join(tokens)
		result = self.parser.annotate(sent)
		ner = []
		for token in result.sentences[0]:
			ner.append(token.ner)
		return ner

def find_index(sen_split, word_split):
	index1 = -1
	index2 = -1
	for i in range(len(sen_split)):
		if str(sen_split[i]) == str(word_split[0]):
			flag = True
			k = i
			for j in range(len(word_split)):
				if word_split[j] != sen_split[k]:
					flag = False
				if k < len(sen_split) - 1:
					k+=1
			if flag:
				index1 = i
				index2 = i + len(word_split)
				break
	return index1, index2


def read(data, in_dir, out_dir):
	cotype_filename = '%s/%s_new.json' % (in_dir, data)
	out_filename = '%s/%s_new_with_ner.json' % (out_dir, data)
	MAXLEN = 0
	instances = []
	nlp = NLPParser()
	with open(cotype_filename, 'r') as cotype_file, open(out_filename, 'w') as out_file:
		for idx, line in enumerate(tqdm(cotype_file.readlines())):
			try:
				sent = json.loads(line.strip())
				tokens = sent['tokens']
				ner_tags = nlp.get_ner(tokens)
				sent['ner'] = ner_tags
				out_file.write(json.dumps(sent))
				out_file.write('\n')
			except:
				pass
	return instances

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--in_dir', type=str, default='./data/cotype_processed_data/KBP')
	parser.add_argument('--out_dir', type=str, default='./data/cotype_processed_data/KBP')
	args = vars(parser.parse_args())

	for data in ['train', 'test', 'dev']:
		read(data, args['in_dir'], args['out_dir'])
	# print(relation_set)
	# print(pos_set)
	# print(ner_set)