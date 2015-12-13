from wikipedia import get_cleaned_wiki_article_word_list_on_topic
import os.path
from topia.termextract import extract

# Defines the files that we will keep testing and training data within.
# These are hardcoded for simplicity.
training_path = '/home/u/fall12/gward/Desktop/AI/data/training_set.tsv'
testing_path = '/home/u/fall12/gward/Desktop/AI/data/validation_set.tsv'


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#             DOWNLOAD WIKI ARTICLES FROM KEYWORDS              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Gets keywords from a given piece of text (Q's or A's), and   #
#  downloads them using the wikipedia APIs that we created.     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def download_all_resources():
	training_set = get_training_lines()
	count = 0
	for tsv_line in training_set:
		print count
		count = count+1
		for keyword in get_keywords(tsv_line):
			get_cleaned_wiki_article_word_list_on_topic(keyword)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                KEYWORD EXTRACTION USING TOPIA                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Gets keywords from a given piece of text (Q's or A's)        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_kewords_for_text(text):
	result = set()
	for kwrd in extractor(text):
		result.add(kwrd[0])
	return result

def get_keywords(tsv_line):
	result = get_kewords_for_text(get_question(tsv_line))
	answers = get_answers(tsv_line)
	for val in answers.values():
		result |= get_kewords_for_text(val)
	return result

# As was mentioned in the writeup, we use an external library [TOPIA]
# to extract keywords from our training set, but not our validation set.
extractor = extract.TermExtractor()
extractor.filter = extract.permissiveFilter


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                        GET TSV DATA LINES                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Gets all of the lines for the specific kind of data as tab   #
#  delimited lines.  Prints errors if they are not found.       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_testing_lines():
	if os.path.isfile(testing_path):
		f = open(testing_path, 'r')
		data = f.read().split("\n")
		f.close()
		return data[1:]
	else:
		print " =!=!=!=!= TESTING FILE WAS NOT FOUND. =!=!=!=!="

def get_training_lines():
	if os.path.isfile(training_path):
		f = open(training_path, 'r')
		data = f.read().split("\n")
		f.close()
		return data[1:]
	else:
		print " =!=!=!=!= TRAINING FILE WAS NOT FOUND. =!=!=!=!="

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                        GET DATA FROM TSV                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Gets data out of a TSV line using the conventions in our data#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_nth_tab(line, n, start_at=0):
	if '\t' not in line:
		return -1
	else:
		if n is 1:
			return line.index('\t', start_at)
		else:
			return get_nth_tab(line, n-1, line.index('\t', start_at) + 1)

def get_id(tsv_line):
	start = 0
	end = get_nth_tab(tsv_line, 1)
	return tsv_line[start:end]

def get_question(tsv_line):
	start = get_nth_tab(tsv_line, 1) + 1
	end = get_nth_tab(tsv_line, 2)
	return tsv_line[start:end]

def get_correct_answer(tsv_line):
	start = get_nth_tab(tsv_line, 2) + 1
	end = get_nth_tab(tsv_line, 3)
	return tsv_line[start:end]

def get_answer(tsv_line, n):
	start = get_nth_tab(tsv_line, 3 + n) + 1
	if n is 3:
		end = len(tsv_line)
	else: 
		end = get_nth_tab(tsv_line, 4 + n)
	return tsv_line[start:end]

def get_answer_t(tsv_line, n):
	start = get_nth_tab(tsv_line, 2 + n) + 1
	if n is 3:
		end = len(tsv_line)
	else: 
		end = get_nth_tab(tsv_line, 3 + n)
	return tsv_line[start:end]
	
def get_answers(tsv_line):
	answers = dict()
	for a in range(0, 4):
		answers["" + chr(65 + a)] = get_answer(tsv_line, a)
	return answers

def get_answers_t(tsv_line):
	answers = dict()
	for a in range(0, 4):
		answers["" + chr(65 + a)] = get_answer_t(tsv_line, a)
	return answers