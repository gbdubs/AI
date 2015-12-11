from wikipedia import get_word_list
import os.path

training_path = '/home/u/fall12/gward/Desktop/AI/data/training_set.tsv'
def get_training_lines():
	if os.path.isfile(training_path):
		f = open(training_path, 'r')
		data = f.read().split("\n")
		f.close()
		return data
	else:
		print " =!=!=!=!= TRAINING FILE WAS NOT FOUND. =!=!=!=!="



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
	
def get_answers(tsv_line):
	answers = dict()
	for a in range(0, 4):
		answers["" + chr(65 + a)] = get_answer(tsv_line, a)
	return answers

def main():
	training_set = get_training_lines()
	print "["+ get_id(training_set[1]) + "]"
	print get_question(training_set[1])
	print get_answers(training_set[1])
	print get_correct_answer(training_set[1])

if __name__ == "__main__":
	main()