import urllib2
import json
import os.path
from os import listdir
from os.path import isfile, join

# Uses keyword search in the TRAINING phase to download potentially relevant
# articles from Wikipedia, for embedding and processing during the VALIDATION
# phase.  Note that we can do this without considering our code to use Keywords
# because we are not using keyword analysis to produce our final results, only
# to determine what might be useful as a later result.

# Defines the Root of the folder in which we will keep the wikipedia articles
# that are downloaded during the testing phase. This is hardcoded for simplicity.
data_root = '/home/u/fall12/gward/Desktop/AI/running/wikipedia/'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                        GET TRANING DATA                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  We need to get all of our articles as a list in order to     #
#  create embeddings for them.  We sepearte them using a custom #
#  delimiter, which we later ignore in processing.  This allows #
#  us to create a skip-gram model easily.                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_all_wiki_files_as_list():
	onlyfiles = sorted([f for f in listdir(data_root) if isfile(join(data_root, f))])
	result = list()
	for f in onlyfiles:
		print f
		l = get_cleaned_wiki_article_word_list_on_topic(f[:-4])
		if l is not None:
			result.append("NEWFILENEWFILE")
			result.extend(l)
	return result

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                        WIKIPEDIA API                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  We recieve back a packet of JSON Data, which we need to      #
#  process over in order to yield formatted information.        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_cleaned_wiki_article_word_list_on_topic(topic_name):
	l = get_word_list(topic_name)
	if is_worth_processing(l) :
		return l
	return None

# Memoizes files on disk, so that we need not download them every time.
# Prints out a lot of information to allow the user to quickly see how
# much network activity is dominating their run time (which is a good)
# deal the first time one runs this.
def get_word_list(topic_name):
	print "\n==>  FILE RETRIEVAL SEQUENCES STARTED  <=="
	expected_path = data_root + topic_name + ".txt"
	if os.path.isfile(expected_path):
		print "--> Found file matching ["+topic_name+"] <--"
		f = open(expected_path, 'r')
		print "==>  RETURNING WORD LIST SUCCESSFULLY  <==\n"
		return f.read().split(" ")
	else:
		if has_checked_for_article_before(topic_name):
			print "--> Checked for ["+topic_name+"] before. No such Article. <--"
			print "==>           RETURNING NONE           <=="
			return None
		print "--> Missing File Named ["+topic_name+"], Going to Wikipedia <--"
		article = get_wikipedia_article(topic_name)
		if article is None:
			mark_checked_for_article(topic_name)
			print "--> No article exists for ["+topic_name+"]. <--"
			print "==>           RETURNING NONE           <=="
			return None
		data_list = process_article(article)
		data = " ".join(data_list)
		f = open(expected_path, 'w')
		f.write(data)
		f.close()
		print "==>  RETURNING WORD LIST SUCCESSFULLY  <==\n"
		return data_list

def get_wikipedia_article(topic_name):
	opener = urllib2.build_opener()
	opener.addheaders = [('User-agent', 'Mozilla/5.0')]
	topic_name = topic_name.strip('!@#$%^&*()-=[]{}\'\"')
	topic_name = topic_name.replace(' ', '%20')
	url = 'https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=&titles='+topic_name
	infile = opener.open(url)
	page = infile.read()
	if not article_found(page):
		return None
	return get_content_from_json(page)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                  REMEMBER WHEN NO ARTICLE EXISTS              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  We don't want to check for the same article twice, these two #
#  Functions define when we have checked for an article before. #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def has_checked_for_article_before(topic_name):
	path = data_root + "DATA-404.txt"
	if os.path.isfile(path):
		f = open(path, 'r')
		already = f.read().split("\n")
		result = topic_name in already
		f.close()
		return result
	else:
		return False

def mark_checked_for_article(topic_name):
	path = data_root + "DATA-404.txt"
	with open(path, 'a') as f:
		f.write("\n" + topic_name)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                 PROCESS RESPONSE FROM WIKIPEDIA               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  We recieve back a packet of JSON Data, which we need to      #
#  process over in order to yield formatted information.        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def article_found(json_response):
	if "\"missing\":\"\"" in json_response and len(json_response) < 500:
		return False
	return True

def get_content_from_json(json_response):
	try:
		page_id_start = json_response.index("\"pageid\":") + 9
		page_id_end = json_response.index(',', page_id_start)
		page_id = json_response[page_id_start: page_id_end]
		resp = json.loads(json_response)
		return resp["query"]["pages"][page_id]["extract"].encode('utf-8')
	except ValueError:
		return None

def process_article(article):
	# A set of stop words that I found on a website for stop words.
	stop_words = set(["", "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours  ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"])
  	strip_chars = '().-\'":!.,?![]{}+= '
  	new_words = list()
  	for word in article.split():
  		new_word = word.strip(strip_chars).lower()
  		if new_word not in stop_words:
  			new_words.append(new_word)
  	return new_words


def is_worth_processing(word_list):
	if word_list is None:
		return False
	if len(word_list) < 20:
		return False
	if word_list[0] == 'redirect':
		return False
	if word_list[1] == 'may' and word_list[2] == 'refer':
		return False
	return True





# Boilerplate, creates the files, ensures that this works.

def main():
	print(len(get_all_wiki_files_as_list()))

if __name__ == "__main__":
	main()