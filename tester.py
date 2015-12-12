from __future__ import absolute_import
from __future__ import print_function
import tensorflow.python.platform
import collections
import math
import numpy as np
import os
import random
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import zipfile
import urllib2
from wikipedia import get_all_wiki_files_as_list
from datagetter import *
import csv
import scipy.stats as stats

embedding_root = '/home/u/fall12/gward/Desktop/AI/running/embeddings/'
embedding_file = 'friday-6pm'
embedding_path = embedding_root + embedding_file + ".npy"
dictionary_path = embedding_root + embedding_file + "-dict.csv"
result_path = embedding_root + embedding_file + "-result-test.tsv"

vocabulary_size = 8000

dictionary = dict()

final_embeddings = None

if os.path.isfile(dictionary_path):
  csv_reader = csv.reader(open(dictionary_path))
  print(csv_reader)
  for key, val in csv_reader:
      dictionary[key] = val

if os.path.isfile(embedding_path):
  final_embeddings = np.load(embedding_path)


def embed(text):
  l = text.split(" ")
  encoding = np.ndarray(shape=(1, vocabulary_size), dtype=np.int32)
  for i in range(0, len(encoding[0])):
    encoding[0,i] = 0
  for word in l:
    index = dictionary.get(word)
    if index != None:
      encoding[0, index] = encoding[0, index] + 1;
  result = np.dot(encoding, final_embeddings)
  return result

def flatten(a):
  b = np.ndarray(shape=(len(a[0])), dtype=np.float64)
  for i in range(0, len(a[0])):
    b[i] = a[0,i]
  return b


all_confidences = list()

def raw_confidence(scores):
  dim = (scores[1] * scores[1] + scores[2] * scores[2] + scores[3] * scores[3] + .01)/4
  raw = pow(2, scores[0] / dim)
  all_confidences.append(raw)
  return raw

def scaled_confidence(score):
  sorted_conf = sorted(all_confidences)
  for i in range(0, len(all_confidences)):
    if sorted_conf[i] > score:
      return (i * 100.0) / len(all_confidences)

def predict(question, answers):
  # somehow predict, by embedding each one, then going through the training set?
  equestion = flatten(embed(question))
  eanswers = dict()
  for a in answers.keys():
    eanswers[a] = flatten(embed(answers[a]))
  best_score = -10000000;
  best_ans = 0;
  scores = list()
  for a in answers.keys():
    score = np.dot(equestion, eanswers[a])
    scores.append(score)
    if (score > best_score):
      best_score = score
      best_ans = a
    scores = sorted(scores)
  raw_conf = raw_confidence(scores)
  return (best_ans, raw_conf)


all_results = list()

count = 0

for tsv_line in get_testing_lines():
  if count % 50 == 1:
    print("Making Progress! - ", count)

  count = count + 1
  result = dict()
  
  question_id = get_id(tsv_line)
  result["id"] = question_id

  question = get_question(tsv_line)
  answers = get_answers_t(tsv_line)
  (prediction, conf) = predict(question, answers)
  result["predict"] = prediction
  result["raw"] = conf
  all_results.append(result)

print(all_results)

for result in all_results:
  result["conf"] = scaled_confidence(result["raw"])

print(all_results)

w = open(result_path, 'w')
for result in all_results:
  s = ""+result["id"]+'\t'+result["predict"]+'\t'+str(result["conf"])+'\n'
  w.write(s)
