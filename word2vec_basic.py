from __future__ import absolute_import
from __future__ import print_function
import tensorflow.python.platform
import collections
import math
import numpy as np
import os
import random
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
import zipfile
import urllib2
from wikipedia import get_all_wiki_files_as_list
from datagetter import *
import csv
import scipy.stats as stats

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # #                      MODEL PARAMETERS                 # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# DEFINES THE RUN ID, HOW VARIABLES WILL BE CREATED
# MEMOIZED, AND STORED.  MUST BE UNIQUE BETWEEN RUNS
# WHERE DATA OVERLAP IS NOT 100%.  ALLOWS US TO DO
# REPEATED EVALUATION OVER PRECOMPUTED DATA.
run_id_label = 'friday-6pm'

# ASSIGNS WHETHER WE ARE IN TEST MODE OR TRAINING MODE
# IN ORDER TO DO THE CORRECT IO.
TESTING_FLAG = False 

# VARIABLES THAT WERE OPTIMIZED FOR THEIR VALUES, 
# THESE WERE THE ONES THAT WERE USED FOR THE FINAL
# RESULT, BUT THEY WERE CONSTANTLY MANIPULATED.
vocabulary_size = 8000    # Number of words to create embeddings for
num_steps = 6000001       # Number of steps to train over
batch_size = 128          # Size of single step batch
embedding_size = 128      # Embedding Vector Dimension
skip_window = 2           # Size of the Skip-Gram Window
num_skips = 2             # Input Reuse for a Label

# Path values that are determined by the run_id_label,
# which may be used to store intermediate results.
embedding_root = '/home/u/fall12/gward/Desktop/AI/running/embeddings/'
embedding_path = embedding_root + run_id_label + ".npy"
dictionary_path = embedding_root + run_id_label + "-dict.csv"
result_path = embedding_root + run_id_label + "-TRAIN.tsv"
if TESTING_FLAG:
  result_path = embedding_root + run_id_label + "-TEST.tsv"

# Validation set size for training, was not modified.
valid_size = 30
valid_window = 100

# Allows the scope of these variables to be flexible, allowing us to 
# generate them if they do not yet exist, or to go off of a memoized
# version if they do exist.
final_embeddings = None
dictionary = dict()
reverse_dictionary = dict()

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # #                  DICTIONARY GENERATION                # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checks to see if we have run this computation beforehand, and
# uses the precomputed value if we have.  Note that this is contingent
# upon run_id_label being the same across multiple runs of the python file.
if os.path.isfile(dictionary_path):
  csv_reader = csv.reader(open(dictionary_path))
  print(csv_reader)
  for key, val in csv_reader:
      dictionary[key] = val
else:

  # Defines the corpus texts that were generated through the training set.
  # Then gets them as a cohesive, monolithic list of words.  Very Large.
  words = get_all_wiki_files_as_list()

  # Constructs a dictionary out of the wiki articles, which maps every word
  # in its size to an integer which is the key for the embedding.  The dict
  # ionary generated through this step will be written to disk to allow us
  # to skip this step in the future.
  def build_dataset(words):
    count = [['UNK', -1]]
    # Counts the words in the list and makes a dictionary of the most common.
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
      dictionary[word] = len(dictionary)
    
    # Goes through corpus text and replaces words with numerical translations 
    data = list()
    unk_count = 0
    for word in words:
      if word in dictionary:
        index = dictionary[word]
      else:
        index = 0  # the word is not defined in our dictionary, as it is not of high enough frequency.
        unk_count = unk_count + 1
      data.append(index)
    count[0][1] = unk_count

    # Creates a translation between word number and the word it stands for.
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
  
  # Builds our dataset.
  data, count, dictionary, reverse_dictionary = build_dataset(words)
  
  # Writes the CSV for the dictionary to disk so that we don't have to rebuild if we run again.
  w = csv.writer(open(dictionary_path, "w"))
  for key, val in dictionary.items():
    w.writerow([key, val])

  data_index = 0

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # #                    EMBEDDING TRAINING                 # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# if the embeddings already exist, load them using numpy instead of re-training them.
if os.path.isfile(embedding_path):
  final_embeddings = np.load(embedding_path)

else:

  # Utilize a training model directly out of the Tensor Flow Examples section.
  def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
      buffer.append(data[data_index])
      data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
      target = skip_window  # target label at the center of the buffer
      targets_to_avoid = [ skip_window ]
      for j in range(num_skips):
        while target in targets_to_avoid:
          target = random.randint(0, span - 1)
        targets_to_avoid.append(target)
        batch[i * num_skips + j] = buffer[skip_window]
        labels[i * num_skips + j, 0] = buffer[target]
      buffer.append(data[data_index])
      data_index = (data_index + 1) % len(data)
    return batch, labels
  # Generates a batch to train on.
  batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

  # Building and training a skip-gram model 
  valid_examples = np.array(random.sample(np.arange(valid_window), valid_size))
  num_sampled = 64    # Number of negative examples to sample during each training phase.

  # Defines the Tensorflow skip-gram Model.
  graph = tf.Graph()
  with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    # Construct the variables.
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,num_sampled, vocabulary_size))
    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

  # Trains the tensor flow model.
  with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    tf.initialize_all_variables().run()
    print("Initialized")
    average_loss = 0
    for step in xrange(num_steps):
      batch_inputs, batch_labels = generate_batch(
          batch_size, num_skips, skip_window)
      feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
      # We perform one update step by evaluating the optimizer op (including it
      # in the list of returned values for session.run()
      _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
      average_loss += loss_val
      if step % 1000 == 0:
        if step > 0:
          average_loss = average_loss / 1000
        # The average loss is an estimate of the loss over the last 2000 batches.
        print("Average loss at step ", step, "/",num_steps,": ", average_loss)
        average_loss = 0
      # Updates the user on the progress of the training model with qualative examples
      if step % 10000 == 0:
        np.save(embedding_path, normalized_embeddings.eval())
        sim = similarity.eval()
        for i in xrange(valid_size):
          valid_word = reverse_dictionary[valid_examples[i]]
          top_k = 8 # number of nearest neighbors
          nearest = (-sim[i, :]).argsort()[1:top_k+1]
          log_str = "Nearest to %s:" % valid_word
          for k in xrange(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log_str = "%s %s," % (log_str, close_word)
          print(log_str)
    final_embeddings = normalized_embeddings.eval()

    # Saves the embeddings so that we do not have to recalculate them so frequently.
    np.save(embedding_path, final_embeddings)


  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # #                       EMBEDDING                       # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Embeds a piece of text using the embeddings that were calculated
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

# Flattens an 1xN vector into a N vector.
def flatten(a):
  b = np.ndarray(shape=(len(a[0])), dtype=np.float64)
  for i in range(0, len(a[0])):
    b[i] = a[0,i]
  return b

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # #                       PREDICTION                      # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# A list to store all of our confidence values so that we can easily normalize them
# on a scale of 1 to 100.
all_confidences = list()

# An ad-hoc function to estimate a raw confidence value given a list of scores.
def raw_confidence(scores):
  scores = sorted(scores)
  dim = (scores[1] * scores[1] + scores[2] * scores[2] + scores[3] * scores[3] + .01)/4
  raw = pow(2, scores[0] / dim)
  all_confidences.append(raw)
  return raw

# Utilizes all of the confidence values computed to compute normalized confidence.
def scaled_confidence(score):
  sorted_conf = sorted(all_confidences)
  for i in range(0, len(all_confidences)):
    if sorted_conf[i] > score:
      return (i * 100.0) / len(all_confidences)

# Embed questions and answers, and compute the dot-product between them.
def predict(question, answers):
  # Embeds the questions and answers
  equestion = flatten(embed(question))
  eanswers = dict()
  for a in answers.keys():
    eanswers[a] = flatten(embed(answers[a]))

  # Finds the best dot-product score out of all of the potential answers.
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

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # #                       EVALUATION                      # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

n_correct = 0
n_incorrect = 0

all_results = list()

# Gets one of the datasets to evaluate against.
lines = get_training_lines()
if TESTING_FLAG:
  lines = get_testing_lines()

# Predicts each one of the values, calculates a confidence value and prints it out, along with whether or not
# it was right. 
for tsv_line in lines:
  if (n_correct + n_incorrect) % 50 == 1:
    print("Making Progress! - ", (n_correct + n_incorrect),' - ' ,  (n_correct * 100.0)/(n_correct + n_incorrect))

  result = dict()
  
  question_id = get_id(tsv_line)
  result["id"] = question_id
  question = get_question(tsv_line)

  if TESTING_FLAG:
    answers = get_answers_t(tsv_line)
  else:
    answers = get_answers(tsv_line)
  correct_answer = get_correct_answer(tsv_line)
  
  (prediction, conf) = predict(question, answers)
  result["predict"] = prediction
  result["raw"] = conf

  all_results.append(result)

  if correct_answer == prediction:
    n_correct = n_correct + 1
  else:
    n_incorrect = n_incorrect + 1
  score = (n_correct * 100.0) / (n_correct + n_incorrect)

print(all_results)

# Calculates the scaled confidence. 
for result in all_results:
  result["conf"] = scaled_confidence(result["raw"])

# Saves the results of the evaluation.
w = open(result_path, 'w')
for result in all_results:
  s = ""+result["id"]+'\t'+result["predict"]+'\t'+str(result["conf"])+"\n"
  w.write(s)
w.close()
