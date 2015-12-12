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

TESTING_FLAG = False

embedding_root = '/home/u/fall12/gward/Desktop/AI/running/embeddings/'
embedding_file = 'friday-6pm'
embedding_path = embedding_root + embedding_file + ".npy"
dictionary_path = embedding_root + embedding_file + "-dict.csv"
result_path = embedding_root + embedding_file + "-TRAIN.tsv"
if TESTING_FLAG:
  result_path = embedding_root + embedding_file + "-TEST.tsv"

vocabulary_size = 8000

dictionary = dict()
reverse_dictionary = dict()

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 2       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 30     # Random set of words to evaluate similarity on.
valid_window = 100

num_steps = 6000001

final_embeddings = None





if os.path.isfile(dictionary_path):
  csv_reader = csv.reader(open(dictionary_path))
  print(csv_reader)
  for key, val in csv_reader:
      dictionary[key] = val
else:
  # Step 1: Defines the corpus texts, and then uploads them all.
  words = get_all_wiki_files_as_list()

  # Step 2: Build the dictionary and replace rare words with UNK token.
  def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
      dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
      if word in dictionary:
        index = dictionary[word]
      else:
        index = 0  # dictionary['UNK']
        unk_count = unk_count + 1
      data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
  data, count, dictionary, reverse_dictionary = build_dataset(words)
  del words  # Hint to reduce memory.
  print('Most common words (+UNK)', count[:25])
  print('Sample data', data[:10])
  data_index = 0

  w = csv.writer(open(dictionary_path, "w"))
  for key, val in dictionary.items():
    w.writerow([key, val])

if os.path.isfile(embedding_path):
  final_embeddings = np.load(embedding_path)
else:
  # Step 4: Function to generate a training batch for the skip-gram model.
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
  batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
  for i in range(8):
    print(batch[i], '->', labels[i, 0])
    print(reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])

 



  # Step 5: Build and train a skip-gram model.
  # Only pick dev samples in the head of the distribution.
  valid_examples = np.array(random.sample(np.arange(valid_window), valid_size))
  num_sampled = 64    # Number of negative examples to sample.

  graph = tf.Graph()
  with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    # Construct the variables.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                       num_sampled, vocabulary_size))
    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)


  # Step 6: Begin training
  
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
      # note that this is expensive (~20% slowdown if computed every 500 steps)
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

    np.save(embedding_path, final_embeddings)




print(final_embeddings)
print(len(final_embeddings))
print(len(final_embeddings[0]))
print("HELLO WOLRD")
print(vocabulary_size)

#print(reverse_dictionary)

def embed(text):
  l = text.split(" ")
  encoding = np.ndarray(shape=(1, vocabulary_size), dtype=np.int32)
  for i in range(0, len(encoding[0])):
    encoding[0,i] = 0
  for word in l:
    index = dictionary.get(word)
    if index != None:
      encoding[0, index] = encoding[0, index] + 1;

  # print("ENCODING")
  # print(encoding)
  result = np.dot(encoding, final_embeddings)
  return result

def flatten(a):
  b = np.ndarray(shape=(len(a[0])), dtype=np.float64)
  for i in range(0, len(a[0])):
    b[i] = a[0,i]
  return b

print("HERE!!!!")
print(embed("cells can used one may water many"))
print(flatten(embed("cells can used one may water many")))

# This currently doesn't do anything.

# correct_q_a_pairs = list()
# incorrect_q_a_pairs = list()

# for tsv_line in get_training_lines():
#   q = get_question(tsv_line)
#   a = get_answers(tsv_line)
#   ca = get_correct_answer(tsv_line)
#   for key in a.keys():
#     if key == ca:
#       t = (embed(q), embed(a[key]))
#       correct_q_a_pairs.append(t)
#     else:
#       incorrect_q_a_pairs.append({embed(q), embed(a[key])})

# Train this part.

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
    # print ("    ", a ,"-> ",score)
    scores.append(score)

    if (score > best_score):
      best_score = score
      best_ans = a

    scores = sorted(scores)

  raw_conf = raw_confidence(scores)

  return (best_ans, raw_conf)


n_correct = 0
n_incorrect = 0

all_results = list()

lines = get_training_lines()
if TESTING_FLAG:
  lines = get_testing_lines()

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

for result in all_results:
  result["conf"] = scaled_confidence(result["raw"])

print(all_results)

w = open(result_path, 'w')
for result in all_results:
  s = ""+result["id"]+'\t'+result["predict"]+'\t'+str(result["conf"])+"\n"
  w.write(s)
w.close()
