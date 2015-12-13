We used the Tensorflow framework (by Google) to create a really simple skip-gram model which trained on a wide window (size 5) on relevant wikipedia texts (determined through keywords) using a naieve similarity model between questions and answers after embeddings were calculated using the tensorflow model. 

Step 1: Training and Corpus Selection
 
We used a libarary (TOPIA) to extract keywords from our training data.  We used these keywords to download and simplify wikipedia articles on each, using careful parsing to ellimanate responses which were not found, and following redirects to relevant pages. We ellimanated stop words, and got rid of plurals.  Note that this use of keywords was only done in the training phase, in generating our corpus text.  The result was 38MB of text which we treated as a unified corpus text for our validation and testing phases.

Step 2: Training word embeddings based on wikipedia subset corpus

We used a very basic Tensorflow workflow (modified from an example on their site) to train a 5-wide skip-gram model on this corpus text. We used clever memoization of our parameters to minimize the running time and ran many different embeddings across four of the iMacs in Vertica (massively paralellizing this work across a total of 24 cores). We saved each embedding for later evaluation, and set up a model which can easily save and load embeddings for manipulation and prediction.

Step 3: Prediction and Confidence Estimation

In order to apply our model, we represented answers and questions with n words as n-hot vectors in the word space, and then retrieved back vectors with their embedded vectorization using the precomputed word embeddings.  We then performed a simple dot-product similarity comparison, and chose the answer whose embedding representation was closest to the question's embedded representation. We estimated confidence using an ad-hoc function that compares the dot-product score of the best answer to the 
dot-product values of inferior answers, then normalized this on a scale from 1 to 100.

The best embeddings achieved an accuracy of 32% on the training set, 29% on the test set, which is pretty reasonable, given that we didn't actually do much to augment the Tensorflow model, besides figuring out how to select our corpus text creatively.

Step 4: Combine with other model.
