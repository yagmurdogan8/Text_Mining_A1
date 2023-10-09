from sklearn.datasets import fetch_20newsgroups  # to import the newsgroup data directly from internet
# We could also download the dataset file extract it and use it with scikitlearn's load_files att. instead of importing
# directly from the internet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# Tutorial example 4, but in the assignment we are using all 20 as question 1 asked to
# categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

# Below we are getting the newsgroup data directly from internet using scikitlearn

twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)

print(twenty_train.keys())  # we can see the keys of the dictionary
print(twenty_test.keys())  # same as above

# twenty_train and twenty_test data are stored as dictionary objects.

# the length of the training data can be shown as:
# print(len(twenty_train.data))
# Let’s print the first lines of the first loaded file:
# print("\n".join(twenty_train.data[0].split("\n")[:3]))
# print(twenty_train.target_names[twenty_train.target[0]])
#
# print(twenty_train.target[:10])

for t in twenty_train.target[:20]:
    print("Target Name:", twenty_train.target_names[t])

# Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer,
# which builds a dictionary of features and transforms documents to feature vectors:
# How many words (CountVectorizer() does)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_test_counts = count_vect.fit_transform(twenty_test.data)
print("train counts shape", X_train_counts.shape)
print("test counts shape", X_test_counts.shape)

# CountVectorizer supports counts of N-grams of words or consecutive characters.
# Once fitted, the vectorizer has built a dictionary of feature indices:
print("count vect vocab (u'algorithm')", count_vect.vocabulary_.get(u'algorithm'))
# The index value of a word in the vocabulary is linked to its frequency in the whole training corpus

# Codelet below is the Term Frequency (tf). TF is the frequency of a given word in the text
# idf is false to disable idf feat.
tf_transformer_train = TfidfTransformer(use_idf=False, norm=None).fit(X_train_counts)
X_train_tf = tf_transformer_train.transform(X_train_counts)

tf_transformer_test = TfidfTransformer(use_idf=False, norm=None).fit(X_test_counts)
X_test_tf = tf_transformer_test.transform(X_test_counts)

print("train tf shape", X_train_tf.shape)
print("test tf shape", X_test_tf.shape)

# TFID is the words appears most in text
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print("tfidf shape", X_train_tfidf.shape)

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
