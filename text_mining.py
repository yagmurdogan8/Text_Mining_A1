from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups  # to import the newsgroup data directly from internet
# We could also download the dataset file extract it and use it with scikitlearn's load_files att. instead of importing
# directly from the internet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# Tutorial example 4, but in the assignment we are using all 20 as question 1 asked to
# categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

# Below we are getting the newsgroup data directly from internet using scikitlearn

twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)

print(twenty_train.keys())  # we can see the keys of the dictionary
print(twenty_test.keys())  # same as above

# twenty_train and twenty_test data are stored as dictionary objects.

# the length of the training data can be shown as:
print(len(twenty_train.data))
# Let’s print the first lines of the first loaded file:
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])
print(twenty_train.target[:10])

# names of the categories
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
# print("count vect vocab (u'algorithm')", count_vect.vocabulary_.get(u'algorithm'))
# The index value of a word in the vocabulary is linked to its frequency in the whole training corpus

# Codelet below is the Term Frequency (tf). TF is the frequency of a given word in the text
# idf is false to disable idf feature
# tf_transformer_train = TfidfTransformer(use_idf=False, norm=None).fit(X_train_counts)
# X_train_tf = tf_transformer_train.transform(X_train_counts)
#
# tf_transformer_test = TfidfTransformer(use_idf=False, norm=None).fit(X_test_counts)
# X_test_tf = tf_transformer_test.transform(X_test_counts)

print("------------------------------------------------------")
print("The Classification Algorithms & Their Metric Scores:")
# Naive Bayes
naive_pipeline = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('clf', MultinomialNB())])
naive_pipeline.fit(twenty_train.data, twenty_train.target)
naive_predicted = naive_pipeline.predict(twenty_train.data)
print("Precision of NAive Bayes:", precision_score(twenty_train.target, naive_predicted, average='weighted'))
print("Recall of Naive Bayes:", recall_score(twenty_train.target, naive_predicted, average='weighted'))
print("F1 score of Naive bayes:", f1_score(twenty_train.target, naive_predicted, average='weighted'))
print("Accuracy score of Naive Bayes:", accuracy_score(twenty_train.target, naive_predicted))

print(metrics.classification_report(twenty_train.target, naive_predicted, target_names=twenty_train.target_names))
print("------------------------------------------------------")

# Stochastic Gradient Descent Classifier
sgdc_pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                                                random_state=42, max_iter=5, tol=None))])
sgdc_pipeline.fit(twenty_train.data, twenty_train.target)
sgdc_predicted = sgdc_pipeline.predict(twenty_train.data)
print("Precision of SGDC:", precision_score(twenty_train.target, sgdc_predicted, average='weighted'))
print("Recall of SGDC:", recall_score(twenty_train.target, sgdc_predicted, average='weighted'))
print("F1 score of SGDC:", f1_score(twenty_train.target, sgdc_predicted, average='weighted'))
print("Accuracy score of SGDC:", accuracy_score(twenty_train.target, sgdc_predicted))

print(metrics.classification_report(twenty_train.target, sgdc_predicted, target_names=twenty_train.target_names))
print("------------------------------------------------------")

# Decision Tree Classifier
dtc_pipeline = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ("clf", DecisionTreeClassifier())])
dtc_pipeline.fit(twenty_train.data, twenty_train.target)
dtc_predicted = dtc_pipeline.predict(twenty_train.data)
print("Precision of DTC:", precision_score(twenty_train.target, dtc_predicted, average='weighted'))
print("Recall of DTC:", recall_score(twenty_train.target, dtc_predicted, average='weighted'))
print("F1 score of DTC:", f1_score(twenty_train.target, dtc_predicted, average='weighted'))
print("Accuracy score of DTC:", accuracy_score(twenty_train.target, dtc_predicted))

print(metrics.classification_report(twenty_train.target, dtc_predicted, target_names=twenty_train.target_names))

print("------------------------------------------------------")

# The best combination of classifiers and features is countvectorizer with dtc so from now on we will be
# investigating those for the question 4.

# Lowercase : just adding lowercase=True (default) makes the text converted to lowercase. So, let us just add
# lowercase=false for seeing the difference

lower_dtc_pipeline = Pipeline([('vect', CountVectorizer(lowercase=True)),
                               ("clf", DecisionTreeClassifier())])
lower_dtc_pipeline.fit(twenty_train.data, twenty_train.target)
lower_dtc_predicted = lower_dtc_pipeline.predict(twenty_train.data)
print("Lowercased Precision of DTC:", precision_score(twenty_train.target, lower_dtc_predicted, average='weighted'))
print("Lowercased Recall of DTC:", recall_score(twenty_train.target, lower_dtc_predicted, average='weighted'))
print("Lowercased F1 score of DTC:", f1_score(twenty_train.target, lower_dtc_predicted, average='weighted'))
print("Lowercased Accuracy score of DTC:", accuracy_score(twenty_train.target, lower_dtc_predicted))

nlower_dtc_pipeline = Pipeline([('vect', CountVectorizer(lowercase=False)),
                         ("clf", DecisionTreeClassifier())])
nlower_dtc_pipeline.fit(twenty_train.data, twenty_train.target)
nlower_dtc_predicted = nlower_dtc_pipeline.predict(twenty_train.data)
print("Not Lowercased Precision of DTC:", precision_score(twenty_train.target, nlower_dtc_predicted, average='weighted'))
print("Not Lowercased Recall of DTC:", recall_score(twenty_train.target, nlower_dtc_predicted, average='weighted'))
print("Not Lowercased F1 score of DTC:", f1_score(twenty_train.target, nlower_dtc_predicted, average='weighted'))
print("Not lowercased Accuracy score of DTC:", accuracy_score(twenty_train.target, nlower_dtc_predicted))

print("------------------------------------------------------")

# Stop- words : We can either create a list of stop words ourselves or import sklearn's english stop words lib
# We are going to use second method

stop_dtc_pipeline = Pipeline([('vect', CountVectorizer(stop_words='english')),
                         ("clf", DecisionTreeClassifier())])
stop_dtc_pipeline.fit(twenty_train.data, twenty_train.target)
stop_dtc_predicted = stop_dtc_pipeline.predict(twenty_train.data)
print("With stop words Precision of DTC:", precision_score(twenty_train.target, stop_dtc_predicted, average='weighted'))
print("With stop words Recall of DTC:", recall_score(twenty_train.target, stop_dtc_predicted, average='weighted'))
print("With stop words F1 score of DTC:", f1_score(twenty_train.target, stop_dtc_predicted, average='weighted'))
print("With stop words Accuracy score of DTC:", accuracy_score(twenty_train.target, stop_dtc_predicted))

print("------------------------------------------------------")

# Ngram_range : We are going to use (1,1) unigram comp. (1,2) unigram and bigram

ngram_dtc_pipeline = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=(1, 1))),
                         ("clf", DecisionTreeClassifier())])
ngram_dtc_pipeline.fit(twenty_train.data, twenty_train.target)
ngram_dtc_predicted = ngram_dtc_pipeline.predict(twenty_train.data)
print("'Word' Unigram Precision of DTC:", precision_score(twenty_train.target, ngram_dtc_predicted, average='weighted'))
print("'Word' Unigram Recall of DTC:", recall_score(twenty_train.target, ngram_dtc_predicted, average='weighted'))
print("'Word' Unigram F1 score of DTC:", f1_score(twenty_train.target, ngram_dtc_predicted, average='weighted'))
print("'Word' Unigram Accuracy score of DTC:", accuracy_score(twenty_train.target, ngram_dtc_predicted))

dtc_pipeline = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=(1, 2))),
                         ("clf", DecisionTreeClassifier())])
dtc_pipeline.fit(twenty_train.data, twenty_train.target)
dtc_predicted = dtc_pipeline.predict(twenty_train.data)
print("'Word' Unigram and bigram Accuracy score of DTC:", accuracy_score(twenty_train.target, dtc_predicted))

dtc_pipeline = Pipeline([('vect', CountVectorizer(analyzer='char', ngram_range=(1, 1))),
                         ("clf", DecisionTreeClassifier())])
dtc_pipeline.fit(twenty_train.data, twenty_train.target)
dtc_predicted = dtc_pipeline.predict(twenty_train.data)
print("'Char' Unigram Accuracy score of DTC:", accuracy_score(twenty_train.target, dtc_predicted))

dtc_pipeline = Pipeline([('vect', CountVectorizer(analyzer='char', ngram_range=(1, 2))),
                         ("clf", DecisionTreeClassifier())])
dtc_pipeline.fit(twenty_train.data, twenty_train.target)
dtc_predicted = dtc_pipeline.predict(twenty_train.data)
print("'Char' Unigram and bigram Accuracy score of DTC:", accuracy_score(twenty_train.target, dtc_predicted))

print("------------------------------------------------------")

# Max_feature : Limits the process to the max feature that has been assigned

dtc_pipeline = Pipeline([('vect', CountVectorizer(max_features=1000)),
                         ("clf", DecisionTreeClassifier())])
dtc_pipeline.fit(twenty_train.data, twenty_train.target)
dtc_predicted = dtc_pipeline.predict(twenty_train.data)
print("Max features of 1000, Accuracy score of DTC:", accuracy_score(twenty_train.target, dtc_predicted))

dtc_pipeline = Pipeline([('vect', CountVectorizer(max_features=3000)),
                         ("clf", DecisionTreeClassifier())])
dtc_pipeline.fit(twenty_train.data, twenty_train.target)
dtc_predicted = dtc_pipeline.predict(twenty_train.data)
print("Max features of 3000, Accuracy score of DTC:", accuracy_score(twenty_train.target, dtc_predicted))

dtc_pipeline = Pipeline([('vect', CountVectorizer(max_features=500)),
                         ("clf", DecisionTreeClassifier())])
dtc_pipeline.fit(twenty_train.data, twenty_train.target)
dtc_predicted = dtc_pipeline.predict(twenty_train.data)
print("Max features of 500, Accuracy score of DTC:", accuracy_score(twenty_train.target, dtc_predicted))