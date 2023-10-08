from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# Tutorial example 4, but in the assignment we are using all 20 as question 1 asked to
# categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train', categories=None, shuffle=True, random_state=42)

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
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)

# CountVectorizer supports counts of N-grams of words or consecutive characters.
# Once fitted, the vectorizer has built a dictionary of feature indices:
print(count_vect.vocabulary_.get(u'algorithm'))