from sklearn.datasets import fetch_20newsgroups

# Tutorial example 4, but in the assignment we are using 20
# categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train', categories=None, shuffle=True, random_state=42)

