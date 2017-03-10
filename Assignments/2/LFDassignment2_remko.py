import argparse
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, make_scorer, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.model_selection import cross_val_score, cross_val_predict

stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

parser = argparse.ArgumentParser(
    prog='naive bayes classifier',
    description='classifies reviews by sentiment of topic'
)

parser.add_argument(
    'train_file',
    type=str,
    help='file name of the training file',
    metavar='training file'
)

parser.add_argument(
    'test_file',
    default=None,
    nargs='?',
    type=str,
    help='file name of the test file',
    metavar='test file'
)

parser.add_argument(
    '-use_sentiment',
    default=False,
    type=bool,
    dest='use_sentiment',
    help='boolean to select the use of sentiment labels',
    metavar='use sentiment',
    choices=[True, False]
)

parser.add_argument(
    '-algo',
    default='bayes',
    type=str,
    dest='algo',
    help='string to select algorithm used',
    metavar='algorithm',
    choices=['bayes', 'tree', 'neighbour']
)

args = parser.parse_args()

print(args)

# This function takes a filename and a boolean that selects
# whether the sentiment or topic labels are used.
# It returns two array's one containing the review texts
# and another containing the labels.
def read_corpus(corpus_file, use_sentiment):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()

            documents.append(tokens[3:])

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append( tokens[1] )
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append( tokens[0] )
    return documents, labels



# the wordnet lemmatizer uses the wordnet tags NOUN, VERB, ADV, ADJ
# the pos tagger outputs Penn Treebank tags, so we convert
def mapPosToWordNetLem(token, tag):
    if tag == 'VB' or tag == 'VBD' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ':
        return lemmatiser.lemmatize(token, pos='v')
    if tag == 'RB' or tag == 'RBR' or tag == 'RBS':
        return lemmatiser.lemmatize(token, pos='r')
    if tag == 'JJ' or tag == 'JJR' or tag == 'JJS':
        return lemmatiser.lemmatize(token, pos='s')
    return lemmatiser.lemmatize(token, pos='n')


def lemmatise(doc):
    # pos_tag returns a (token, tag) tuple, that needs to be unpacked
    return map(lambda x: mapPosToWordNetLem(*x), pos_tag(doc))

def stem(doc):
    return map(stemmer.stem, doc)

# a dummy function that just returns its input
def identity(x):
    return x

# Combines the vectorizer with a Naive Bayes classifier
if args.algo == 'bayes':
    classifier = Pipeline(
        [('vec', TfidfVectorizer(
            preprocessor = lemmatise,
            tokenizer = identity,
            stop_words = 'english'
        )),
        ('cls', MultinomialNB(
            alpha=0.9,
            fit_prior=False
        ))])
if args.algo == 'tree':
    classifier = Pipeline(
        [('vec', CountVectorizer(
            preprocessor = lemmatise,
            tokenizer = identity,
            stop_words = 'english'
        )),
        ('cls', DecisionTreeClassifier(
         criterion='gini',
         splitter='best',
         max_features=None,
         presort=False,
         max_depth=50,
         min_samples_split=5,
         min_samples_leaf=1,
         max_leaf_nodes=150
        ))])
if args.algo == 'neighbour':
    classifier = Pipeline(
        [('vec', TfidfVectorizer(
            preprocessor = stem,
            tokenizer = identity,
            stop_words = 'english'
        )),
        ('cls', KNeighborsClassifier(
            n_neighbors=49,
            weights='uniform',
            algorithm='brute',
            metric='minkowski',
            p=1,
            metric_params=None,
            n_jobs=4
        ))])

if args.test_file == None:
    documents, labels = read_corpus(args.train_file, args.use_sentiment)

    predictions = cross_val_predict(
        estimator=classifier,
        X=documents,
        y=labels,
        cv=4,
        verbose=3
    )

    print(classification_report(labels, predictions, digits=4))
else:
    documentsTrain, labelsTrain = read_corpus(args.train_file, args.use_sentiment)
    documentsTest, labelsTest = read_corpus(args.test_file, args.use_sentiment)
    classifier.fit(documentsTrain, labelsTrain)
    predictions = classifier.predict(documentsTest)

    print(classification_report(labelsTest, predictions, digits=4))


