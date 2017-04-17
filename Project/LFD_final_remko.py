import xml.etree.ElementTree as ET
import re, numpy, pickle, sys
from sys import argv
from collections import defaultdict
from functools import reduce, partial
from pathlib import Path
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.base import BaseEstimator

# set to true to use GridSearchCV to find optimal parameters for SVC
param_search = False

# check command line arguments for training and test directories
if len(sys.argv) == 2:
    train_dir = sys.argv[1]
    test_dir = None
elif len(sys.argv) == 3:
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
else:
    raise ValueError('you need to specify a training dir or both a training dir and a test dir')

tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
labelRegex = re.compile('^(.*):::(F|M):::(..-..):::')

# after http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
# vectorizer that computes the mean vector for the embeddings representing the words
class MeanEmbeddingVectorizer(BaseEstimator):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return numpy.array([
            numpy.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [numpy.zeros(self.dim)], axis=0)
            for words in X
        ])

# vectorizers that computes the weighted mean vector for the embedings representing the words
# the weighting is inverse proportional to the document frequency
class TfidfEmbeddingVectorizer(BaseEstimator):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.values())
        self._trick = None
    # trick to enable pickling by multiprocess
    def identity(self, x): return x
    # trick to enable pickling by multiprocess
    def mx(self): return self._trick
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=self.identity)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        self._trick = max(tfidf.idf_)
        self.word2weight = defaultdict(self.mx,[(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        return numpy.array([
                numpy.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [numpy.zeros(self.dim)], axis=0)
                for words in X
            ])

# the tweets are preprocessed:
# trim whitespace
# replace retweets by a single RT
# replace @username with USR
# replace urls with URL
# tokenize using the nltk tweettokenizer removing user handles and limiting
#   repititions to three
# I leave hashtags and emoticons as they are
def preprocess_tweet(tweet):
    trimmed = tweet.strip()
    # if pattern 'RT @' until the end of the tweet is replace by RT
    replaced_RT = re.sub('RT @.*','RT',trimmed)
    replaced_USR = re.sub('@username','USR',replaced_RT)
    replaced_URL = re.sub('https?://[a-zA-Z0-9/.]+','URL',replaced_USR)
    tokens = tokenizer.tokenize(replaced_URL)
    return ['STA'] + tokens + ['END']

# parses the xml in the author file calls the preprocessing of tweets and concats
# all the text
def parse_author_file(file):
    tree = ET.parse(file)
    root = tree.getroot()
    text = []
    for document in root.findall('document'):
        text += preprocess_tweet(document.text)
    # print(text)
    return text

# builds a dictionary of the authors guids and the preprocessed, tokenized
# and concatenated texts
def load_language(language_dir):
    language = {}
    print('loading language from {}'.format(language_dir.name))
    for author_file in language_dir.iterdir():
        if author_file.is_file() and author_file.suffix == '.xml':
            language[author_file.stem] = parse_author_file(author_file)
    print('done loading language')
    return language

# parses the gold labels
def load_labels(language_dir):
    labels = {}
    file_path = language_dir.joinpath('truth.txt')
    print('loading labels from {}'.format(language_dir.name))
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            match = labelRegex.match(line)
            labels[match.group(1)] = match.group(2,3)
    print('done loading labels')
    return labels

# gets the labels for all the authors and returns an array
# with the texts, with the gender labels and with the age labels
# the three arrays are alligned
def match_text_annotation(texts, annotation):
    docs = []
    labels_gender = []
    labels_age = []
    for k, v in texts.items():
        docs.append(v)
        labels_gender.append(annotation[k][0])
        labels_age.append(annotation[k][1])
    return docs, labels_gender, labels_age

# undo the tokenization for the character n-grams
def concat(doc):
    return reduce(lambda acc, val: '{} {}'.format(acc, val), doc)

# need a named identity function in the standard multiprocessing
def identity(x): return x

# reeds the plain text pretrained word embeddings that are in the vocab used in
# in the tweets into a numpy array
# it prints some stats about the number of words in the vocab not in the embeddings
def read_embedding_file(file, vocab):
    print('reading embeddings from {}'.format(file.name))
    with open(file, "rb") as lines:
        embeddings = {line.split()[0].decode('utf-8', 'ignore'): numpy.array(list(map(float, line.split()[1:])))
            for line in lines if line.split()[0].decode('utf-8', 'ignore') in vocab }
    not_in_embeddings = len([x for x in vocab if x not in embeddings])
    print('{} in vocab and {} not in embeddings'.format(len(vocab), not_in_embeddings))
    print('done reading embeddings')
    return embeddings

# either read all embedding files in the training directory or read a single file
def read_embeddings(path, vocab):
    embeddings_list = []
    if path.is_dir():
        for file in path.iterdir():
            if file.is_file() and (file.suffix == '.txt' or file.suffix == '.vec' or file.suffix == '.vecs') and file.stem != 'truth':
                embeddings_list.append((file.name, read_embedding_file(file, vocab)))
    if path.is_file():
        embeddings_list.append((path.name, read_embedding_file(path, vocab)))
    return embeddings_list

# returns a set with all the words in all the preprocessed tweets of all the authors
def get_vocab(data):
    # return a set with all the words in the texts
    # for all authors in the language
    return set(reduce(lambda acc, val: acc + val, data.values()))

# writes out a truth file in for each of the languages in the test directory
# the file contains the guid of the author the gender and the age if available in the
# training data's annotation
def write_truth_file(test_directory, test_authors, gender_predictions, age_predictions):
    f = open('{}/{}'.format(test_directory, 'truth.txt'), 'w')
    if gender_predictions is not None and age_predictions is not None:
        for author, gender, age in zip(test_authors, gender_predictions, age_predictions):
            f.write('{}:::{}:::{}\n'.format(author, gender, age))
    elif gender_predictions is not None:
        for author, gender in zip(test_authors, gender_predictions):
            f.write('{}:::{}:::{}\n'.format(author, gender, 'XX-XX'))

# word trigram vector using tfidf
wordTrigram = TfidfVectorizer(
    analyzer = 'word',
    preprocessor = identity,
    ngram_range = (3,3),
    tokenizer = identity,
    stop_words = None
)

# character 6-gram vector using tfidf
chargram = TfidfVectorizer(
    analyzer = 'char',
    preprocessor = concat,
    ngram_range = (6,6),
    tokenizer = identity,
    stop_words = None
)

# f1 scorer for the gridsearchcv it avarage the score within a class
f1_scorer = make_scorer(f1_score, average='micro')

# either uses a gridsearch to find an optimal config and uses that to predict
# or it uses parameter settings passed to it. The global var param_search selects
# either uses a ten-fold cross validation or a test data set if one is available
def fit_predict(documents, labels, embed, test_documents, embeddings, c_value):
    print('embedding size {}'.format(len(embeddings)))
    print('using {}'.format(embed))

    if embed == 'mean':
        embedding_vectorizer = MeanEmbeddingVectorizer(embeddings)
    elif embed == 'tfidf':
        embedding_vectorizer = TfidfEmbeddingVectorizer(embeddings)
    else:
        embedding_vectorizer = None

    vec = FeatureUnion([
        ('wordTrigram', wordTrigram),
        ('chargram', chargram),
        ('embedding_vectorizer', embedding_vectorizer)
    ])

    svc = SVC(kernel='linear', C=c_value)

    classifier = Pipeline([
        ('vec', vec),
        ('cls', svc)
    ])

    if not test_documents:
        if param_search and len(embeddings) > 0:
            param_grid = [
              {
                'cls__C': numpy.logspace(-2, 1, 10), # numpy.arange(0.1, 2.2, 0.1), numpy.logspace(-3, 2, 6)
                'cls__kernel': ['linear'],
                'vec__embedding_vectorizer': [
                    MeanEmbeddingVectorizer(embeddings),
                    TfidfEmbeddingVectorizer(embeddings)
                ]
              },
              {
                'cls__C': numpy.logspace(-2, 1, 10), # numpy.logspace(-3, 2, 6),
                'cls__gamma': numpy.logspace(-5, -2, 4),
                'cls__kernel': ['rbf'],
                'vec__embedding_vectorizer': [
                    MeanEmbeddingVectorizer(embeddings),
                    TfidfEmbeddingVectorizer(embeddings)
                ]
              },
             ]

            clf = GridSearchCV(classifier, param_grid, f1_scorer, cv=5, n_jobs = -1)

            clf.fit(documents, labels)

            print(clf.best_params_)
            print(clf.best_score_)

            return cross_val_predict(
                    estimator=clf.best_estimator_,
                    X=documents,
                    y=labels,
                    cv=10,
                    verbose=3,
                    n_jobs=-1
                )
        else:
            return cross_val_predict(
                    estimator=classifier,
                    X=documents,
                    y=labels,
                    cv=10,
                    verbose=3,
                    n_jobs=-1
                )
    # if test_data
    else:
        classifier.fit(documents, labels)
        return classifier.predict(test_documents)

# run the train test report process for a single language
# loads the embeddings if they are used
# loads test data if a directory is provided
# does age prediction if annotation is provided
# prints the scores or writes a truth file to a test directory if provided
def run(lang, embed_gender, embed_age, c_value, embed_file=None):
    test_data = None
    test_authors = None
    test_documents = None
    gender_predictions = None
    age_predictions = None

    training_directory = Path('{}/{}'.format(train_dir, lang))
    training_data = load_language(training_directory)
    training_annotation = load_labels(training_directory)
    vocab = get_vocab(training_data)

    if test_dir:
        test_directory = Path('{}/{}'.format(test_dir, lang))
        test_data = load_language(test_directory)
        test_authors = [item[0] for item in test_data.items()]
        test_documents = [item[1] for item in test_data.items()]
        vocab = vocab.union(get_vocab(test_data))

    if embed_gender == 'none' and embed_age == 'none':
        embeddings = [('none', [])]
    elif embed_file:
        path = training_directory.joinpath(embed_file)
        embeddings = read_embeddings(path, vocab)
    else:
        embeddings = read_embeddings(training_directory, vocab)

    training_documents, training_labels_gender, training_labels_age = match_text_annotation(training_data, training_annotation)

    for embedding in embeddings:
        print(embedding[0])
        print('start predict gender for {}'.format(lang))
        gender_predictions = fit_predict(
            training_documents,
            training_labels_gender,
            embed_gender,
            test_documents,
            embedding[1],
            c_value
        )
        if training_labels_age[1] != 'XX-XX':
            print('start predict age for {}'.format(lang))
            age_predictions = fit_predict(
                training_documents,
                training_labels_age,
                embed_age,
                test_documents,
                embedding[1],
                c_value
            )
        else:
            print('no age labels')

        if test_dir:
            write_truth_file(test_directory, test_authors, gender_predictions, age_predictions)
        else:
            if gender_predictions != None:
                print(classification_report(training_labels_gender, gender_predictions, digits=4))
            if age_predictions != None:
                print(classification_report(training_labels_age, age_predictions, digits=4))

# runs the program for the languages specified with the optimal settings specified
run('english', 'tfidf', 'tfidf', 1.0, 'glove.twitter.27B.200d.txt')
run('dutch', 'tfidf', 'none', 2.1, 'rob-nl.txt')
run('italian', 'none', 'none', 1.0)
run('spanish', 'mean', 'tfidf', 1.0, 'wiki.es.vec')