import xml.etree.ElementTree as ET
import re, numpy, pickle
from sys import argv
from collections import defaultdict
from functools import reduce, partial
from pathlib import Path
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict

tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
labelRegex = re.compile('^(.*):::(F|M):::(..-..):::')

# http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
class MeanEmbeddingVectorizer(object):
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

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

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

def parse_author_file(file):
    tree = ET.parse(file)
    root = tree.getroot()
    text = []
    for document in root.findall('document'):
        text += preprocess_tweet(document.text)
    # print(text)
    return text

def load_language(language_dir):
    language = {}
    for author_file in language_dir.iterdir():
        if author_file.is_file() and author_file.suffix == '.xml':
            language[author_file.stem] = parse_author_file(author_file)
    return language

def load_labels(language_dir):
    labels = {}
    file_path = language_dir.joinpath('truth.txt')
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            match = labelRegex.match(line)
            labels[match.group(1)] = match.group(2,3)
    # print(labels)
    return labels

# def iterate_dirs(path, load):
#     p = Path(path)
#     data = {}
#     for directory in p.iterdir():
#         if directory.is_dir():
#             data[directory.name] = load(directory)
#     return data

def match_text_annotation(texts, annotation,l):
    docs = []
    labels = []
    for k, v in texts.items():
        docs.append(v)
        labels.append(annotation[k][l])
    return docs, labels

def concat(doc):
    return reduce(lambda acc, val: '{} {}'.format(acc, val), doc)

def identity(x): return x

def seq_tag(doc):
    return [e for l in pos_tag(doc) for e in l]

# Read in word embeddings
def read_embeddings(directory):
    for file in directory.iterdir():
        if file.is_file() and file.suffix == '.txt' and file.stem != 'truth':
            print('reading embeddings from {}'.format(file.name))
            with open(file, "rb") as lines:
                embeddings = {
                    line.split()[0]: numpy.array(map(float, line.split()[1:]))
                       for line in lines}
            print('done reading embeddings')
    return embeddings

# Turn words into embeddings, i.e. replace words by their corresponding embeddings
def document_embeddings(words, embeddings):
    vectorized_words = []
    for word in words:
        try:
            vectorized_words.append(embeddings[word.lower()])
        except KeyError:
            vectorized_words.append(embeddings['UNK'])
    return numpy.array(vectorized_words)

svc = SVC(
    kernel='linear',
    C=0.9,
    gamma=0.9
)

bayes = MultinomialNB(
    alpha=0.9,
    fit_prior=False
)

wordTrigram = TfidfVectorizer(
    analyzer = 'word',
    preprocessor = identity,
    ngram_range = (3,3),
    tokenizer = identity,
    stop_words = None
)

posTrigram = TfidfVectorizer(
    analyzer = 'word',
    preprocessor = seq_tag,
    ngram_range = (3,3),
    tokenizer = identity,
    stop_words = None
)

chargram = TfidfVectorizer(
    analyzer = 'char',
    preprocessor = concat,
    ngram_range = (6,6),
    tokenizer = identity,
    stop_words = None
)

# embedded = TfidfVectorizer(
#     analyzer = 'word',
#     preprocessor = partial(document_embeddings, embeddings),
#     ngram_range = (1,1),
#     tokenizer = identity,
#     stop_words = None
# )



def fit_predict_report(
    training_data,
    annotation,
    select_label,
    embedding_vectorizer
):
    documents, labels = match_text_annotation(
        training_data,
        annotation,
        select_label
    )
    vec = FeatureUnion([
        ('wordTrigram', wordTrigram),
        ('chargram', chargram),
        # ('posTrigram', posTrigram)
        # ('embedded', embedded)
         ('EmbeddingVectorizer', embedding_vectorizer)
        # ('MeanEmbeddingVectorizer', )
    ])

    classifier = Pipeline([
        ('vec', vec),
        # ('selector', selector),
        ('cls', svc)
    ])

    predictions = cross_val_predict(
        estimator=classifier,
        X=documents,
        y=labels,
        cv=10,
        verbose=3,
        n_jobs=-1
    )

    print(classification_report(labels, predictions, digits=4))

def run(lang):
    directory = Path('./training/' + lang)
    training_data = load_language(directory)
    annotation = load_labels(directory)
    embeddings = read_embeddings(directory)
    fit_predict_report(
        training_data,
        annotation,
        0,
        TfidfEmbeddingVectorizer(embeddings)
    )
    fit_predict_report(
        training_data,
        annotation,
        1,
        MeanEmbeddingVectorizer(embeddings)
    )

run('english')
run('dutch')
run('italian')
run('spanish')


# english none                                      gender  7754
# english MeanEmbeddingVectorizer embeddings.pickle gender  7757
# english tfidembeddingvecgtorizer embeddings.pickle gender 7849

# english none                                      age  6983
# english MeanEmbeddingVectorizer embeddings.pickle age  7153
# english tfidembeddingvecgtorizer embeddings.pickle age 6810


# initially I use a lot of features and then try to select the relevenat ones
# features are: the preprocessed words, word trigrams, character 6-grams,
# pos-tags and word embeddings using the binary file from assignment 4.

# remove stopwords, count out of dictionary words,


# the labels for age are only present for English and Spanish
# so I use a single language model for all four


# the labels for gender are present for all languages
# so I will train seperate models for each language


# I use a SVM with a linear kernel, because participants in the shared task
# have been succesful with it and I did not do assignment 5 and no little of
# multi layer neural networks.

