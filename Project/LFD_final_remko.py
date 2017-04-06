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
        # for words in X:
        #     for w in words:
        #         if w in self.word2vec:
        #             print(len(self.word2vec[w]))
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
    print('loading language from {}'.format(language_dir.name))
    for author_file in language_dir.iterdir():
        if author_file.is_file() and author_file.suffix == '.xml':
            language[author_file.stem] = parse_author_file(author_file)
    print('done loading language')
    return language

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

# def iterate_dirs(path, load):
#     p = Path(path)
#     data = {}
#     for directory in p.iterdir():
#         if directory.is_dir():
#             data[directory.name] = load(directory)
#     return data

def match_text_annotation(texts, annotation):
    docs = []
    labels_gender = []
    labels_age = []
    for k, v in texts.items():
        docs.append(v)
        labels_gender.append(annotation[k][0])
        labels_age.append(annotation[k][1])
    return docs, labels_gender, labels_age

def concat(doc):
    return reduce(lambda acc, val: '{} {}'.format(acc, val), doc)

def identity(x): return x

def seq_tag(doc):
    return [e for l in pos_tag(doc) for e in l]

# Read in word embeddings
def read_embeddings(directory, vocab):
    for file in directory.iterdir():
        if file.is_file() and (file.suffix == '.txt' or file.suffix == '.vec' or file.suffix == '.vecs') and file.stem != 'truth':
            print('reading embeddings from {}'.format(file.name))
            with open(file, "rb") as lines:
                embeddings = {line.split()[0].decode('utf-8', 'ignore'): numpy.array(list(map(float, line.split()[1:])))
                    for line in lines if line.split()[0].decode('utf-8', 'ignore') in vocab }
            not_in_embeddings = len([x for x in vocab if x not in embeddings])
            print('{} in vocab and {} not in embeddings'.format(len(vocab), not_in_embeddings))
            print('done reading embeddings')
    return embeddings

def get_vocab(data):
    # return a set with all the words in the texts
    # for all authors in the language
    return set(reduce(lambda acc, val: acc + val, data.values()))

# Turn words into embeddings, i.e. replace words by their corresponding embeddings
# def document_embeddings(words, embeddings):
#     vectorized_words = []
#     for word in words:
#         try:
#             vectorized_words.append(embeddings[word.lower()])
#         except KeyError:
#             vectorized_words.append(embeddings['UNK'])
#     return numpy.array(vectorized_words)

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



def fit_predict_report(documents, labels, embed, embeddings):
    print('embedding size {}'.format(len(embeddings)))
    print('using {}'.format(embed))
    if embed == 'none':
        vec = FeatureUnion([
            ('wordTrigram', wordTrigram),
            ('chargram', chargram)
        ])
    if embed == 'mean':
        vec = FeatureUnion([
            ('wordTrigram', wordTrigram),
            ('chargram', chargram),
            ('EmbeddingVectorizer', MeanEmbeddingVectorizer(embeddings))
        ])
    if embed == 'tfidf':
        vec = FeatureUnion([
            ('wordTrigram', wordTrigram),
            ('chargram', chargram),
            ('EmbeddingVectorizer', TfidfEmbeddingVectorizer(embeddings))
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
    vocab = get_vocab(training_data)
    embeddings = read_embeddings(directory, vocab)
    documents, labels_gender, labels_age = match_text_annotation(training_data,annotation)
    print('start predict gender for {}'.format(lang))
    fit_predict_report(documents, labels_gender, 'tfidf', embeddings)
    if labels_age[1] != 'XX-XX':
        print('start predict age for {}'.format(lang))
        fit_predict_report(documents, labels_age, 'mean', embeddings)
    else:
        print('no age labels')

run('english')
# run('dutch')
# run('italian')
# run('spanish')

# https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

# english none                              gender  7754
# english tfid glove.twitter.27B.50d.txt    gender 8223
# english mean glove.twitter.27B.50d.txt    gender 8319
# english tfid glove.twitter.27B.200d.txt   gender 8503
# english mean glove.twitter.27B.200d.txt   gender 8505
# mean fasttext wiki.en.vec 7659 21376 in vocab and 12463 not in embeddings done reading embeddings embedding size 8913
# tfidf fasttext wiki.en.vec 7944
# glove.6B.300d.txt mean 7941 21376 in vocab and 12859 not in embeddings done reading embeddings start predict gender for english embedding size 8517
# glove.6B.300d.txt tfidf 8224

# english none                            age  6983
# english tfid glove.twitter.27B.50d.txt  age  7180 21376 in vocab and 12468 not in embeddings done reading embeddings emb 8908
# english mean glove.twitter.27B.50d.txt  age  7213
# english tfid glove.twitter.27B.200d.txt age  7232
# english mean glove.twitter.27B.200d.txt age  7259
# mean fasttext wiki.en.vec 7005
# tfidf fasttext wiki.en.vec 7343
# glove.6B.300d.txt tfidf 6996
# glove.6B.300d.txt mean 7153

# dutch none                          6951
# https://github.com/clips/dutchembeddings
# dutch mean combined-320.txt gender 6951 7491 in vocab and 3033 not in embeddings done reading embeddings embedding size 4458
# dutch mean fasttext wiki.nl.vec     7429 7491 in vocab and 3133 not in embeddings done reading embeddings embedding size 4358
# dutch tfidf fasttext wiki.nl.vec 7884

# eerst https://github.com/marekrei/convertvec
# met het bestand /net/shared/rob/nlTweets/tw.vecs
# mean rob 400d  6951 in vocab and 548 not in embeddings done reading embeddings embedding size 6943
# tfidf rob 7884

# italian none 8203
# italian tfidf fasttext wiki.it.vec gender 7823 10425 in vocab and 4487 not in embeddings done reading embeddings embedding size 5938
# italian mean fasttext wiki.it.vec gender 7823
# http://hlt.isti.cnr.it/wordembeddings/

# spanish none gender 7842
# spanish node age 6390
# spanish tfidf fasttext wiki.es.vec age 7966 20045 in vocab and 8415 not in embeddings done reading embeddings embedding size 11630
# spanish tfidf fasttext wiki.es.vec gender 7992
# spanish mean fasttext wiki.es.vec age 6598
# spanish tfidf fasttext wiki.es.vec gender 8279
# http://crscardellino.me/SBWCE/
# SBW-vectors-300-min5.txt mean age 6390 gender 7992;  20045 in vocab and 4264 not in embeddings done reading embeddings start predict gender for spanish embedding size 15781
# tfidf sbw gender 8279 age 6598


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

# experiment with breadth or depth embeddings vocab vs. dimensions

