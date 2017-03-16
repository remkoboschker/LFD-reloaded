import sys
from functools import reduce

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.cluster import KMeans

stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

def read_corpus(corpus_file):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            labels.append(tokens[1])
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

def word_tag(doc):
    return map(lambda x: '{}_{}'.format(x[0], x[1]), pos_tag(doc))

def lemma_tag(doc):
    return map(
        lambda x: '{}_{}'.format(mapPosToWordNetLem(*x),
        x[1]), pos_tag(doc))

def seq_tag(doc):
    return [e for l in pos_tag(doc) for e in l]

def concat(doc):
    return reduce(lambda acc, val: '{} {}'.format(acc, val), doc)

wordbipos = TfidfVectorizer(
    analyzer = 'word',
    preprocessor = seq_tag,
    ngram_range = (1,2),
    tokenizer = identity,
    stop_words = 'english'
)

wordbi = TfidfVectorizer(
    analyzer = 'word',
    preprocessor = identity,
    ngram_range = (1,2),
    tokenizer = identity,
    stop_words = 'english'
)

chargram = TfidfVectorizer(
    analyzer = 'char',
    preprocessor = concat,
    ngram_range = (6,6),
    tokenizer = identity,
    stop_words = 'english'
)

km = KMeans(
    n_clusters=2,
    n_init=10,
    n_jobs=1
)

clusters = Pipeline([
    ('wordbi', wordbi),
    ('km', km)
])

cls = SVC(
    kernel='linear',
    C=0.9,
    gamma=0.9
)

vec = FeatureUnion([
    ('chargram', chargram),
    ('clusters',clusters)
])

classifier = Pipeline([
    ('vec', vec),
    # ('selector', selector),
    ('cls', cls)
])




if len(sys.argv) == 2:
    documents, labels = read_corpus(sys.argv[1])


    predictions = cross_val_predict(
        estimator=classifier,
        X=documents,
        y=labels,
        cv=4,
        verbose=3,
        n_jobs=4
    )

    print(classification_report(labels, predictions, digits=4))

else:
    documentsTrain, labelsTrain = read_corpus(sys.argv[1])
    documentsTest, labelsTest = read_corpus(sys.argv[2])

    classifier.fit(documentsTrain, labelsTrain)
    predictions = classifier.predict(documentsTest)

    print(classification_report(labelsTest, predictions, digits=4))


