from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.model_selection import cross_val_predict
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline, FeatureUnion
from pathlib import Path
from functools import reduce
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag

lemmatiser = WordNetLemmatizer()
stemmer = PorterStemmer()

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

p = Path('../Resources/Assignment3/C50/C50test')

def identity(x):
    return x

def path_to_name_text(path):
    p = Path(path)
    author = p.name
    texts = []
    for corpus_file in p.iterdir():
        if corpus_file.is_file():
            text = []
            with open(corpus_file, encoding='utf-8') as f:
                for line in f:
                    tokens = line.strip().split()
                    text.extend(tokens)
        texts.append(text)
    return author, texts

authors = []
texts = []

for path in p.iterdir():
    if path.is_dir():
        a, ts = path_to_name_text(path)
        for t in ts:
            authors.append(a)
            texts.append(t)

def concat(doc):
    return reduce(lambda acc, val: '{} {}'.format(acc, val), doc)

def tags(doc):
    return map(lambda x: x[1], pos_tag(doc))

wordBigram = TfidfVectorizer(
    preprocessor = identity,
    tokenizer = identity,
    analyzer = 'word',
    ngram_range = (1,2),
    stop_words = 'english',
    # norm = 'l1'
)

charFiveGram = TfidfVectorizer(
    preprocessor = concat,
    tokenizer = identity,
    analyzer = 'char',
    ngram_range = (5,5),
    stop_words = 'english',
    # norm = 'l1'
)

posTags = TfidfVectorizer(
    preprocessor = tags,
    tokenizer = identity,
    analyzer = 'word',
    ngram_range = (1,1),
    stop_words = 'english',
    # norm = 'l1'
)


cls = KMeans(
    n_clusters=50,
    n_init=10,
    n_jobs=1
)

vec = FeatureUnion([
    ('wordBigram', wordBigram),
    # ('posTags', posTags)
])

#selector = VarianceThreshold(threshold=1e-5)
# selector = SelectKBest(score_func = mutual_info_classif, k=10)

classifier = Pipeline([
    ('vec', vec),
    # ('selector', selector),
    ('cls', cls)
])

print(len(texts))
print(len(authors))

predictions = cross_val_predict(
    estimator=classifier,
    X=texts,
    y=authors,
    cv=4,
    verbose=3,
    n_jobs=4
)

print(homogeneity_completeness_v_measure(authors, predictions))