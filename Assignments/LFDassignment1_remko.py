import argparse

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser(
    prog='naive bayes classifier',
    description='classifies reviews by sentiment of topic'
)
parser.add_argument(
    '--use_sentiment',
    default=False,
    type=bool,
    dest='use_sentiment',
    help='boolean to select the use of sentiment labels',
    metavar='use sentiment',
    choices=[True, False]
)

args = parser.parse_args()


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

            if args.use_sentiment:
                # 2-class problem: positive vs negative
                labels.append( tokens[1] )
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append( tokens[0] )

    return documents, labels

# a dummy function that just returns its input
def identity(x):
    return x

# Calls read corpus with trainset.txt as a filename and using the sentiment labels
# assigning three quarters of the documents and labels as a training set and one quarter as
# a test set
documents, labels = read_corpus('trainset.txt', use_sentiment=True)
split_point = int(0.75*len(documents))
documentsTrain = documents[:split_point]
labelsTrain = labels[:split_point]
documentsTest = documents[split_point:]
labelsTest = labels[split_point:]

# let's use the TF-IDF vectorizer
tfidf = True

# we use a dummy function as tokenizer and preprocessor,
# since the texts are already preprocessed and tokenized.
if tfidf:
    vec = TfidfVectorizer(preprocessor = identity,
                          tokenizer = identity)
else:
    vec = CountVectorizer(preprocessor = identity,
                          tokenizer = identity)

# Combines the vectorizer with a Naive Bayes classifier
classifier = Pipeline( [('vec', vec),
                        ('cls', MultinomialNB())] )


# Fits a naive bayes classifier according to the training vectors that come from
# the vectorization of the documents to the target values in the label training set.
classifier.fit(documentsTrain, labelsTrain)

# Performs classification on the vecorization of the documents in the test set.
labelsGuess = classifier.predict(documentsTest)

# Prints a report containing the f-score, precision, recall and number of supporting instances
print(classification_report(labelsTest, labelsGuess))
# Prints the confusion matrix
print(confusion_matrix(labelsTest, labelsGuess))

print(classifier.predict_proba(documentsTest[:10]))

