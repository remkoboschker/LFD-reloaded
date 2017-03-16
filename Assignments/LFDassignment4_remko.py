import numpy, pickle, os
from sys import argv
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# Read in the NE data, with either 2 or 6 classes
def read_corpus(corpus_file, binary_classes):
    print('Reading in data from {0}...'.format(corpus_file))
    words = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            words.append(parts[0])

            if binary_classes:
                if parts[1] == 'GPE' or parts[1] == 'LOC':
                    labels.append('LOCATION')
                else:
                    labels.append('NON-LOCATION')
            else:
                labels.append(parts[1])

    print('Done!')
    return words, labels

# Read in word embeddings
def read_embeddings(embeddings_file):
    print('Reading in embeddings from {0}...'.format(embeddings_file))
    embeddings = pickle.load(open('embeddings.pickle', 'rb'))
    print('Done!')
    return embeddings

# Turn words into embeddings, i.e. replace words by their corresponding embeddings
def vectorizer(words, embeddings):
    vectorized_words = []
    for word in words:
        try:
            vectorized_words.append(embeddings[word.lower()])
        except KeyError:
            vectorized_words.append(embeddings['UNK'])
    return numpy.array(vectorized_words)

if __name__ == '__main__':
    # Read in the data
    input_filename = argv[1]
    X, Y = read_corpus(input_filename, False)
    # Read in the embeddings
    embeddings_filename = argv[2]
    embeddings = read_embeddings(embeddings_filename)
    # Transform words to embeddings
    X = vectorizer(X, embeddings)
    # Split in training and test data
    split_point = int(0.75*len(X))
    Xtrain = X[:split_point]
    Ytrain = Y[:split_point]
    Xtest = X[split_point:]
    Ytest = Y[split_point:]

    # Initialize the perceptron
    perceptron = SGDClassifier(loss='perceptron', eta0 = 1.0, verbose = 1, random_state = 92, n_iter = 5, penalty='none', learning_rate='constant')
    # Train the perceptron
    perceptron.fit(Xtrain, Ytrain)
    # Make predictions for the test data
    # Yguess = perceptron.predict(Xtest)
    # baseline
    labelDist = ['LOC'] + (len(Xtest) // 6 + 1) * ['GPE', 'CARDINAL', 'PERSON', 'DATE', 'ORG', 'GPE' ]
    Yguess = labelDist[:len(Xtest)]
    print('Classification accuracy: {0}'.format(accuracy_score(Ytest, Yguess)))
    print(classification_report(Ytest, Yguess))
