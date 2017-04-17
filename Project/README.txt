The script uses pre-trained word embeddings and looks for them in the training directories for the different languages.

The script expects the following files in the training directory:

/english/glove.twitter.27B.200d.txt
/dutch/rob-nl.txt
none for Italian
/spanish/wiki.es.vec

glove.twitter.27B.200d.txt is part of this zip: http://nlp.stanford.edu/data/glove.twitter.27B.zip

rob-nl.txt can be found on my google drive
There is also a copy on /net/shared/rob/nlTweets/tw.vecs , but it needs to be converted from binary to text for instance using https://github.com/marekrei/convertvec

wiki.es.vec can be found here https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.es.vec



