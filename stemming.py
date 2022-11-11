#stemming
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
porter_stemmer  = PorterStemmer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
 print("Stemming for {} is {}".format(w,porter_stemmer.stem(w)))





#lemmatization
import nltk
from nltk.stem import 	WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
 print("Lemma for {} is {}".format(w, wordnet_lemmatizer.lemmatize(w)))
