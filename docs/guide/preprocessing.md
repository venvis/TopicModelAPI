# Text Preprocessing

The preprocessing pipeline includes several steps to clean and prepare the text data:

## 1. Tokenization
* Using NLTK's RegexpTokenizer to split text into tokens

## 2. Normalization
* Converting to lowercase 
* Removing numbers
* Removing short tokens (length < 2)

## 3. Stop Word Removal
* Using NLTK's English stop words
* Removing custom high-frequency words

## 4. Lemmatization
* Using WordNet lemmatizer to reduce words to their base form

## 5. Bigram Detection
* Adding meaningful word pairs that frequently occur together

## 6. High-Frequency Word Filtering
The system automatically identifies and removes high-frequency words that appear across multiple topics (frequency > 10) to improve topic distinctiveness.