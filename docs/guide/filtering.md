# Text Filtering

## TF-IDF Based Filtering
The system employs TF-IDF (Term Frequency-Inverse Document Frequency) filtering to:
* Remove the top 200 most common terms (potentially too generic)
* Remove the bottom 50 least common terms (potentially noise)

## Dictionary Filtering
Additional filtering is applied using Gensim's Dictionary:
* Removes terms that appear in less than 2 documents
* Removes terms that appear in more than 90% of documents