# Installation

## Requirements

```bash
pip install nltk gensim pyLDAvis matplotlib scikit-learn pandas numpy
```

Required Python packages:
- nltk: For text preprocessing
- gensim: For topic modeling
- pyLDAvis: For interactive topic visualization
- matplotlib: For plotting coherence scores
- scikit-learn: For TF-IDF vectorization
- pandas: For data manipulation
- numpy: For numerical operations

## Initial Setup

```bash
python -m spacy download en_core_web_md
```
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```