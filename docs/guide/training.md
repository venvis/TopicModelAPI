# Training Parameters

The system uses the following batch configurations:

```python
param_combinations = [
   (4, 2000, 10, 200),   # (num_topics, chunksize, passes, iterations)
   (8, 2000, 10, 400),
   (12, 2000, 20, 200),
   (16, 2000, 20, 400),
   (20, 2000, 20, 400),
   (24, 2000, 20, 400)
]
```
# Why Gensim?

Gensim was chosen for this implementation because it offers:

* Memory-efficient processing of large text corpora
* Built-in support for streaming large datasets
* Robust implementation of LDA with:
  * Automatic hyperparameter optimization
  * Multi-core processing support
  * Comprehensive model evaluation metrics
* Integration with pyLDAvis for interactive visualization
* Built-in support for model persistence and loading

