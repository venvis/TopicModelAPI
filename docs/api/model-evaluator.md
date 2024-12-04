# Model Evaluator API

The main class for handling model loading and text classification.


```python
evaluator = ModelEvaluator(base_path="./new_filtering/lda_models")
```

Parameters
- `base_path` (str): Directory containing the trained LDA models

Methods 
- `load_all_models()`: Loads all the LDA models from the specified directory

```python
evaluator.load_all_models()
```

Each model directory should contain:
- `trained_model`: The gensim LDA model file
- `dictionary`: The gensim Dictionary file
- `lda_visualization.html`: Optional visualization data


- `classify_text(text, preprocess_func)`: Classifies text using all loaded models and returns aggregated results.

```python
result = evaluator.classify_text(text, preprocess_func)
```
Parameters:
- `text` (str): Text to classify
- `preprocess_func` (callable): Function to preprocess text
Returns:
- Dictionary containing classification results from each model

- `get_ensemble_prediction(text, preprocess_func, threshold=0.5)`: Gets the most confident prediction above the threshold from all models

```python
result = evaluator.get_ensemble_prediction(text, preprocess_func, threshold=0.5)
```

Parameters:
- `text` (str): Text to classify
- `preprocess_func` (callable): Function to preprocess text
- `threshold` (float): Minimum confidence threshold (default: 0.5)

Returns:
- Dictionary containing the best prediction and model information 