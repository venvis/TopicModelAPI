# Single Document Classification

```python
def preprocess_text(text):
    return text.lower().split()

# Initialize and load models
evaluator = ModelEvaluator("./models")
evaluator.load_all_models()

# Classify text
text = "Your document text here..."
result = evaluator.get_ensemble_prediction(
    text,
    preprocess_text,
    threshold=0.3
)

if result:
    print(f"Best prediction from model: {result['model']}")
    print(f"Topic ID: {result['prediction']['topic_id']}")
    print(f"Confidence: {result['prediction']['probability']:.3f}")
    print("Top terms:", ', '.join(result['prediction']['top_terms'][:5]))
```

# Batch Classification

```python
texts = [
    "Document 1 text...",
    "Document 2 text...",
    "Document 3 text..."
]

for idx, text in enumerate(texts):
    print(f"\nDocument {idx + 1}:")
    result = evaluator.get_ensemble_prediction(text, preprocess_text)
    if result:
        print(f"Classified as Topic {result['prediction']['topic_id']}")
        print(f"Confidence: {result['prediction']['probability']:.3f}")
        print("Key terms:", ', '.join(result['prediction']['top_terms'][:3]))

```

# Working with Model Results

The classification results contain detailed information about topic distributions:

```python
results = evaluator.classify_text(textpreprocess_text)

for model_name, data in results.items():
    print(f"\nModel: {model_name}")
    print(f"Topic ID: {data['topic_id']}")
    print(f"Probability: {data['probability']:.3f}")
    print("Top terms:", data['top_terms'])
    print("Term weights:", data['term_weights'])
```
