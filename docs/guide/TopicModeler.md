# Class: TopicModeler

# `__init__(self, input_path)`
Creates a new TopicModeler instance with the specified input path for documents.

## Key Methods

### Document Loading and Preprocessing
* `load_documents()`: Loads text documents from the input path
* `preprocess_documents()`: Applies the full preprocessing pipeline
* `add_bigrams()`: Detects and adds significant bigrams

### Model Training
* `train_lda_model(num_topics=6, chunksize=2000, passes=20, iterations=400)`: Trains the LDA model
* `print_model_info()`: Displays model statistics and performance metrics
* `visualize_topics()`: Generates interactive visualizations

### Model Persistence
* `save_model(file_path)`: Saves the trained model
* `load_model(file_path)`: Loads a previously trained model