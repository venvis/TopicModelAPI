# Topic Modeler API Overview

The Topic Modeler provides functionality for loading and using multiple LDA (Latent Dirichlet Allocation) topic models to classify text documents. It supports both single document and batch classification with confidence scoring.

## Key Features

- Load multiple pre-trained LDA models
- Single document classification
- Ensemble predictions with confidence thresholds
- Support for model visualizations
- Batch text classification

## Installation

The Topic Modeler requires the following dependencies: 

```python
import gensim
import os
import json
from bs4 import BeautifulSoup
import numpy as np

```

## Initialize ModelEvaluator

```python
evaluator = ModelEvaluator("./path/to/models")
evaluator.load_all_models()
```

