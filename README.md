
# Word Sense Disambiguation (WSD) Project

## Overview
This project implements various algorithms for Word Sense Disambiguation (WSD), focusing on classifying the meaning of words based on their context. The main classifiers used are:

1. **RandomSense**: A baseline model that randomly predicts senses.
2. **MostFrequentSense**: Predicts the most frequent sense for each word based on training data.
3. **Simplified Lesk Algorithm**: Utilizes definitions and context words to find the best sense by calculating overlap between sense definitions and context.

## Structure
- **wsd.py**: The main file containing the WSD classifiers and the evaluation logic.
- **utils.py**: Contains utility functions for data manipulation, IDF calculation, and other supporting tasks.
- **twa.py**: Handles the parsing of the TWA dataset and creates WSD instances.
  
## Logic
The classifiers evaluate their performance based on a dataset of word senses and contexts, measuring accuracy against a test set. The project includes functions for:

- Loading and normalizing data.
- Splitting data into training and testing sets.
- Evaluating classifiers using accuracy metrics.

### Evaluation Metrics
- The accuracy of the models is computed using the proportion of correctly predicted senses over the total instances. 

## Results
Upon running the classifiers on the TWA dataset, the following average accuracies were achieved through cross-validation:

- Random Baseline: [insert accuracy]
- Most Frequent Sense: [insert accuracy]
- Simplified Lesk: [insert accuracy]
- Simplified Lesk with Window Size: [insert accuracy]
- Simplified Lesk with IDF: [insert accuracy]

## Installation
Ensure you have the required libraries:
```bash
pip install nltk scikit-learn
```

## Usage
Run the main script with the path to the TWA dataset:
```bash
python wsd.py path/to/twa_dataset.xml
```

## License
This project is licensed under the MIT License.
