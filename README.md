# Fake vs Real Hotel Review Detector

An NLP classification project that predicts whether a hotel review is truthful or deceptive. It uses TF-IDF vectorization and Logistic Regression inside a scikit-learn Pipeline.

## Dataset

Uses the **Deceptive Opinion Spam Corpus** (1,600 Chicago hotel reviews, 800 truthful and 800 deceptive), loaded directly from a GitHub-hosted CSV file. An internet connection is needed on each run.

## How It Works

1. Remove rows with missing text or labels.
2. Clean text: lowercase, remove punctuation, normalize whitespace.
3. Convert text to TF-IDF features (top 3,000 terms).
4. Train Logistic Regression in a Pipeline, so the vectorizer learns only from training data.
5. Evaluate on a stratified 80/20 hold-out split and with 5-fold cross-validation.
6. Print the 15 words most associated with each class.
7. Classify new reviews typed in by the user.

## Performance

- **Hold-out accuracy (320 test reviews):** 84.06%
- **5-fold cross-validation accuracy:** 87.13% (+/- 1.36 percentage points)
- Precision, recall and F1 are about 0.84 for both classes.

The cross-validation mean is the more reliable estimate. These scores measure how well the model separates this dataset's crowd-written fake reviews from its real reviews. They are not the accuracy on real-world fake reviews.

## Setup & Run

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

2. Activate it:

   ```bash
   # Windows
   .venv\Scripts\activate

   # macOS / Linux
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the script:
   ```bash
   python classify_review.py
   ```

The script prints the evaluation results, runs two sample predictions, and then opens an interactive prompt. Paste a review on one line and press Enter. Type `exit` to quit.

## Interactive Output

For each review the tool prints a result and a confidence value. If the confidence is below 60%, the result is shown as **Uncertain** along with the most likely class. The 60% threshold was chosen by hand and has not been validated.

## Limitations

- Small dataset covering hotels in a single city, so it may not generalize to other cities or domains.
- Deceptive reviews were written by paid workers, so real-world fake reviews may look different.
- Some learned words, such as "chicago" and "hilton", are artifacts of how the dataset was collected.
- The model reacts to individual words and ignores context. It is less reliable on short or generic text and on reviews unlike the training data.
- Hyperparameters were not tuned.

## Project Files

- `classify_review.py`: training, evaluation and interactive prediction
- `requirements.txt`: dependencies (pandas, scikit-learn, numpy)
- `.gitignore`: files excluded from version control

## Reference

Ott, M., Choi, Y., Cardie, C., and Hancock, J. T. (2011). Finding Deceptive Opinion Spam by Any Stretch of the Imagination. Proceedings of ACL.

<!-- # Fake vs Real Review Detector

An NLP classification project that predicts whether a hotel review is truthful or deceptive. It uses TF-IDF vectorization and Logistic Regression.

## Dataset

Uses the **Deceptive Opinion Spam Corpus** (1,600 hotel reviews) fetched directly from GitHub.

## Performance

- **Accuracy:** 84.06%

## Setup & Run

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ``` -->
