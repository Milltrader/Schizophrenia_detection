import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

###############################################################################
# 1) Simple ROI feature extractor
###############################################################################
class ROIsFeatureExtractor(BaseEstimator, TransformerMixin):
    """Select only the 284 ROIs features."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, :284]

###############################################################################
# 2) Common data preprocessing pipeline
###############################################################################
def build_preprocessing_pipeline():
    """
    Returns a Pipeline that:
      - extracts the ROI features,
      - scales them,
      - drops near-constant features,
      - optionally does L1-based selection (LogisticRegression).
    """
    return Pipeline([
        ('roi_extract', ROIsFeatureExtractor()),
        ('scaler', StandardScaler()),
        ('variance_thresh', VarianceThreshold(threshold=1e-6)),
        ('l1_select', SelectFromModel(
            LogisticRegression(
                penalty='l1',
                solver='liblinear',
                C=0.2,
                random_state=42
            ),
            prefit=False  # Model is fit during the pipeline's fit(...)
        ))
    ])

###############################################################################
# 3) Define sub-pipelines for each estimator
###############################################################################
def build_lr_pipeline():
    """Logistic Regression pipeline (reusing the same transformations)."""
    prep = build_preprocessing_pipeline()
    lr = LogisticRegression(
        random_state=42,
        class_weight='balanced',  # optional
        solver='lbfgs',
        penalty='l2',
        max_iter=300
    )
    return make_pipeline(prep, lr)

def build_svm_pipeline():
    """SVM pipeline."""
    prep = build_preprocessing_pipeline()
    svm = SVC(
        probability=True,  # Needed for soft voting + .predict_proba
        kernel='linear',
        C=0.01,
        class_weight='balanced',
        random_state=42
    )
    return make_pipeline(prep, svm)

def build_mlp_pipeline():
    """MLP pipeline."""
    prep = build_preprocessing_pipeline()
    mlp = MLPClassifier(
        random_state=42,
        max_iter=200,
        hidden_layer_sizes=(150, 50, 50),
        early_stopping=True,
        learning_rate_init=0.01,
        learning_rate='adaptive',
        batch_size=32,
        alpha=0.06,
        activation='logistic'
    )
    return make_pipeline(prep, mlp)

###############################################################################
# 4) Custom wrapper to:
#    - convert string labels into numeric in .fit(...)
#    - return 2D probabilities in .predict(...) for RAMP’s scoring
###############################################################################
class NumericLabelProbWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper that:
       1) Forces numeric labels in fit() => 1 for 'schizophrenia', 0 otherwise
       2) Uses predict_proba(...) as predict(...) => shape (n_samples, 2).
    """
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        # Convert string labels to numeric
        y_numeric = np.where(y == 'schizophrenia', 1, 0)
        self.estimator.fit(X, y_numeric)
        return self

    def predict(self, X):
        # RAMP calls .predict(...), but expects shape (n_samples, n_classes)
        return self.estimator.predict_proba(X)

    def predict_proba(self, X):
        # Optionally define this too
        return self.estimator.predict_proba(X)

###############################################################################
# 5) Build and return the final ensemble
###############################################################################
def get_estimator():
    """
    Build the ensemble for RAMP submission.
    Each sub-pipeline includes the same transformations.
    Then we wrap everything so labels become numeric and
    RAMP sees (n_samples, 2) shaped predictions.
    """
    # Sub-pipelines
    lr_pipe = build_lr_pipeline()
    svm_pipe = build_svm_pipeline()
    mlp_pipe = build_mlp_pipeline()

    # Voting ensemble (soft-voting, arbitrary weights as example)
    ensemble = VotingClassifier(
        estimators=[
            ('lr', lr_pipe),
            ('svm', svm_pipe),
            ('mlp', mlp_pipe),
        ],
        voting='soft',  # Probability-based
        weights=[1, 1, 2],
        n_jobs=-1
    )

    # Wrap so:
    #   - numeric labels are used during fit(...)
    #   - .predict(...) returns (n_samples, 2) probabilities
    return NumericLabelProbWrapper(ensemble)
