# RVFLN

<p>
A Python implementation of Random Vector Functional Link Networks using the Sklearn API
</p>

## Installation

    pip install rvfln

<br>

## Usage

Since the API is based on that of sklearn the usage is very similar.
The package contains:

- a base model, where both X and Y are matrices,
- a regressor, where X is a matrix and y is a vector
- a classifier, where X is a matrix and y is an Iterable of labels
  ``
  from rvfln import RVFLNClassifier

  model = RVFLNClassifier(n_enhancement = 2000)
  model.fit(X_train, y_train)
  model.score(X_test, y_test)
  ``
  <br>

YOH-HAN PAO, STEPHEN M. PHILLIPS & DEJAN J. SOBAJIC (1992) Neural-net computing and the intelligent control of systems, International Journal of Control, 56:2, 263-289, DOI: 10.1080/00207179208934315
