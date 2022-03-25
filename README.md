# RVFLN

<p>
A Python implementation of Random Vector Functional Link Networks and Broad Learning Systems using the Sklearn API
</p>

## Installation

    pip install rvfln

<br>

## Usage

Since the API is based on that of sklearn the usage is very similar.
The package contains:

- An rvfln module containing a regressor and a classifier estimator
- A bls module containing a regressor and a classifier estimator

<br>

### Example:

<br>

    from rvfln.bls import BLSClassifier

    clf = BLSClassifier(
        n_z = 10,
        n_z_features = 400,
        n_h = 2000,
        alpha = 1
    ).fit(x_train, y_train)

    print(clf.score(x_test, y_test))

<br>

YOH-HAN PAO, STEPHEN M. PHILLIPS & DEJAN J. SOBAJIC (1992) Neural-net computing and the intelligent control of systems, International Journal of Control, 56:2, 263-289, DOI: 10.1080/00207179208934315
