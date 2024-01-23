"""
Harmonic Linear Discriminant Analysis and Quadratic Discriminant Analysis
"""

# Authors: Clemens Brunner
#          Martin Billinger
#          Matthieu Perrot
#          Mathieu Blondel

# License: BSD 3-Clause

import warnings
import numpy as np
from scipy import linalg
from scipy.special import expit

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.covariance import ledoit_wolf, empirical_covariance, shrunk_covariance
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.extmath import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import *


__all__ = ["LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis", "HarmonicLinearDiscriminantAnalysis"]



def _cov(X, shrinkage=None, covariance_estimator=None):
    """Estimate covariance matrix (using optional covariance_estimator).
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    shrinkage : {'empirical', 'auto'} or float, default=None
        Shrinkage parameter, possible values:
          - None or 'empirical': no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

        Shrinkage parameter is ignored if  `covariance_estimator`
        is not None.

    covariance_estimator : estimator, default=None
        If not None, `covariance_estimator` is used to estimate
        the covariance matrices instead of relying on the empirical
        covariance estimator (with potential shrinkage).
        The object should have a fit method and a ``covariance_`` attribute
        like the estimators in :mod:`sklearn.covariance``.
        if None the shrinkage parameter drives the estimate.

        .. versionadded:: 0.24

    Returns
    -------
    s : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.
    """
    if covariance_estimator is None:
        shrinkage = "empirical" if shrinkage is None else shrinkage
        if isinstance(shrinkage, str):
            if shrinkage == "auto":
                sc = StandardScaler()  # standardize features
                X = sc.fit_transform(X)
                s = ledoit_wolf(X)[0]
                # rescale
                s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
            elif shrinkage == "empirical":
                s = empirical_covariance(X)
            else:
                raise ValueError("unknown shrinkage parameter")
        elif isinstance(shrinkage, float) or isinstance(shrinkage, int):
            if shrinkage < 0 or shrinkage > 1:
                raise ValueError("shrinkage parameter must be between 0 and 1")
            s = shrunk_covariance(empirical_covariance(X), shrinkage)
        else:
            raise TypeError("shrinkage must be a float or a string")
    else:
        if shrinkage is not None and shrinkage != 0:
            raise ValueError(
                "covariance_estimator and shrinkage parameters "
                "are not None. Only one of the two can be set."
            )
        covariance_estimator.fit(X)
        if not hasattr(covariance_estimator, "covariance_"):
            raise ValueError(
                "%s does not have a covariance_ attribute"
                % covariance_estimator.__class__.__name__
            )
        s = covariance_estimator.covariance_
    return s


def _class_means(X, y):
    """Compute class means.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    Returns
    -------
    means : array-like of shape (n_classes, n_features)
        Class means.
    """
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]
    return means


def _class_cov(X, y, priors, shrinkage=None, covariance_estimator=None):
    """Compute weighted within-class covariance matrix.

    The per-class covariance are weighted by the class priors.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    priors : array-like of shape (n_classes,)
        Class priors.

    shrinkage : 'auto' or float, default=None
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

        Shrinkage parameter is ignored if `covariance_estimator` is not None.

    covariance_estimator : estimator, default=None
        If not None, `covariance_estimator` is used to estimate
        the covariance matrices instead of relying the empirical
        covariance estimator (with potential shrinkage).
        The object should have a fit method and a ``covariance_`` attribute
        like the estimators in sklearn.covariance.
        If None, the shrinkage parameter drives the estimate.

        .. versionadded:: 0.24

    Returns
    -------
    cov : array-like of shape (n_features, n_features)
        Weighted within-class covariance matrix
    """
    classes = np.unique(y)
    cov = np.zeros(shape=(X.shape[1], X.shape[1]))
    for idx, group in enumerate(classes):
        Xg = X[y == group, :]
        cov += priors[idx] * np.atleast_2d(_cov(Xg, shrinkage, covariance_estimator))
    return cov




def _harmonic_class_inv_cov(X, y, priors, shrinkage=None, covariance_estimator=None):
    """Compute harmonic weighted within-class covariance matrix.

    The per-class covariance are weighted by the class priors.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    priors : array-like of shape (n_classes,)
        Class priors.

    shrinkage : 'auto' or float, default=None
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

        Shrinkage parameter is ignored if `covariance_estimator` is not None.

    covariance_estimator : estimator, default=None
        If not None, `covariance_estimator` is used to estimate
        the covariance matrices instead of relying the empirical
        covariance estimator (with potential shrinkage).
        The object should have a fit method and a ``covariance_`` attribute
        like the estimators in sklearn.covariance.
        If None, the shrinkage parameter drives the estimate.

        .. versionadded:: 0.24

    Returns
    -------
    cov : array-like of shape (n_features, n_features)
        Weighted within-class covariance matrix
    """
    classes = np.unique(y)
    total_priors = 0 
    inv_cov = np.zeros(shape=(X.shape[1], X.shape[1]))
    for idx, group in enumerate(classes):
        Xg = X[y == group, :]
        total_priors += priors[idx]
        inv_cov += priors[idx] * linalg.inv(np.atleast_2d(_cov(Xg, shrinkage, covariance_estimator)))
    return inv_cov / total_priors 

def _harmonic_class_cov(X, y, priors, shrinkage=None, covariance_estimator=None):
    """Compute harmonic weighted within-class covariance matrix.

    The per-class covariance are weighted by the class priors.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    priors : array-like of shape (n_classes,)
        Class priors.

    shrinkage : 'auto' or float, default=None
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

        Shrinkage parameter is ignored if `covariance_estimator` is not None.

    covariance_estimator : estimator, default=None
        If not None, `covariance_estimator` is used to estimate
        the covariance matrices instead of relying the empirical
        covariance estimator (with potential shrinkage).
        The object should have a fit method and a ``covariance_`` attribute
        like the estimators in sklearn.covariance.
        If None, the shrinkage parameter drives the estimate.

        .. versionadded:: 0.24

    Returns
    -------
    cov : array-like of shape (n_features, n_features)
        Weighted within-class covariance matrix
    """
    classes = np.unique(y)
    total_priors = 0 
    inv_cov = np.zeros(shape=(X.shape[1], X.shape[1]))
    for idx, group in enumerate(classes):
        Xg = X[y == group, :]
        total_priors += priors[idx]
        inv_cov += priors[idx] * linalg.inv(np.atleast_2d(_cov(Xg, shrinkage, covariance_estimator)))
    return total_priors * linalg.inv(inv_cov) 


class HarmonicLinearDiscriminantAnalysis(
    LinearDiscriminantAnalysis
    ):
    """Harmonic Linear Discriminant Analysis.

    A classifier with a linear decision boundary, generated by fitting class
    conditional densities to the data and using Bayes' rule.

    The model fits a Gaussian density to each class, assuming that all classes
    share the same covariance matrix.

    The fitted model can also be used to reduce the dimensionality of the input
    by projecting it to the most discriminative directions, using the
    `transform` method.

    .. versionadded:: 0.17
       *LinearDiscriminantAnalysis*.

    Read more in the :ref:`User Guide <lda_qda>`.

    Parameters
    ----------
    solver : {'eigen'}, 
        Solver to use, possible values:
          - 'eigen': Eigenvalue decomposition.
            Can be combined with shrinkage or custom covariance estimator.

    shrinkage : 'auto' or float, default=None
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

        This should be left to None if `covariance_estimator` is used.
        Note that shrinkage works only with 'lsqr' and 'eigen' solvers.

    priors : array-like of shape (n_classes,), default=None
        The class prior probabilities. By default, the class proportions are
        inferred from the training data.

    n_components : int, default=None
        Number of components (<= min(n_classes - 1, n_features)) for
        dimensionality reduction. If None, will be set to
        min(n_classes - 1, n_features). This parameter only affects the
        `transform` method.

    store_covariance : bool, default=False
        If True, explicitly compute the weighted within-class covariance
        matrix when solver is 'svd'. The matrix is always computed
        and stored for the other solvers.

        .. versionadded:: 0.17

    tol : float, default=1.0e-4
        Absolute threshold for a singular value of X to be considered
        significant, used to estimate the rank of X. Dimensions whose
        singular values are non-significant are discarded. Only used if
        solver is 'svd'.

        .. versionadded:: 0.17

    covariance_estimator : covariance estimator, default=None
        If not None, `covariance_estimator` is used to estimate
        the covariance matrices instead of relying on the empirical
        covariance estimator (with potential shrinkage).
        The object should have a fit method and a ``covariance_`` attribute
        like the estimators in :mod:`sklearn.covariance`.
        if None the shrinkage parameter drives the estimate.

        This should be left to None if `shrinkage` is used.
        Note that `covariance_estimator` works only with 'lsqr' and 'eigen'
        solvers.

        .. versionadded:: 0.24

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_classes, n_features)
        Weight vector(s).

    intercept_ : ndarray of shape (n_classes,)
        Intercept term.

    covariance_ : array-like of shape (n_features, n_features)
        Weighted within-class covariance matrix. It corresponds to
        `sum_k prior_k * C_k` where `C_k` is the covariance matrix of the
        samples in class `k`. The `C_k` are estimated using the (potentially
        shrunk) biased estimator of covariance. If solver is 'svd', only
        exists when `store_covariance` is True.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0. Only available when eigen
        or svd solver is used.

    means_ : array-like of shape (n_classes, n_features)
        Class-wise means.

    priors_ : array-like of shape (n_classes,)
        Class priors (sum to 1).

    scalings_ : array-like of shape (rank, n_classes - 1)
        Scaling of the features in the space spanned by the class centroids.
        Only available for 'svd' and 'eigen' solvers.

    xbar_ : array-like of shape (n_features,)
        Overall mean. Only present if solver is 'svd'.

    classes_ : array-like of shape (n_classes,)
        Unique class labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    QuadraticDiscriminantAnalysis : Quadratic Discriminant Analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> clf = LinearDiscriminantAnalysis()
    >>> clf.fit(X, y)
    LinearDiscriminantAnalysis()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """

    def __init__(
        self,
        shrinkage=None,
        priors=None,
        n_components=None,
        store_covariance=False,
        tol=1e-4,
        covariance_estimator=None,
    ):
        self.solver = 'eigen'
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance  # used only in svd solver
        self.tol = tol  # used only in svd solver
        self.covariance_estimator = covariance_estimator


    def _solve_eigen(self, X, y, shrinkage, covariance_estimator):
        """Eigenvalue solver.

        The eigenvalue solver computes the optimal solution of the Rayleigh
        coefficient (basically the ratio of between class scatter to within
        class scatter). This solver supports both classification and
        dimensionality reduction (with any covariance estimator).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        shrinkage : 'auto', float or None
            Shrinkage parameter, possible values:
              - None: no shrinkage.
              - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
              - float between 0 and 1: fixed shrinkage constant.

            Shrinkage parameter is ignored if  `covariance_estimator` i
            not None

        covariance_estimator : estimator, default=None
            If not None, `covariance_estimator` is used to estimate
            the covariance matrices instead of relying the empirical
            covariance estimator (with potential shrinkage).
            The object should have a fit method and a ``covariance_`` attribute
            like the estimators in sklearn.covariance.
            if None the shrinkage parameter drives the estimate.

            .. versionadded:: 0.24

        Notes
        -----
        This solver is based on [1]_, section 3.8.3, pp. 121-124.

        References
        ----------
        .. [1] R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification
           (Second Edition). John Wiley & Sons, Inc., New York, 2001. ISBN
           0-471-05669-3.
        """
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(
            X, y, self.priors_, shrinkage, covariance_estimator
        )
        
        inv_Sw = _harmonic_class_inv_cov(
            X, y, self.priors_, shrinkage, covariance_estimator
        )  # within scatter, corrected for harmonic mean
        #St = _cov(X, shrinkage, covariance_estimator)  # total scatter
        #Sb = St - linalg.inv(inv_Sw)
        Sb = self.means_ - X.mean(axis=0).reshape(1,-1)
        Sb = np.sum(np.outer(Sb[i],Sb[i]) for i in range(len(self.classes_))) # between scatter

        evals, evecs = linalg.eig(inv_Sw @ Sb)
        evals = np.real(evals)
        evecs = np.real(evecs)
        #evals, evecs = linalg.eigh(Sb, Sw )
        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1][
            : self._max_components
        ]
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    
    def fit(self, X, y):
        """Fit the Linear Discriminant Analysis model.

           .. versionchanged:: 0.19
              *store_covariance* has been moved to main constructor.

           .. versionchanged:: 0.19
              *tol* has been moved to main constructor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        return super().fit(X,y)
        
        
    def transform(self, X):
        """Project data to maximize class separation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        return super().transform(X)
        
    def predict_proba(self, X):
        """Estimate probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Estimated probabilities.
        """
        return super().predict_proba(X)
        
    def predict_log_proba(self, X):
        """Estimate log probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Estimated log probabilities.
        """
        return super().predict_log_proba(X)
        
    def decision_function(self, X):
        """Apply decision function to an array of samples.

        The decision function is equal (up to a constant factor) to the
        log-posterior of the model, i.e. `log p(y = k | x)`. In a binary
        classification setting this instead corresponds to the difference
        `log p(y = 1 | x) - log p(y = 0 | x)`. See :ref:`lda_qda_math`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Array of samples (test vectors).

        Returns
        -------
        C : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values related to each class, per sample.
            In the two-class case, the shape is (n_samples,), giving the
            log likelihood ratio of the positive class.
        """
        return super().decision_function(X)