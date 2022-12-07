import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_consistent_length

def check_is_binary(array):
    """Checker if array consists of int or float binary values 0 (0.) and 1 (1.)

    Args:
        array (1d array-like): Array to check.
    """

    if not np.all(np.unique(array) == np.array([0, 1])):
        raise ValueError(f"Input array is not binary. "
                         f"Array should contain only int or float binary values 0 (or 0.) and 1 (or 1.). "
                         f"Got values {np.unique(array)}.")

class TwoModelsExtra(BaseEstimator):
    """
    """

    def __init__(self, estimator_trmnt: object, estimator_ctrl: object, pipe_trmnt: object, pipe_ctrl:object, 
        features_trmnt: list=None, features_ctrl:list=None, method='vanilla'):

        self.estimator_trmnt = estimator_trmnt
        self.estimator_ctrl = estimator_ctrl

        self.features_trmnt = features_trmnt
        self.features_ctrl = features_ctrl

        self.pipe_trmnt = pipe_trmnt
        self.pipe_ctrl = pipe_ctrl

        self.method = method
        self.trmnt_preds_ = None
        self.ctrl_preds_ = None
        self._type_of_target = None

        all_methods = ['vanilla', 'ddr_control', 'ddr_treatment']
        if method not in all_methods:
            raise ValueError("Two models approach supports only methods in %s, got"
                             " %s." % (all_methods, method))

        if estimator_trmnt is estimator_ctrl:
            raise ValueError('Control and Treatment estimators should be different objects.')

    def fit(self, X, y, treatment, estimator_trmnt_fit_params=None, estimator_ctrl_fit_params=None):
        """
        """

        check_consistent_length(X, y, treatment)
        check_is_binary(treatment)
        self._type_of_target = type_of_target(y)

        # Получаем control и treatment выборки
        X_ctrl, y_ctrl = X[treatment == 0], y[treatment == 0]
        X_trmnt, y_trmnt = X[treatment == 1], y[treatment == 1]

        # обучаем control конвейер
        self.pipe_ctrl.fit(X_ctrl[self.features_ctrl], y_ctrl)
        X_ctrl = self.pipe_ctrl.transform(X_ctrl[self.features_ctrl]).reset_index(drop=True)
        y_ctrl = y_ctrl.reset_index(drop=True)

        # обучаем treatment конвейер
        self.pipe_trmnt.fit(X_trmnt[self.features_trmnt], y_trmnt)
        X_trmnt = self.pipe_trmnt.transform(X_trmnt[self.features_trmnt]).reset_index(drop=True)
        y_trmnt = y_trmnt.reset_index(drop=True)

        if estimator_trmnt_fit_params is None:
            estimator_trmnt_fit_params = {}
        if estimator_ctrl_fit_params is None:
            estimator_ctrl_fit_params = {}

        if self.method == 'vanilla':
            
            # обучаем control конвейер
            self.pipe_ctrl.fit(X_ctrl[self.features_ctrl], y_ctrl)
            X_ctrl = self.pipe_ctrl.transform(X_ctrl[self.features_ctrl]).reset_index(drop=True)
            y_ctrl = y_ctrl.reset_index(drop=True)

            # обучаем treatment конвейер
            self.pipe_trmnt.fit(X_trmnt[self.features_trmnt], y_trmnt)
            X_trmnt = self.pipe_trmnt.transform(X_trmnt[self.features_trmnt]).reset_index(drop=True)
            y_trmnt = y_trmnt.reset_index(drop=True)

            self.estimator_ctrl.fit(
                X_ctrl, y_ctrl, **estimator_ctrl_fit_params
            )
            self.estimator_trmnt.fit(
                X_trmnt, y_trmnt, **estimator_trmnt_fit_params
            )

        if self.method == 'ddr_control':

            # обучаем control конвейер
            self.pipe_ctrl.fit(X_ctrl[self.features_ctrl], y_ctrl)
            X_ctrl = self.pipe_ctrl.transform(X_ctrl[self.features_ctrl]).reset_index(drop=True)
            y_ctrl = y_ctrl.reset_index(drop=True)

            # обучаем control модель
            self.estimator_ctrl.fit(
                X_ctrl, y_ctrl, **estimator_ctrl_fit_params
            )

            

            if self._type_of_target == 'binary':
                ddr_control = self.estimator_ctrl.predict_proba(X_trmnt)[:, 1]
            else:
                ddr_control = self.estimator_ctrl.predict(X_trmnt)

            if isinstance(X_trmnt, np.ndarray):
                X_trmnt_mod = np.column_stack((X_trmnt, ddr_control))
            elif isinstance(X_trmnt, pd.DataFrame):
                X_trmnt_mod = X_trmnt.assign(ddr_control=ddr_control)
            else:
                raise TypeError("Expected numpy.ndarray or pandas.DataFrame, got %s" % type(X_trmnt))

            self.estimator_trmnt.fit(
                X_trmnt_mod, y_trmnt, **estimator_trmnt_fit_params
            )

        if self.method == 'ddr_treatment':
            self.estimator_trmnt.fit(
                X_trmnt, y_trmnt, **estimator_trmnt_fit_params
            )
            if self._type_of_target == 'binary':
                ddr_treatment = self.estimator_trmnt.predict_proba(X_ctrl)[:, 1]
            else:
                ddr_treatment = self.estimator_trmnt.predict(X_ctrl)

            if isinstance(X_ctrl, np.ndarray):
                X_ctrl_mod = np.column_stack((X_ctrl, ddr_treatment))
            elif isinstance(X_trmnt, pd.DataFrame):
                X_ctrl_mod = X_ctrl.assign(ddr_treatment=ddr_treatment)
            else:
                raise TypeError("Expected numpy.ndarray or pandas.DataFrame, got %s" % type(X_ctrl))

            self.estimator_ctrl.fit(
                X_ctrl_mod, y_ctrl, **estimator_ctrl_fit_params
            )

        return self

    def predict(self, X):
        """Perform uplift on samples in X.

        Args:
            X (array-like, shape (n_samples, n_features)): Training vector, where n_samples is the number of samples
                and n_features is the number of features.

        Returns:
            array (shape (n_samples,)): uplift
        """
        X = X.copy().reset_index(drop=True)
        X_ctrl = self.pipe_ctrl.transform(X[self.features_ctrl]).reset_index(drop=True)
        X_trmnt = self.pipe_trmnt.transform(X[self.features_trmnt]).reset_index(drop=True)

        if self.method == 'ddr_control':

            self.ctrl_preds_ = self.estimator_ctrl.predict_proba(X_ctrl)[:, 1]
            
            X_mod = X_trmnt.assign(ddr_control=self.ctrl_preds_)
            self.trmnt_preds_ = self.estimator_trmnt.predict_proba(X_mod)[:, 1]
            
        elif self.method == 'ddr_treatment':
            
            self.trmnt_preds_ = self.estimator_trmnt.predict_proba(X_trmnt)[:, 1]

            X_mod = X_trmnt.assign(ddr_treatment=self.trmnt_preds_)
            self.ctrl_preds_ = self.estimator_ctrl.predict_proba(X_mod)[:, 1]
            
        else:
            
            self.ctrl_preds_ = self.estimator_ctrl.predict_proba(X_ctrl)[:, 1]
            self.trmnt_preds_ = self.estimator_trmnt.predict_proba(X_trmnt)[:, 1]

        uplift = self.trmnt_preds_ - self.ctrl_preds_

        return uplift
