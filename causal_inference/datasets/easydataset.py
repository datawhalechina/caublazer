# **
# * ����
# *
# * @author 雁楚
# * @edit 雁楚

import sys
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.util import hash_pandas_object


class EasyDataset:
    def __init__(self,
                 data,
                 covariates,
                 treat,
                 outcomes,
                 id=None,
                 categorical_covariates=None,
                 propensity=None):
        # Preprocess arguments.
        if categorical_covariates is None:
            categorical_covariates = []

        # Check whether covariates / treat / outcomes are valid.
        all_variables = list(list(covariates) + list(treat) + list(outcomes))

        # for v in all_variables:
        #     assert v in data.columns.values, '{} is not in the data.'.format(v)
        #     assert v in categorical_covariates or is_numeric_dtype(
        #         data[v]), '{} is not numeric.'.format(v)
        # for v in categorical_covariates:
        #     assert v in data.columns.values, '{} is not a covariate.'.format(v)dat

        # Check the treat column.
        # assert is_binary(data[treat]), 'Treatment indicator should be 0 or 1.'
        # assert data[data[treat] == 1].shape[0] > 0, 'No treated unit.'
        # assert data[data[treat] == 0].shape[0] > 1, 'No control unit.'

        # Check the propensity score.
        if propensity is not None:
            assert len(propensity) == data.shape[0]

        # Initialize other variables.
        self.data = data[all_variables].fillna(0.0).reset_index(drop=True)
        self.id = id
        self.propensity = propensity
        self.covariates = covariates
        self.treat = treat
        self.outcomes = outcomes
        self.size = self.data.shape[0]

        # One-hot encoding.
        if categorical_covariates:
            # Get one-hot encoding.
            self.data = pd.get_dummies(self.data,
                                       columns=categorical_covariates,
                                       dtype=int)

            # Update covariates.
            self.covariates = sorted(
                list(set(self.data.columns.values) - {treat} - set(outcomes)))

    def get_id(self):
        if self.id is not None:
            return self.data[self.id]
        else:
            return pd.DataFrame(np.arange(self.data.shape[0]), columns=['id'])

    def get_treatment(self):
        return self.data[self.treat]

    def get_covariates(self):
        return self.data[self.covariates]

    def get_outcome(self):
        return self.data[self.outcomes]

    def set_outcome(self, outcome, v):
        self.data[outcome] = v

    def get_data_array(self):
        return np.array(self.data)

    def get_data_columns(self):
        return self.data.columns

    def has_propensity_score(self):
        return self.propensity is not None

    def add_propensity_score(self, propensity):
        assert len(propensity) == self.data.shape[0]
        self.propensity = propensity

    def get_propensity_score(self):
        return self.propensity

    def subset(self, idx, keep_propensity):
        data = self.data.loc[idx, :]
        if keep_propensity and self.propensity is not None:
            propensity = self.propensity[idx]
        else:
            propensity = None
        return EasyDataset(data, self.covariates, self.treat, self.outcomes,
                           propensity)
