"""
Dataset class for the imputation model AIHN.
"""

# Created by Knut Stroemmen <knut.stroemmen@unibe.com> and Rafael Morand <rafael.morand@unibe.ch>
# License: BSD-3-Clause

from typing import Union, Iterable

import torch
from pygrinder import mcar, fill_and_get_mask_torch

from ...data.dataset import BaseDataset


class DatasetMainFrequencyBased(BaseDataset):
    """Dataset that are trained according to the main-frequency-based imputation and forecasting model AIHN.

    Parameters
    ----------
    data :
        The dataset for model input, should be a dictionary including keys as 'X' and 'y',
        or a path string locating a data file.
        If it is a dict, X should be array-like of shape [n_samples, sequence length (n_steps), n_features],
        which is time-series data for input, can contain missing values, and y should be array-like of shape
        [n_samples], which is classification labels of X.
        If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
        key-value pairs like a dict, and it has to include keys as 'X' and 'y'.
        
    main_freq :
        The main frequency (in samples) of the time-series data. The main frequency is the number of samples it takes
        until the time-series data has a high auto-correlation. Examples:
        -   weather: 24 hour daily cycle or the 365 day yearly cycle
        -   heart rate: daily cycle (circadian rhythm)
        
    task :
        The dataset can be preprocessed for either one of ['imputation', 'forecasting']. 
        If 'imputation', the added missings in the dataset will be random (PyGrinder: mcar).
        If 'forecasting', the added missings will be at the end of the time-series data.

    return_y :
        Whether to return labels in function __getitem__() if they exist in the given data. If `True`, for example,
        during training of classification models, the Dataset class will return labels in __getitem__() for model input.
        Otherwise, labels won't be included in the data returned by __getitem__(). This parameter exists because we
        need the defined Dataset class for all training/validating/testing stages. For those big datasets stored in h5
        files, they already have both X and y saved. But we don't read labels from the file for validating and testing
        with function _fetch_data_from_file(), which works for all three stages. Therefore, we need this parameter for
        distinction.

    file_type :
        The type of the given file if train_set and val_set are path strings.

    rate : float, in (0,1),
        Artificially missing rate, rate of the observed values which will be artificially masked as missing.
        Note that, `rate` = (number of artificially missing values) / np.sum(~np.isnan(self.data)),
        not (number of artificially missing values) / np.product(self.data.shape),
        considering that the given data may already contain missing values,
        the latter way may be confusing because if the original missing rate >= `rate`,
        the function will do nothing, i.e. it won't play the role it has to be.
    """

    def __init__(
        self,
        data: Union[dict, str],
        main_freq: int,
        task: str,
        return_X_ori: bool,
        return_y: bool,
        file_type: str = "hdf5",
        prediction_horizon: int=None,
        rate: float = 0.2,
        nan_value: float = 0.0,
    ):
        super().__init__(
            data=data,
            return_X_ori=return_X_ori,
            return_X_pred=False,
            return_y=return_y,
            file_type=file_type,
        )
        assert task in ["imputation", "forecasting"], f"task should be one of ['imputation', 'forecasting'], but got {task}."
        if task == "forecasting": 
            assert prediction_horizon is not None, "prediction_horizon should be given for forecasting task."
            
        self.rate = rate
        self.main_freq = main_freq
        self.nan_value = nan_value
        
        n_samples, n_steps, self.n_features = self.X.shape
        self.tail = n_steps % self.main_freq # needed for trimming of the time-series data if it is not divisible by the main frequency
        self.n_steps_after_trim = n_steps // self.main_freq
        if self.tail != 0:
            self.reshape_by_main_freq_func = self._reshape_by_main_freq_with_trim
        else:
            self.reshape_by_main_freq_func = self._reshape_by_main_freq_without_trim
            
        if task == "imputation":
            self.missing_func = mcar
        elif task == "forecasting":
            # Proposition
            # - set the last row of the time-series data as missing values
            # - set the mask for the loss to a desired prediction horizon. this avoids that far future predictions are penalized.
            raise NotImplementedError("Forecasting task is not implemented yet.")
        
    def _reshape_by_main_freq_with_trim(self, X):
        """ Reshape the time-series data by the main frequency. 
            Trim the tail if the time-series data is not divisible by the main frequency.

        Returns
        -------
        X :
            The reshaped time-series data.
        """
        X = X[:-self.tail, :] 
        return self._reshape_by_main_freq_without_trim(X)
        
    def _reshape_by_main_freq_without_trim(self, X):
        """ Reshape the time-series data by the main frequency.

        Returns
        -------
        X :
            The reshaped time-series data.
        """
        X = X.reshape(self.n_steps_after_trim, self.main_freq, self.n_features)
        # stack the data to create 2D "images" of the time-series data.
        # the width of the image is the main frequency, the height is the number of steps * number of features
        # the features are stacked in order (first n_steps are the first feature, next n_steps are the second feature, etc.)
        X = X.permute(2, 0 ,1)
        X = X.reshape(-1, X.shape[-1])
        return X

    def _fetch_data_from_array(self, idx: int) -> Iterable:
        """Fetch data according to index.

        Parameters
        ----------
        idx :
            The index to fetch the specified sample.

        Returns
        -------
        sample :
            A list contains

            index :
                The index of the sample.

            X_ori :
                Original time-series for calculating mask imputation loss.

            X :
                Time-series data with artificially missing values for model input.

            missing_mask :
                The mask records all missing values in X.

            indicating_mask :
                The mask indicates artificially missing values in X.
        """
        if self.return_X_ori:
            # get the original time-series data
            X = self.X[idx]
            X_ori = self.X_ori[idx]
            missing_mask = self.missing_mask[idx]
            indicating_mask = self.indicating_mask[idx]
            # reshape the time-series data by the main frequency
            X = self.reshape_by_main_freq_func(X)
            X_ori = self.reshape_by_main_freq_func(X_ori)
            missing_mask = self.reshape_by_main_freq_func(missing_mask)
            indicating_mask = self.reshape_by_main_freq_func(indicating_mask)
        else:
            X_ori = self.X[idx]
            X_ori = self.reshape_by_main_freq_func(X_ori)
            X = mcar(X_ori, p=self.rate)
            X, missing_mask = fill_and_get_mask_torch(X, nan=self.nan_value)
            X_ori, X_ori_missing_mask = fill_and_get_mask_torch(X_ori, nan=self.nan_value)
            indicating_mask = (X_ori_missing_mask - missing_mask).to(torch.float32)

        sample = [
            torch.tensor(idx),
            X,
            missing_mask,
            X_ori,
            indicating_mask,
        ]

        if self.return_y:
            sample.append(self.y[idx].to(torch.long))

        return sample

    def _fetch_data_from_file(self, idx: int) -> Iterable:
        raise NotImplementedError("This method is not implemented for this Dataset.")
