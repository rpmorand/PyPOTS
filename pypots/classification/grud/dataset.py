"""
Dataset class for model GRU-D.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


from typing import Union, Iterable

import torch

from pypots.data.base import BaseDataset
from pypots.data.utils import torch_parse_delta
from pypots.imputation.locf import LOCF


class DatasetForGRUD(BaseDataset):
    """Dataset class for model GRU-D.

    Parameters
    ----------
    data : dict or str,
        The dataset for model input, should be a dictionary including keys as 'X' and 'y',
        or a path string locating a data file.
        If it is a dict, X should be array-like of shape [n_samples, sequence length (time steps), n_features],
        which is time-series data for input, can contain missing values, and y should be array-like of shape
        [n_samples], which is classification labels of X.
        If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
        key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

    return_labels : bool, default = True,
        Whether to return labels in function __getitem__() if they exist in the given data. If `True`, for example,
        during training of classification models, the Dataset class will return labels in __getitem__() for model input.
        Otherwise, labels won't be included in the data returned by __getitem__(). This parameter exists because we
        need the defined Dataset class for all training/validating/testing stages. For those big datasets stored in h5
        files, they already have both X and y saved. But we don't read labels from the file for validating and testing
        with function _fetch_data_from_file(), which works for all three stages. Therefore, we need this parameter for
        distinction.

    file_type : str, default = "h5py"
        The type of the given file if train_set and val_set are path strings.
    """

    def __init__(
        self,
        data: Union[dict, str],
        return_labels: bool = True,
        file_type: str = "h5py",
    ):
        super().__init__(data, return_labels, file_type)
        self.locf = LOCF()

        if not isinstance(self.data, str):  # data from array
            self.missing_mask = (~torch.isnan(self.X)).to(torch.float32)
            self.X_filledLOCF = self.locf.locf_torch(self.X)
            self.X = torch.nan_to_num(self.X)
            self.deltas = torch_parse_delta(self.missing_mask)
            self.empirical_mean = torch.sum(
                self.missing_mask * self.X, dim=[0, 1]
            ) / torch.sum(self.missing_mask, dim=[0, 1])

    def _fetch_data_from_array(self, idx: int) -> Iterable:
        """Fetch data according to index.

        Parameters
        ----------
        idx : int,
            The index to fetch the specified sample.

        Returns
        -------
        sample : list,
            A list contains

            index : int tensor,
                The index of the sample.

            X : tensor,
                The feature vector for model input.

            X_filledLOCF: tensor,
                The feature vector filled with last observations.

            missing_mask : tensor,
                The mask indicates all missing values in X.

            delta : tensor,
                The delta matrix contains time gaps of missing values.

            empirical_mean : tensor,
                Mean values of features.
        """
        sample = [
            torch.tensor(idx),
            self.X[idx].to(torch.float32),
            self.X_filledLOCF[idx].to(torch.float32),
            self.missing_mask[idx].to(torch.float32),
            self.deltas[idx].to(torch.float32),
            self.empirical_mean.to(torch.float32),
        ]

        if self.y is not None and self.return_labels:
            sample.append(self.y[idx].to(torch.long))

        return sample

    def _fetch_data_from_file(self, idx: int) -> Iterable:
        """Fetch data with the lazy-loading strategy, i.e. only loading data from the file while requesting for samples.
        Here the opened file handle doesn't load the entire dataset into RAM but only load the currently accessed slice.

        Parameters
        ----------
        idx : int,
            The index of the sample to be return.

        Returns
        -------
        sample : list,
            The collated data sample, a list including all necessary sample info.
        """

        if self.file_handle is None:
            self.file_handle = self._open_file_handle()

        X = torch.from_numpy(self.file_handle["X"][idx])
        missing_mask = (~torch.isnan(X)).to(torch.float32)
        X_filledLOCF = self.locf.locf_torch(X.unsqueeze(dim=0)).squeeze()
        X = torch.nan_to_num(X)
        deltas = torch_parse_delta(missing_mask)
        empirical_mean = torch.sum(missing_mask * X, dim=[0]) / torch.sum(
            missing_mask, dim=[0]
        )

        sample = [
            torch.tensor(idx),
            X,
            X_filledLOCF,
            missing_mask,
            deltas,
            empirical_mean,
        ]

        # if the dataset has labels and is for training, then fetch it from the file
        if "y" in self.file_handle.keys() and self.return_labels:
            sample.append(torch.tensor(self.file_handle["y"][idx], dtype=torch.long))

        return sample