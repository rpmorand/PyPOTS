"""
The base class for imputation models.
"""
# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3


import os
from abc import abstractmethod

import numpy as np
import torch

from pypots.base import BaseModel, BaseNNModel
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_mae

try:
    import nni
except ImportError:
    pass


class BaseImputer(BaseModel):
    """Abstract class for all imputation models."""

    def __init__(self, device):
        super().__init__(device)

    @abstractmethod
    def fit(self, train_set, val_set=None, file_type="h5py"):
        """Train the imputer on the given data.

        Parameters
        ----------
        train_set : dict or str,
            The dataset for model training, should be a dictionary including the key 'X',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (time steps), n_features],
            which is time-series data for training, can contain missing values.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include the key 'X'.

        val_set : dict or str,
            The dataset for model validating, should be a dictionary including the key 'X',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (time steps), n_features],
            which is time-series data for validating, can contain missing values.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include the key 'X'.

        file_type : str, default = "h5py",
            The type of the given file if train_set and val_set are path strings.

        Returns
        -------
        self : object,
            The trained imputer.
        """
        return self

    @abstractmethod
    def impute(self, X, file_type="h5py"):
        """Impute missing values in the given data with the trained model.

        Parameters
        ----------
        X : array-like or str,
            The data samples for testing, should be array-like of shape [n_samples, sequence length (time steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type : str, default = "h5py",
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples, sequence length (time steps), n_features],
            Imputed data.
        """
        pass


class BaseNNImputer(BaseNNModel, BaseImputer):
    def __init__(
        self, learning_rate, epochs, patience, batch_size, weight_decay, device
    ):
        super().__init__(
            learning_rate, epochs, patience, batch_size, weight_decay, device
        )

    @abstractmethod
    def assemble_input_for_training(self, data) -> dict:
        """Assemble the given data into a dictionary for training input.

        Parameters
        ----------
        data : list,
            Input data from dataloader, should be list.

        Returns
        -------
        dict,
            A python dictionary contains the input data for model training.
        """
        pass

    @abstractmethod
    def assemble_input_for_validating(self, data) -> dict:
        """Assemble the given data into a dictionary for validating input.

        Parameters
        ----------
        data : list,
            Data output from dataloader, should be list.

        Returns
        -------
        dict,
            A python dictionary contains the input data for model validating.
        """
        pass

    @abstractmethod
    def assemble_input_for_testing(self, data) -> dict:
        """Assemble the given data into a dictionary for testing input.

        Notes
        -----
        The processing functions of train/val/test stages are separated for the situation that the input of
        the three stages are different, and this situation usually happens when the Dataset/Dataloader classes
        used in the train/val/test stages are not the same, e.g. the training data and validating data in a
        classification task contains labels, but the testing data (from the production environment) generally
        doesn't have labels.

        Parameters
        ----------
        data : list,
            Data output from dataloader, should be list.

        Returns
        -------
        dict,
            A python dictionary contains the input data for model testing.
        """
        pass

    def _train_model(
        self,
        training_loader,
        val_loader=None,
        val_X_intact=None,
        val_indicating_mask=None,
    ):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float("inf")
        self.best_model_dict = None

        try:
            for epoch in range(self.epochs):
                self.model.train()
                epoch_train_loss_collector = []
                for idx, data in enumerate(training_loader):
                    inputs = self.assemble_input_for_training(data)
                    self.optimizer.zero_grad()
                    results = self.model.forward(inputs)
                    results["loss"].backward()
                    self.optimizer.step()
                    epoch_train_loss_collector.append(results["loss"].item())

                mean_train_loss = np.mean(
                    epoch_train_loss_collector
                )  # mean training loss of the current epoch
                self.logger["training_loss"].append(mean_train_loss)

                if val_loader is not None:
                    self.model.eval()
                    imputation_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_loader):
                            inputs = self.assemble_input_for_validating(data)
                            imputed_data, _ = self.model.impute(inputs)
                            imputation_collector.append(imputed_data)

                    imputation_collector = torch.cat(imputation_collector)
                    imputation_collector = imputation_collector.numpy()

                    mean_val_loss = cal_mae(
                        imputation_collector, val_X_intact, val_indicating_mask
                    )
                    self.logger["validating_loss"].append(mean_val_loss)
                    logger.info(
                        f"epoch {epoch}: training loss {mean_train_loss:.4f}, validating loss {mean_val_loss:.4f}"
                    )
                    mean_loss = mean_val_loss
                else:
                    logger.info(f"epoch {epoch}: training loss {mean_train_loss:.4f}")
                    mean_loss = mean_train_loss

                if mean_loss < self.best_loss:
                    self.best_loss = mean_loss
                    self.best_model_dict = self.model.state_dict()
                    self.patience = self.original_patience
                else:
                    self.patience -= 1

                if os.getenv("enable_nni", False):
                    nni.report_intermediate_result(mean_loss)
                    if epoch == self.epochs - 1 or self.patience == 0:
                        nni.report_final_result(self.best_loss)

                if self.patience == 0:
                    logger.info(
                        "Exceeded the training patience. Terminating the training procedure..."
                    )
                    break

        except Exception as e:
            logger.info(f"Exception: {e}")
            if self.best_model_dict is None:
                raise RuntimeError(
                    "Training got interrupted. Model was not get trained. Please try fit() again."
                )
            else:
                RuntimeWarning(
                    "Training got interrupted. "
                    "Model will load the best parameters so far for testing. "
                    "If you don't want it, please try fit() again."
                )

        if np.equal(self.best_loss.item(), float("inf")):
            raise ValueError("Something is wrong. best_loss is Nan after training.")

        logger.info("Finished training.")
