"""
The implementation of AIHN for the partially-observed time-series imputation task.

"""

# Created by Knut Stroemmen <knut.stroemmen@unibe.com> and Rafael Morand <rafael.morand@unibe.ch>, based on the structure of PyPOTS
# License: BSD-3-Clause

from typing import Union, Optional, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import _AIHN
from .data import DatasetMainFrequencyBased
from ..base import BaseNNImputer
from ...data.checking import key_in_data_set
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger
from ...utils.metrics import calc_mae

import os
from ...utils.file import create_dir_if_not_exist


class AIHN(BaseNNImputer):
    """The PyTorch implementation of the AIHN model.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.
        
    main_freq :
        The main frequency of the time-series data in samples. Example: Heart rate has 24-hour cycle (circadian rhythm)
        
    task :
        The task of the model. It can be either 'imputation' or 'forecasting'.

    n_layers :
        The number of layers in the 1st and 2nd DMSA blocks in the AIHN model.

    d_model :
        The dimension of the model's backbone.
        It is the input dimension of the multi-head DMSA layers.

    n_heads :
        The number of heads in the multi-head DMSA mechanism.
        ``d_model`` must be divisible by ``n_heads``, and the result should be equal to ``d_k``.

    d_k :
        The dimension of the `keys` (K) and the `queries` (Q) in the DMSA mechanism.
        ``d_k`` should be the result of ``d_model`` divided by ``n_heads``. Although ``d_k`` can be directly calculated
        with given ``d_model`` and ``n_heads``, we want it be explicitly given together with ``d_v`` by users to ensure
        users be aware of them and to avoid any potential mistakes.

    d_v :
        The dimension of the `values` (V) in the DMSA mechanism.

    d_ffn :
        The dimension of the layer in the Feed-Forward Networks (FFN).

    dropout :
        The dropout rate for all fully-connected layers in the model.

    attn_dropout :
        The dropout rate for DMSA.

    diagonal_attention_mask :
        Whether to apply a diagonal attention mask to the self-attention mechanism.
        If so, the attention layers will use DMSA. Otherwise, the attention layers will use the original.

    ORT_weight :
        The weight for the ORT loss.

    MIT_weight :
        The weight for the MIT loss.

    batch_size :
        The batch size for training and evaluating the model.

    epochs :
        The number of epochs for training the model.

    patience :
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    customized_loss_func:
        The customized loss function designed by users for the model to optimize.
        If not given, will use the default MAE loss as claimed in the original paper.

    optimizer :
        The optimizer for model training.
        If not given, will use a default Adam optimizer.

    num_workers :
        The number of subprocesses to use for data loading.
        `0` means data loading will be in the main process, i.e. there won't be subprocesses.

    device :
        The device for the model to run on. It can be a string, a :class:`torch.device` object, or a list of them.
        If not given, will try to use CUDA devices first (will use the default CUDA device if there are multiple),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        If given a list of devices, e.g. ['cuda:0', 'cuda:1'], or [torch.device('cuda:0'), torch.device('cuda:1')] , the
        model will be parallely trained on the multiple devices (so far only support parallel training on CUDA devices).
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path :
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss values recorded during
        training into a tensorboard file). Will not save if not given.

    model_saving_strategy :
        The strategy to save model checkpoints. It has to be one of [None, "best", "better", "all"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.
        The "all" strategy will save every model after each epoch training.

    verbose :
        Whether to print out the training logs during the training process.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        main_freq: int,
        task: str,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float = 0,
        attn_dropout: float = 0,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        customized_loss_func: Callable = calc_mae,
        optimizer: Optional[Optimizer] = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: Optional[str] = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            batch_size,
            epochs,
            patience,
            num_workers,
            device,
            saving_path,
            model_saving_strategy,
            verbose,
        )

        if d_model != n_heads * d_k:
            logger.warning(
                "‼️ d_model must = n_heads * d_k, it should be divisible by n_heads "
                f"and the result should be equal to d_k, but got d_model={d_model}, n_heads={n_heads}, d_k={d_k}"
            )
            d_model = n_heads * d_k
            logger.warning(f"⚠️ d_model is reset to {d_model} = n_heads ({n_heads}) * d_k ({d_k})")

        self.n_steps = n_steps//main_freq 
        self.n_features = n_features
        self.main_freq = main_freq
        self.task = task
        # model hype-parameters
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.customized_loss_func = customized_loss_func

        # set up the model
        self.model = _AIHN(
            self.n_layers,
            self.n_steps,
            self.main_freq,
            self.n_features,
            self.d_model,
            self.n_heads,
            self.d_k,
            self.d_v,
            self.d_ffn,
            self.dropout,
            self.attn_dropout,
            self.customized_loss_func
        )
        self._print_model_size()
        self._send_model_to_given_device()

        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: list) -> dict:
        (
            indices,
            X,
            missing_mask,
            X_ori,
            indicating_mask,
        ) = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        indices, X, missing_mask, X_ori, indicating_mask = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }
        return inputs

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetMainFrequencyBased(train_set, main_freq=self.main_freq, task=self.task,
                                                 return_X_ori=False, return_y=False, file_type=file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_set = DatasetMainFrequencyBased(val_set, main_freq=self.main_freq, task=self.task,
                                                return_X_ori=True, return_y=False, file_type=file_type)
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Step 2: train the model and freeze it
        self._train_model(training_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        return_latent_vars: bool = False,
    ) -> dict:
        """Make predictions for the input data with the trained model.

        Parameters
        ----------
        test_set :
            The dataset for model validating, should be a dictionary including keys as 'X',
            or a path string locating a data file supported by PyPOTS (e.g. h5 file).
            If it is a dict, X should be array-like of shape [n_samples, sequence length (n_steps), n_features],
            which is time-series data for validating, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        file_type :
            The type of the given file if test_set is a path string.

        diagonal_attention_mask :
            Whether to apply a diagonal attention mask to the self-attention mechanism in the testing stage.

        return_latent_vars :
            Whether to return the latent variables in AIHN, e.g. attention weights of two DMSA blocks and
            the weight matrix from the combination block, etc.

        Returns
        -------
        file_type :
            The dictionary containing the clustering results and latent variables if necessary.

        """
        
        n_individuals, n_steps, n_features = test_set['X'].shape
        
        # Step 1: wrap the input data with classes Dataset and DataLoader
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetMainFrequencyBased(
            test_set,
            self.main_freq,
            self.task,
            return_X_ori=False,
            return_y=False,
            file_type=file_type,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        imputation_collector = []

        # Step 2: process the data with the model
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(inputs, training=False)
                imputation_collector.append(results["imputed_data"])

        # Step 3: output collection and return
        imputation = torch.cat(imputation_collector).cpu().detach().numpy()
        
        # Step 4: reshape (i.e., undo the permutation and reshape)
        imputation = imputation.reshape(n_individuals, n_features, -1).transpose(0, 2, 1)
        
        result_dict = {
            "imputation": imputation,
        }

        return result_dict

    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:
        """Impute missing values in the given data with the trained model.

        Parameters
        ----------
        test_set :
            The data samples for testing, should be array-like of shape [n_samples, sequence length (n_steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples, sequence length (n_steps), n_features],
            Imputed data.
        """

        result_dict = self.predict(test_set, file_type=file_type)
        return result_dict["imputation"]


    def save(
        self,
        saving_path: str,
        overwrite: bool = False,
    ) -> None:
        """Save the model <AIHN> with current parameters to a disk file.

        Because this model has a modular structure, it is not saved in the pypots format. Instead, it is saved in the
        PyTorch format that allows this behavior.

        Parameters
        ----------
        saving_path :
            The given path to save the model. The directory will be created if it does not exist.

        overwrite :
            Whether to overwrite the model file if the path already exists.

        """
        # split the saving dir and file name from the given path
        saving_dir, file_name = os.path.split(saving_path)
        # if parent dir is not given, save in the current dir
        saving_dir = "." if saving_dir == "" else saving_dir
        # rejoin the path for saving the model
        saving_path = os.path.join(saving_dir, file_name)
        # add the suffix ".pypots" if not given
        if file_name.split(".")[-1] != "pth":
            file_name += ".pth"
        # rejoin the path for saving the model
        saving_path = os.path.join(saving_dir, file_name)

        if os.path.exists(saving_path):
            if overwrite:
                logger.warning(f"‼️ File {saving_path} exists. Argument `overwrite` is True. Overwriting now...")
            else:
                logger.error(
                    f"❌ File {saving_path} exists. Saving operation aborted. "
                    f"Use the arg `overwrite=True` to force overwrite."
                )
                return

        try:
            create_dir_if_not_exist(saving_dir)
            torch.save(self.model.state_dict(), saving_path)
            logger.info(f"Saved the model to {saving_path}")
        except Exception as e:
            raise RuntimeError(f'Failed to save the model to "{saving_path}" because of the below error! \n{e}')