from typing import Optional, List, Tuple, Union

import torch
from torch.nn import Module, Parameter, LogSoftmax
from tqdm import tqdm
import numpy as np
from collections import Counter
from time import time

from babylon_pgm.triage_models.defaults import UtilityModelParameters
from babylon_pgm.triage_models.triage_decisions import TRIAGE_MODEL_DECISION_TYPE, TriageModelDecisionDefault
from babylon_pgm.metrics.triage_metrics import TriageMetrics

from ..interfaces import (
    TriageDataset,
    TriageModel,
    TriageDataTransformer,
    PreparedDataTransformer,
)
from ..triage_data_transformers import UtilityModelTransformer


DEFAULT_NUM_EPOCHS = 20_000
DEFAULT_BATCH_SIZE = 500
DEFAULT_PREPROCESSING = [('utility', UtilityModelTransformer())]


def fbeta_score(
        y_true: torch.LongTensor,
        y_pred: torch.LongTensor,
        num_classes: int,
        beta: int
) -> float:
    """
    Computes the macro-averaged F-beta score

    :param y_true: 1-dimensional torch.LongTensor representing predicted class
             labels, or a 2-dimensional torch.LongTensor where the first
             dimension represents batches.
    :param y_pred: 1-dimensional torch.LongTensor representing predicted class
             labels, or a 2-dimensional torch.LongTensor where the first
             dimension represents batches.
    :param num_classes: number of classes.
    :param beta: importance of recall relative to precision.
    :return: fbeta score for each batch
    """
    eps = 1e-9
    y_pred_one_hot = one_hot_encode(y_pred, num_classes)
    y_true_one_hot = one_hot_encode(y_true, num_classes)

    true_positive = (y_pred_one_hot * y_true_one_hot).sum(dim=1)
    precision = true_positive.div(y_pred_one_hot.sum(dim=1).add(eps)).mean(
        dim=y_pred.dim() - 1
    )
    recall = true_positive.div(y_true_one_hot.sum(dim=1).add(eps)).mean(
        dim=y_pred.dim() - 1
    )
    beta2 = beta ** 2
    fbeta = (precision * recall).div(precision.mul(beta2) + recall + eps).mul(1 + beta2)

    return fbeta


def one_hot_encode(
        x: torch.LongTensor,
        num_classes: int,
) -> torch.FloatTensor:
    """
    Encode an tensor as a one-hot tensor.

    :param x:  N-dimensional torch.LongTensor representing the num_classes classes.
    :param num_classes: number of classes.
    :return:  (N+1)-dimensional torch.FloatTensor, where one-hot encoding is
              along the (N+1)th dimension represents the classes
    """
    one_hot = x.new(*[*list(x.shape), num_classes]).zero_().float()
    num_dims = len(x.shape)
    one_hot.scatter_(num_dims, torch.unsqueeze(x, num_dims), 1)
    return one_hot


class TrainableTriageUtilityModel(TriageModel, Module):
    def __init__(
        self,
        triage_decisions: Optional[TRIAGE_MODEL_DECISION_TYPE] = TriageModelDecisionDefault,
        preprocessing_pipeline: Optional[
            List[Tuple[str, Union[TriageDataTransformer, PreparedDataTransformer]]]
        ] = DEFAULT_PREPROCESSING,
        alpha: Optional[float] = 0.5
    ):
        """
        :param triage_decisions: Triage decisions to be used (according to world region).
        :param preprocessing_pipeline: List of transformations to be applied to the data
                                        before passing them to the model.
                                        Example: [("transform0", TriageDataTransformer)]
        """
        Module.__init__(self)

        self._triage_decisions = triage_decisions
        self.preprocessing_pipeline = preprocessing_pipeline
        self._params_to_track = {
            "preprocessing": [
                step[1].__class__.__name__
                for step in self.preprocessing_pipeline
            ],
            "alpha": alpha,
        }

        parameters = UtilityModelParameters(
            cruelty_parameters=[0.0] * (len(triage_decisions) - 1),
            system_parameters=[0.0] * (len(triage_decisions) - 1),
            alpha=alpha,
        )
        self.cruelty_parameters = Parameter(torch.Tensor(parameters.cruelty_parameters))
        self.system_parameters = Parameter(torch.Tensor(parameters.system_parameters))
        self.alpha = parameters.alpha

    def __str__(self):
        return "Trainable utility model"

    def _get_cruelty_costs(self):
        return torch.cat((torch.zeros(1), self.cruelty_parameters))

    def _get_system_costs(self):
        return torch.cat((torch.zeros(1), self.system_parameters))

    def get_urgency_utility(self, likelihoods, disease_triages, decision_mask):
        cruelty_costs = self._get_cruelty_costs()
        cruelties = cruelty_costs[disease_triages]
        # Expected utilities for cruelty costs
        cruelty_utilities = (likelihoods * cruelties).expand(
            len(self._triage_decisions), -1, -1
        ).permute(1, 2, 0) * decision_mask
        cruelty_utility = cruelty_utilities.sum(1)  # Sum over diseases
        return cruelty_utility

    def _get_system_utility(self):
        return self._get_system_costs()

    def fit(
            self,
            data: TriageDataset,
            num_initialisations: Optional[int] = 1,
            num_epochs: Optional[int] = DEFAULT_NUM_EPOCHS,
            batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
    ) -> None:
        """
        Train the model.

        :param data: training data
        :param num_initialisations: number of initialisations for the parameters optimisation
        :param num_epochs: number of training epochs
        :param batch_size: batch size
        """
        self._params_to_track["num_initialisations"] = num_initialisations
        self._params_to_track["num_epochs"] = num_epochs
        self._params_to_track["batch_size"] = batch_size

        for proc in self.preprocessing_pipeline:
            data = proc[1].fit_transform(data)

        # Randomise parameters
        self.cruelty_parameters.data = torch.randn(self.cruelty_parameters.shape)
        self.system_parameters.data = torch.randn(len(self.system_parameters))

        best_loss = np.Inf
        with tqdm(total=num_initialisations) as pbar:
            for i in range(num_initialisations):
                pbar.set_description(f"Training model, initialisation {i+1}")
                training_loss = self._train(
                    data=data, num_epochs=num_epochs, batch_size=batch_size
                )
                if training_loss < best_loss:
                    best_parameters = {
                        "cruelty_parameters": self.cruelty_parameters,
                        "system_parameters": self.system_parameters,
                    }
                pbar.update()

        self.cruelty_parameters.data = best_parameters["cruelty_parameters"]
        self.system_parameters.data = best_parameters["system_parameters"]

    def _train(self, *, data, num_epochs, batch_size):
        counts = Counter(data.correct_decisions.tolist())
        weights = torch.Tensor([counts[key] for key in sorted(counts.keys())])
        weights = max(weights) / weights
        # https://datascience.stackexchange.com/questions/48369/what-loss-function-to-use-for-imbalanced-classes-using-pytorch

        criterion = torch.nn.NLLLoss(weight=weights)
        targets = data.correct_decisions

        optimizer = torch.optim.Adam(self.parameters())

        print_interval = 1.0  # seconds
        last_updated = 0.0
        with tqdm(total=num_epochs, unit="epochs") as pbar:
            for i in range(num_epochs):
                # Draw samples
                batch_inds = torch.LongTensor(
                    np.random.choice(
                        len(data.correct_decisions), batch_size, replace=False
                    )
                )

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self(
                    data.likelihoods[batch_inds, :],
                    data.disease_triages[batch_inds, :],
                    data.decision_mask[batch_inds, :],
                )

                batch_targets = targets[batch_inds]
                batch_loss = criterion(outputs, batch_targets)

                batch_loss.backward()
                optimizer.step()

                # Only calculate progressbar stats at intervals
                if time() > last_updated + print_interval:
                    outputs = self(
                        data.likelihoods,
                        data.disease_triages,
                        data.decision_mask,
                    )
                    training_loss = criterion(outputs, targets)
                    _, predicted = outputs.max(1)
                    f1_score = fbeta_score(
                        data.correct_decisions.data,
                        predicted.data,
                        len(self._triage_decisions),
                        1,
                    )
                    pbar.set_postfix(loss=float(training_loss), f1=float(f1_score))
                    last_updated = time()
                pbar.update()

        return training_loss

    def forward(self, likelihoods, disease_triages, decision_mask):
        """
        Return decision probabilities in log space.
        """
        cruelty_utility = self.get_urgency_utility(
            likelihoods, disease_triages, decision_mask
        )
        system_utility = self._get_system_utility()

        total_utility = (self.alpha * system_utility) + (
            (1.0 - self.alpha) * cruelty_utility
        )
        logsoftmax = LogSoftmax(dim=1)
        return logsoftmax(total_utility)

    def predict(self, data: TriageDataset) -> List[TRIAGE_MODEL_DECISION_TYPE]:
        """
        Predict triage decisions given a TriageDataset object.
        """
        for proc in self.preprocessing_pipeline:
            data = proc[1].transform(data, train=False)

        outputs = self.forward(data.likelihoods, data.disease_triages, data.decision_mask)
        _, decision_indices = outputs.data.max(1)
        decisions = [
            self._triage_decisions.get_by_index(decision_index)
            for decision_index in decision_indices
        ]
        return decisions



class TriageUtilityModel(TrainableTriageUtilityModel):
    def __init__(
        self,
        triage_decisions: Optional[TRIAGE_MODEL_DECISION_TYPE] = TriageModelDecisionDefault,
        preprocessing_pipeline: Optional[
            List[Tuple[str, Union[TriageDataTransformer, PreparedDataTransformer]]]
        ] = DEFAULT_PREPROCESSING,
    ):
        """
        :param triage_decisions: Triage decisions to be used (according to world region).
        :param preprocessing_pipeline: List of transformations to be applied to the data
                                        before passing them to the model.
                                        Example: [("transform0", TriageDataTransformer)]
        """
        Module.__init__(self)

        self._triage_decisions = TriageModelDecisionDefault
        self.preprocessing_pipeline = preprocessing_pipeline

        parameters = UtilityModelParameters(
            cruelty_parameters=[-0.3417, -2.3638, -4.1604, -8.2556, -10.0988],
            system_parameters=[-0.093, -0.7589, -1.6567, -3.6925, -5.362],
            alpha=0.2361,
        )
        self.cruelty_parameters = Parameter(torch.Tensor(parameters.cruelty_parameters))
        self.system_parameters = Parameter(torch.Tensor(parameters.system_parameters))
        self.alpha = parameters.alpha

        self._params_to_track = {
            "preprocessing": [
                step[1].__class__.__name__
                for step in self.preprocessing_pipeline
            ],
            "alpha": parameters.alpha,
        }

    def __str__(self):
        return "Production utility model"

    def fit(
            self,
            data: TriageDataset,
            num_initialisations: Optional[int] = 1,
            num_epochs: Optional[int] = DEFAULT_NUM_EPOCHS,
            batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
    ):
        pass
