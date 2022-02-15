from typing import List, Optional, Union, Dict, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from collections import Counter

from babylon_pgm.triage_models.triage_decisions import TRIAGE_MODEL_DECISION_TYPE, \
    TriageDecisionBase, TriageModelDecisionUs

from ..interfaces import (
    TriageModel,
    TriageDataset,
    PreparedData,
    TriageDataTransformer,
    PreparedDataTransformer
)
from ..triage_data_transformers import BaseDataExtractor


class Rewards:
    def __init__(
            self,
            *,
            reward_appropriate: float = 2.0,
            reward_safe: float = 0.0,
            reward_unsafe: float = -2.0,
    ):
        """

        :param reward_appropriate: reward for appropriate triage decision
        :param reward_safe: reward for save triage decision
        :param reward_unsafe: reward for unsafe triage decision
        """
        self.reward_appropriate = reward_appropriate
        self.reward_safe = reward_safe
        self.reward_unsafe = reward_unsafe
        self.action_costs = {
            "SELF_CARE": 0,
            "PHARMACY": 0,
            "GP": 0.1,
            "GP_URGENT": 0.1,
            "HOSPITAL": 0.3,
            "HOSPITAL_URGENT": 0.5,
        }
        self.max_cost = max([v for v in self.action_costs.values()])

    def get(
            self,
            action: Union[int, TRIAGE_MODEL_DECISION_TYPE],
            label: List[TRIAGE_MODEL_DECISION_TYPE],
    ) -> float:
        """
        :param action: predicted triage decision index or value
        :param label: list of triage decisions provided by doctors
        :return: reward for the selected action
        """
        if isinstance(action, TriageDecisionBase):
            action = action.index
        cost = self.action_costs[TriageModelDecisionUs.get_by_index(action).value]
        if action < min(label).index:
            return self.reward_unsafe * (1 + cost)
        if action > max(label).index:
            return self.reward_safe - cost
        return self.reward_appropriate * (1 + self.max_cost - cost)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, output_dim),
            nn.Softmax())

    def forward(self, X: torch.Tensor):
        return self.net(X)


class Agent:
    def __init__(
            self,
            actions: Union[npt.NDArray[int], List[int]],
            context_size: int,
            learning_rate: float = 1e-3,
            epsilon: float = 0.2,
            beta: float = 1e-2,
            alpha: float = 0.1,
            act_greedy: bool = False,
    ):
        """
        :param actions: action space
        :param context_size:  environment state size
        :param learning_rate: optimiser learning rate
        :param epsilon: parameter for epsilon-greedy exploration
        :param beta: entropy regularization weight
        :param alpha: parameter to encourage exploration
        :param act_greedy:  if True, agent acts greedy, if False, Thompson sampling is used.
        """
        self.policy_network = PolicyNetwork(
            input_dim=context_size,
            output_dim=len(actions),
        )
        self.actions = actions
        self.action_count = {action: 0 for action in self.actions}
        self.epsilon = epsilon
        self.beta = beta
        self.alpha = alpha
        self.act_greedy = act_greedy
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), learning_rate, weight_decay=1e-2
        )

    def predict(
            self,
            context: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict action probabilities using policy network

        :param context: environment state
        :return:  policy network predictions
        """
        outputs = self.policy_network(context)
        outputs = outputs + 1e-10
        outputs = outputs / torch.sum(outputs)
        return outputs

    def act(
            self,
            context: torch.Tensor,
    ) -> int:
        """
        Choose action given context.

        :param context: environment state
        :return: selected action
        """
        action_probs = self.predict(context).detach().numpy()

        if np.random.rand(1) < self.epsilon:
            action = np.random.choice(self.actions)
        elif self.act_greedy:
            action = list(self.actions)[action_probs.argmax().tolist()]
        else:
            action = np.random.choice(self.actions, p=action_probs)

        if action == np.argmax(self.action_count) and np.random.rand(1) < self.alpha:
            action = np.random.randint(len(self.actions))

        self.action_count[action] += 1
        return action

    def update(
            self,
            context: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor,
    ) -> None:
        """
        Update policy network parameters.

        :param context: batch of environment contexts
        :param action: batch of agent's actions
        :param reward: batch of rewards assigned to agent's actions
        """
        self.optimizer.zero_grad()
        probs = self.predict(context)
        logprobs = torch.log(probs)
        selected_logprobs = reward * torch.gather(logprobs, 1, (action).unsqueeze_(-1)).squeeze()
        entropy = -(probs * logprobs).sum()
        loss = -selected_logprobs.sum() - self.beta * entropy
        loss.backward()
        self.optimizer.step()


class Environment(Dataset):
    def __init__(
            self,
            *,
            contexts: np.ndarray,
            labels: List[List[TRIAGE_MODEL_DECISION_TYPE]],
            rewards: Rewards,
            agent: Agent
    ):
        """
        :param contexts: environment states
        :param labels: list of doctor decisions per case card.
                 Each entry is a list containing all the doctors' unique decisions.
        :param rewards: Rewards object
        :param agent: Agent object
        """
        self.contexts = contexts
        self.labels = labels
        if labels is None:
            self.class_weights = {}
        else:
            class_frequencies = Counter(
                [item.index for sublist in labels for item in sublist]
            )
            self.class_weights = {
                cl: .5 * self.contexts.shape[0] / class_frequencies[cl]
                for cl in class_frequencies
            }

        self.rewards = rewards
        self.agent = agent

    def __len__(self):
        """
        Return dataset length
        """
        return self.contexts.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the environment.

        :param idx: sample index
        :return: dictionary containing environment context, agent's action
            and corresponding reward
        """
        context = torch.from_numpy(self.contexts[idx]).float()
        action = self.agent.act(context)
        if self.labels is None:
            reward = -np.inf
        else:
            reward = self.rewards.get(action, self.labels[idx])
            # if reward > 0:
            #     reward *= self.class_weights[action]
            # else:
            #     reward *= 1 / self.class_weights[action]
        return {
            "context": context,
            "action": torch.tensor(action),
            "reward": torch.tensor(reward),
        }


class TriageBandit(TriageModel):
    def __init__(
            self,
            triage_decisions: Optional[TRIAGE_MODEL_DECISION_TYPE] = TriageModelDecisionUs,
            preprocessing_pipeline: Optional[
                List[Tuple[str, Union[TriageDataTransformer, PreparedDataTransformer]]]
            ] = [('data_extraction', BaseDataExtractor())],
            learning_rate: Optional[float] = 1e-3,
            epsilon: Optional[float] = 0.2,
            beta: Optional[float] = 1e-1,
            alpha: Optional[float] = 0.1,
            act_greedy: Optional[bool] = False,
            reward_appropriate: Optional[float] = 2.0,
            reward_safe: Optional[float] = 0.0,
            reward_unsafe: Optional[float] = -2.0,
    ):
        """
        :param triage_decisions: Triage decisions to be used (according to world region).
        :param preprocessing_pipeline: List of transformations to be applied to the data
                                        before passing them to the model.
                                        Example: [("transform0", TriageDataTransformer)]
        :param learning_rate: optimiser learning rate
        :param epsilon: parameter for epsilon-greedy exploration
        :param beta: entropy regularization weight
        :param alpha: parameter to encourage exploration
        :param act_greedy: if True, agent acts greedy, if False, Thompson sampling is used.
        :param reward_appropriate: reward for appropriate decisions
        :param reward_safe: reward for safe decisions
        :param reward_unsafe: reward for unsafe decisions
        """
        self._triage_decisions = triage_decisions
        self.preprocessing_pipeline = preprocessing_pipeline
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta = beta
        self.alpha = alpha
        self.act_greedy = act_greedy
        self.rewards = Rewards(
            reward_appropriate=reward_appropriate,
            reward_safe=reward_safe,
            reward_unsafe=reward_unsafe,
        )
        self._params_to_track = {
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "beta": self.beta,
            "alpha": self.alpha,
            "reward_appropriate": reward_appropriate,
            "reward_safe": reward_safe,
            "reward_unsafe": reward_unsafe,
            "preprocessing": [
                step[1].__class__.__name__
                for step in self.preprocessing_pipeline
            ],
        }

    def __str__(self):
        return "Bandit"

    def fit(
            self,
            data: TriageDataset,
            batch_size: Optional[int] = None,
            num_epochs: Optional[int] = 300,
    ):
        """
        Train the model.

        :param data: Training data.
        :param batch_size: Optimization batch size.
                            If None, batch_size is equal to sample size.
        :param num_epochs: Number of training epochs.
        :return:
        """
        self._params_to_track["batch_size"] = batch_size
        self._params_to_track["num_epochs"] = num_epochs

        for proc in self.preprocessing_pipeline:
            data = proc[1].fit_transform(data, train=True)

        # Create agent
        self.agent = Agent(
            np.arange(self._triage_decisions.max_index() + 1),
            data.features.shape[1],
            learning_rate=self.learning_rate,
            epsilon=self.epsilon,
            beta=self.beta,
            alpha=self.alpha,
            act_greedy=self.act_greedy,
        )

        train_env = Environment(
            contexts=data.features,
            labels=data.correct_decisions,
            rewards=self.rewards,
            agent=self.agent,
        )

        if batch_size is None:
            batch_size = len(train_env)
        train_ldr = DataLoader(train_env, batch_size=batch_size, shuffle=True)
        epoch_rewards = np.zeros(num_epochs)

        for epoch in range(num_epochs):
            for (batch_idx, batch) in enumerate(train_ldr):
                train_env.agent.update(
                    batch["context"], batch["action"], batch["reward"]
                )
                epoch_rewards[epoch] += batch["reward"].sum()

            epoch_rewards[epoch] /= len(train_env)
            print(f"Epoch {epoch}: average reward = {epoch_rewards[epoch]}")

            if epoch == 0:
                best_agent = deepcopy(train_env.agent)
            elif epoch_rewards[epoch] > np.max(epoch_rewards[:epoch]):
                best_agent = deepcopy(train_env.agent)

        self.agent = best_agent

    def predict(self, data: TriageDataset) -> List[TRIAGE_MODEL_DECISION_TYPE]:
        """
        Predict triage decisions given a TriageDataset object.
        """
        for proc in self.preprocessing_pipeline:
            data = proc[1].transform(data, train=True)

        actions, rewards = self._predict(data)
        decisions = [
            self._triage_decisions.get_by_index(actions[i])
            for i in range(len(data))
        ]
        return decisions

    def _predict(self, data: PreparedData):
        test_agent = deepcopy(self.agent)
        test_agent.epsilon = 0.0
        test_agent.alpha = 0.0
        test_agent.act_greedy = True

        test_env = Environment(
            contexts=data.features,
            labels=data.correct_decisions,
            rewards=self.rewards,
            agent=test_agent,
        )
        test_ldr = DataLoader(test_env, batch_size=len(test_env), shuffle=False)
        actions = next(iter(test_ldr))["action"]
        rewards = next(iter(test_ldr))["reward"]
        return actions, rewards
