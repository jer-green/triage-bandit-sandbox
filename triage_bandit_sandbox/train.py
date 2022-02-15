import json
from pathlib import Path
from tqdm import tqdm
from babylon_pgm.constants import LabelType, Gender, BOOLEAN_STATES
from babylon_pgm.triage_models.triage_decisions import TriageDecisionUs, TriageModelDecisionUs

from clinical_validation_sandbox.models import CaseCard
from .triage_models_sandbox import (
    TriageData,
    TriageDataset,
    TriageBandit,
)
from .triage_models_sandbox.triage_data_transformers.base_data_extractor import BaseDataExtractor

from .inference.engine import InferenceEngine, FeatureFlags
from .utils.inference_model_from_json import JSONDatasetToInferenceModelConverter

CASE_CARDS_PATH = 'data/raw'
CASE_CARDS_FILE = '20220110173226-US-Cards.json'
MODEL_NAME = "us-deep-south-to-great-lakes-dolphin-8.20"
MODELS_PATH = Path.home() / 'dev/git/diagnostic-engine/data/models/'


def train():
    case_cards = []
    for case_card in json.load(open(Path(f'{CASE_CARDS_PATH}/{CASE_CARDS_FILE}'))):
        card = CaseCard.from_card_with_outcomes_dict(case_card)
        if card.id == 6049:
            # Card 6049 has a UK triage attached to a US doctor outcome
            # so skip adding this
            continue

        card.judgements = [j for j in card.judgements if j.region == 'us']
        card.doctor_outcomes = [o for o in card.doctor_outcomes if o.region == 'us']

        case_cards.append(card)

    # dumb split for the moment
    n = len(case_cards)
    n_test = n // 5
    test_case_cards = case_cards[-n_test:]
    case_cards = case_cards[:-n_test]

    # Get differentials
    model_generator = JSONDatasetToInferenceModelConverter(MODELS_PATH, MODEL_NAME)

    differentials_per_case = []
    test_differentials_per_case = []
    failed_train_count = 0
    failed_test_count = 0
    for card in tqdm(case_cards, desc="Running inference on train case cards", leave=True):
        model = model_generator.get_model(age=card.age, sex=Gender(card.sex))
        inference_engine = InferenceEngine(model=model, feature_flags=FeatureFlags())
        try:
            differentials_per_case.append(
                inference_engine.analyse(
                    evidence={
                        item.evidence.node.id: item.evidence.state for item in card.evidence
                    },
                )
            )
        except:
            failed_train_count += 1

    for card in tqdm(
            test_case_cards, desc="Running inference on test case cards", leave=True
    ):
        model_data = model_generator.get_model(age=card.age, sex=Gender(card.sex))
        inference_engine = InferenceEngine(model=model_data, feature_flags=FeatureFlags())
        try:
            test_differentials_per_case.append(
                inference_engine.analyse(
                    evidence={
                        item.evidence.node.id: item.evidence.state for item in card.evidence
                    },
                )
            )
        except:
            print("Failed to run inference on test case card")
            failed_test_count += 1

    train_dataset = []
    for c, d in zip(case_cards, differentials_per_case):
        try:
            datapoint = TriageData.from_case_card(card=c, differentials=d)
        except AttributeError as e:
            print(f"Error with card in: {c.id}", e)
        train_dataset.append(datapoint)

    # Filter data with no judgements or outcomes
    train_dataset = [d for d in train_dataset if (d.doctor_outcomes or d.judgements)]
    train_triage_dataset = TriageDataset(
        data=train_dataset
    )

    test_dataset = []
    for c, d in zip(test_case_cards, test_differentials_per_case):
        try:
            datapoint = TriageData.from_case_card(card=c, differentials=d)
        except AttributeError as e:
            print(f"Error with card in: {c.id}", e)
        test_dataset.append(datapoint)

    # Filter data with no judgements or outcomes
    test_dataset = [d for d in test_dataset if (d.doctor_outcomes or d.judgements)]
    test_triage_dataset = TriageDataset(
        data=test_dataset
    )

    # bandit
    bandit = TriageBandit(triage_decisions=TriageDecisionUs,
                          preprocessing_pipeline=[
                              ('data_extraction', BaseDataExtractor(triage_decisions=TriageModelDecisionUs))])
    bandit.fit(data=train_triage_dataset, num_epochs=100)
    bandit_metrics = bandit.score(
        data=test_triage_dataset, use="judgements", how="by_doctor"
    )
    for name, metrics in {
        "Bandit": bandit_metrics,
    }.items():
        print(name)
        print(f"Appropriateness: {metrics.appropriateness.score}")
        print(f"Safety: {metrics.safety.score}")
        print(f"Undertriage: {metrics.undertriage.score}")
        print(f"Overtriage: {metrics.overtriage.score}")
        print("")


if __name__ == "__main__":
    # data = load_samples("data/features", "train")
    train()
