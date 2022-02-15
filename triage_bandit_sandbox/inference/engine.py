import logging
import math
from typing import Dict, List, Mapping, Tuple

from babylon_pgm.constants import (ABSENT, DOSI, DOSI_NODE_ID, PRESENT,
                                   LabelType)
from babylon_pgm.models.diagnosis import Differential

from .config import DOSIMode, FeatureFlags
from .constants import (DEFAULT_MAX_DISEASE_PRIOR, DEFAULT_MAX_SYMPTOM_PRIOR,
                        DEFAULT_SYMPTOM_LEAK_PROPORTION, LEAK_NODE)
from .inference import calculate_disease_posteriors
from .models import (EvidenceMap, InferenceModel, Priors)
from .probabilities import (Prob_Disease_Given_RiskFactorFalse_Ratio,
                            Prob_Disease_Given_RiskFactorTrue_Ratio)

logger = logging.getLogger(__name__)


def marginal_to_lambda(marginal: float) -> float:
    # lambda_ = 1 - 0.99 * marginal  - TODO: Jon was trying this
    lambda_ = marginal
    lambda_ *= 1.0 - math.exp(-1.0 * lambda_)
    lambda_ = 1.0 - lambda_
    return lambda_


def calculate_symptom_posterior(
    disease_posteriors: Dict[str, float],
    symptom_marginals: Dict[str, float],
    symptom_leak: float,
):
    # TODO: Should symptoms be explained by diseases for which we have evidence?
    p_symptom_absent = 1 - symptom_leak
    for disease_id, marginal in symptom_marginals.items():
        lambda_ = marginal_to_lambda(marginal)

        # TODO: Check this is okay to do if disease not in disease posteriors
        disease_posterior = disease_posteriors.get(disease_id, 0.0)

        # TODO: Check this
        p_symptom_absent *= lambda_ * disease_posterior + 1.0 - disease_posterior
        return 1 - p_symptom_absent


def calculate_symptom_leaks(
    model: InferenceModel,
    symptom_leak_proportion: float = DEFAULT_SYMPTOM_LEAK_PROPORTION,
    max_symptom_prior: float = DEFAULT_MAX_SYMPTOM_PRIOR,
) -> Mapping[str, float]:
    """
    Calculate symptom leak terms.
    Either using a special leak term, or based on a proportion of the symptom prior
    """

    # TODO: This will need to be done on the fly during inference when priors are dynamic
    symptom_leaks = {}

    symptoms = model.get_nodes_by_label(LabelType.symptom)
    for symptom in symptoms:
        if symptom.id == DOSI_NODE_ID:
            continue

        special_symptom_leak = model.get_special_symptom_leak(symptom.id)
        if special_symptom_leak:
            symptom_leaks[symptom.id] = 1 - special_symptom_leak
            continue

        # Approximate symptom prior by max P(S|Di)*P(Di) over disease Di..N
        symptom_prior = 0.0
        marginals = model.get_symptom_disease_marginals(symptom_id=symptom.id)
        for disease in model.get_parents(symptom.id):
            marginal = model.get_symptom_disease_marginal(
                symptom_id=symptom.id, disease_id=disease.id
            )

            assert isinstance(marginal, float), "expecting a float marginal"
            symptom_prior = max(symptom_prior, model.priors[disease.id] * marginal)

        # symptom_leak = min(symptom_prior, max_symptom_prior) * symptom_leak_proportion
        symptom_prior = min(symptom_prior, max_symptom_prior)
        symptom_leak = symptom_prior / (1 - symptom_leak_proportion)
        symptom_leak *= symptom_leak_proportion

        # TODO: redefining leak as P(L=T) not a lambda - check inference
        symptom_leaks[symptom.id] = symptom_leak

    return symptom_leaks


class InferenceEngine:
    def __init__(
        self,
        model: InferenceModel,
        feature_flags: FeatureFlags,
    ) -> None:
        self._model = model
        self._feature_flags = feature_flags

        # Pre-compute the symptom leaks
        self._symptom_leaks = calculate_symptom_leaks(
            model=self._model,
            symptom_leak_proportion=self._feature_flags.symptom_leak_proportion,
            max_symptom_prior=self._feature_flags.max_symptom_prior,
        )

    @property
    def model(self) -> InferenceModel:
        return self._model

    def analyse(self, evidence: EvidenceMap) -> List[Differential]:
        symptom_evidence, risk_factor_evidence, dosi = self._categorise_evidence(
            evidence=evidence
        )

        disease_priors = self._calculate_disease_priors(
            risk_factor_evidence=risk_factor_evidence,
            dosi=dosi,
        )

        # Get diseases connected to any present symptoms or searchable risk factors
        # TODO: Consider also restricting to presenting complaint?
        connected_diseases = set()
        for symptom_id, state in symptom_evidence.items():
            if state == PRESENT:
                connected_diseases.update(self.model.get_parents(symptom_id))

        for risk_id, state in risk_factor_evidence.items():
            risk_factor = self.model.get_node(risk_id)
            if state == PRESENT and risk_factor.is_searchable:
                connected_diseases.update(self.model.get_children(risk_id))

        # Filter priors for connected diseases (if feature flag set)
        if self._feature_flags.ignore_other_diseases:
            disease_priors = {
                disease.id: disease_priors.get(disease.id, 0.0)
                for disease in connected_diseases
            }

        if self._feature_flags.apply_concept_groups:
            raise NotImplementedError("Concept groups feature is not yet supported")

        # Calculate symptom lambdas
        lambdas: Mapping[str, Mapping[str, float]] = {}
        for symptom_id in symptom_evidence:
            symptom = self.model.get_node(symptom_id)
            symptom_lambdas = {}
            for disease in self.model.get_parents(symptom.id):
                if disease not in connected_diseases:
                    continue  # Just in case, it always should be
                marginal = self.model.get_symptom_disease_marginal(
                    symptom_id=symptom.id, disease_id=disease.id
                )
                symptom_lambdas[disease.id] = marginal_to_lambda(marginal)
            if not symptom_lambdas:
                continue
            symptom_lambdas[LEAK_NODE] = self._symptom_leaks[symptom.id]
            lambdas[symptom.id] = symptom_lambdas

        # Get DOSI marginals
        dosi_marginals = self.model.get_symptom_disease_marginals(
            symptom_id=DOSI_NODE_ID
        )

        # Calculate posteriors
        disease_posteriors = calculate_disease_posteriors(
            symptom_evidence=symptom_evidence,
            dosi=None,  # TODO: switch between dosi nodes
            disease_priors=disease_priors,
            lambdas=lambdas,
            dosi_marginals=dosi_marginals,
            single_disease_query=self._feature_flags.single_disease_query,
        )

        # Add zero posteriors for non-connected diseases
        all_posteriors = {}
        for disease in self.model.get_nodes_by_label(LabelType.disease):
            all_posteriors[disease.id] = disease_posteriors.get(disease.id, 0.0)
            # TODO: How are we handling diseases for which we have evidence?

        # Add symptom posteriors
        for symptom in self.model.get_nodes_by_label(LabelType.symptom):
            if symptom.id == DOSI_NODE_ID:
                continue
            state = evidence.get(symptom.id)
            if state == PRESENT:
                all_posteriors[symptom.id] = 1.0
            elif state == ABSENT:
                all_posteriors[symptom.id] = 0.0
            else:
                all_posteriors[symptom.id] = calculate_symptom_posterior(
                    disease_posteriors=all_posteriors,
                    symptom_marginals=self.model.get_symptom_disease_marginals(
                        symptom_id=symptom.id
                    ),
                    symptom_leak=self._symptom_leaks[symptom.id],
                )

        # TODO: Add risk factor posteriors

        return [
            Differential(
                node=self.model.get_node(node_id),
                probability=probability,
            )
            for node_id, probability in all_posteriors.items()
            # if node_id in self.model.nodes  # i.e. excludes supernodes
        ]

    def _categorise_evidence(
        self, evidence: EvidenceMap
    ) -> Tuple[EvidenceMap, EvidenceMap, DOSI]:
        """
        Split the binary evidence into symptom, risk evidence and dosi
        """
        symptom_evidence = {}
        risk_factor_evidence = {}
        dosi_state = None

        for node_id, state in evidence.items():

            node = self.model.get_node(node_id)
            if not node:
                continue

            if node.id == DOSI_NODE_ID:
                dosi_state = next((dosi for dosi in DOSI if state == dosi.value), None)
                # TODO: Check if dosi is required
                continue

            if state not in {PRESENT, ABSENT}:
                continue

            if node.label == LabelType.symptom:
                symptom_evidence[node.id] = state
            elif node.label == LabelType.risk:
                risk_factor_evidence[node.id] = state
            # TODO: Add concept group evidence
            # TODO: Add disease evidence (for VOI)

        return symptom_evidence, risk_factor_evidence, dosi_state

    def _calculate_disease_priors(
        self,
        risk_factor_evidence: EvidenceMap,
        dosi: DOSI,
    ) -> Priors:
        """
        Use the state of any risk factors in the evidence to calculate
        updated priors for the disease layer
        Note: this essentially transforms our 3-layer PGM to a 2-layer PGM
        """

        diseases = self.model.get_nodes_by_label(LabelType.disease)

        sum_of_disease_priors = sum(
            [self.model.get_prior(disease.id) for disease in diseases]
        )
        logger.info(f"Sum of disease priors: {sum_of_disease_priors}")

        updated_disease_priors = {}
        for disease in diseases:
            prior = self.model.get_prior(disease.id)

            # TODO: Assume this is just for efficiency...
            if prior == 0.0:
                continue

            """
            Make an apriori assumption that the patient has a disease by
            normalising the prior distribution - i.e. that our population of
            interest includes people who are unwell, not the general population
            """
            # TODO: Feature flag this
            prior /= sum_of_disease_priors

            for risk_factor_id, risk_factor_state in risk_factor_evidence.items():
                relative_risk = self.model.get_relative_risk(
                    risk_factor_id=risk_factor_id, disease_id=disease.id
                )
                if not relative_risk:
                    continue

                if risk_factor_state == PRESENT:
                    prior *= Prob_Disease_Given_RiskFactorTrue_Ratio(
                        self.model.get_prior(risk_factor_id), relative_risk
                    )
                elif risk_factor_state == ABSENT:
                    prior *= Prob_Disease_Given_RiskFactorFalse_Ratio(
                        self.model.get_prior(risk_factor_id), relative_risk
                    )

            # Update prior with DOSI
            if self._feature_flags.dosi_mode == DOSIMode.PRIOR_ADJUSTMENT:
                dosi_cpt = self.model.get_symptom_disease_marginals(
                    symptom_id=DOSI_NODE_ID
                )[disease.id]
                if dosi:
                    dosi_index = list(DOSI).index(dosi)
                    # TODO: Make this a method of InferenceModel
                    prior *= dosi_cpt[dosi_index]

            # TODO: check this, but the previous function was crazy
            prior = min(prior, DEFAULT_MAX_DISEASE_PRIOR)

            if not (0.0 < prior < 1.0):
                raise ValueError

            updated_disease_priors[disease.id] = prior

        return updated_disease_priors
