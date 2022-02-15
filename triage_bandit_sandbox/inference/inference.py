import logging

import numpy as np
from babylon_pgm.constants import ABSENT, DOSI, DOSI_NODE_ID, PRESENT

from .constants import LEAK_NODE

logger = logging.getLogger(__name__)


def calculate_disease_posteriors(
    *,
    symptom_evidence,
    dosi,
    disease_priors,
    lambdas,
    dosi_marginals,
    single_disease_query=False,
):

    """
    Steps:

    Get lambdas for evidence, calculating over concept groups
    Perform inference
    """

    # ( P(D_k = T | R), ...) ordered by diseases list
    priors = []
    for disease_id, disease_prior in disease_priors.items():
        priors.append(disease_prior)
    prior_vec = np.array(priors)

    # the disease priors we need are the zero state (scalar),
    # all single-disease-on states (vector),
    # and the (upper triangular) matrix of all joint disease states

    zero_state = np.prod(1 - prior_vec)
    skew_vec = prior_vec / (1 - prior_vec)
    one_states = zero_state * skew_vec
    two_states = zero_state * np.triu(np.outer(skew_vec, skew_vec), k=1)

    # Now, the lambda factors.
    # lambda factor for no diseases on (scalar)
    zero_lambda = 1
    # lambda factor for one disease on (vector)
    one_lambdas = np.ones(len(disease_priors))
    # lambda factor for two diseases on (matrix)
    two_lambdas = np.triu(np.ones((len(disease_priors), len(disease_priors))), k=1)

    for symptom_id, state in symptom_evidence.items():
        if symptom_id not in lambdas:
            continue

        symptom_lambdas = lambdas[symptom_id]
        lambda0 = 1 - symptom_lambdas[LEAK_NODE]

        if state == PRESENT:
            lambda_vec = np.array(
                [symptom_lambdas.get(disease_id, 1.0) for disease_id in disease_priors]
            )

            zero_lambda *= 1 - lambda0
            one_lambdas *= 1 - lambda0 * lambda_vec
            two_lambdas *= np.outer(1 - lambda0 * lambda_vec, 1 - lambda0 * lambda_vec)

        elif state == ABSENT:
            lambda_vec = np.array(
                [symptom_lambdas.get(disease_id, 1.0) for disease_id in disease_priors]
            )
            zero_lambda *= lambda0
            one_lambdas *= lambda0 * lambda_vec
            two_lambdas *= lambda0 * np.outer(lambda_vec, lambda_vec)

    # now include dosi states
    if dosi:
        dosi_index = list(DOSI).index(dosi)
        dosi_values = np.zeros((len(disease_priors), len(DOSI)))
        for i, disease_id in enumerate(disease_priors):
            dosi_values[i] = dosi_marginals[disease_id]
        dosi_cum_values = np.cumsum(dosi_values, axis=1)

        if dosi_index > 0:
            one_lambdas *= (
                dosi_cum_values[:, dosi_index] - dosi_cum_values[:, dosi_index - 1]
            )
            two_lambdas *= np.outer(
                dosi_cum_values[:, dosi_index], dosi_cum_values[:, dosi_index]
            ) - np.outer(
                dosi_cum_values[:, dosi_index - 1], dosi_cum_values[:, dosi_index - 1]
            )
        else:
            one_lambdas *= dosi_cum_values[:, dosi_index]

    zero_marg = zero_state * zero_lambda
    one_marg = one_states * one_lambdas
    two_marg = two_states * two_lambdas

    # There is no need to normalize, as the numerator and denominator are normalized by the same amount,
    # i.e. zero_state + sum(one_states) + sum(two_states)

    if single_disease_query:
        logger.info("Using single disease query")
        two_marg = two_marg * 0.0

    denominator = zero_marg + np.sum(one_marg) + np.sum(two_marg)

    numerator = np.array(
        [
            one_marg[num] + sum(two_marg[num, :]) + sum(two_marg[:, num])
            for num in range(len(disease_priors))
        ]
    )

    posteriors = numerator / denominator

    output_posteriors = {}
    for disease_id, posterior in zip(disease_priors, posteriors):
        output_posteriors[disease_id] = posterior

    return output_posteriors
