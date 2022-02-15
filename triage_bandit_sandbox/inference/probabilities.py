def Prob_Disease_Given_RiskFactorTrue_Ratio(
    prob_riskfactor_true: float, relative_risk: float
) -> float:
    val = 1.0 / (prob_riskfactor_true + (1 - prob_riskfactor_true) / relative_risk)
    assert val > 1.0
    return val


def Prob_Disease_Given_RiskFactorFalse_Ratio(
    prob_riskfactor_true: float, relative_risk: float
) -> float:
    val = 1.0 / (relative_risk * prob_riskfactor_true + (1.0 - prob_riskfactor_true))
    assert val < 1.0
    return val
