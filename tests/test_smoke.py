"""Smoke test: the README's minimal example must fit and predict end to end."""
import numpy as np
from RuleTree import RuleTreeRegressor

from ltr_utility import ModelParam
from ruletreerank import KNNRegFast, PairwiseDistanceTree, RuleTreeRank


def make_model() -> RuleTreeRank:
    return RuleTreeRank(
        distance_f=ModelParam(PairwiseDistanceTree, {
            "base_regressor": ModelParam(RuleTreeRegressor, {"max_depth": 2, "random_state": 0}),
            "feature_concat": False,
            "feature_diff": True,
            "feature_sq_diff": True,
            "subsample": 0.5,
        }),
        aggregation_f=ModelParam(KNNRegFast, {"n_neighbors": 5, "n_jobs": 1}),
        base_regressor=RuleTreeRegressor(max_depth=2, random_state=0),
        dist_objective="residuals",
    )


def test_fit_predict_two_stages():
    X = np.random.default_rng(0).normal(size=(100, 5))
    y = X @ np.array([1.4, -0.7, 0.5, 0.0, 0.2])
    q = np.repeat(np.arange(10), 10)

    model = make_model().fit(X, y, q)

    full = model.predict(X, q=q, output="full")
    score = model.predict(X, q=q, output="score")
    corr = model.predict(X, q=q, output="corr")

    assert full.shape == (100,)
    assert np.isfinite(full).all()
    # the two stages must compose into the final score
    np.testing.assert_allclose(full, score + corr)
    # stage II has to move something, otherwise the local refinement is dead
    assert np.any(corr != 0)


if __name__ == "__main__":
    test_fit_predict_two_stages()
    print("ok")
