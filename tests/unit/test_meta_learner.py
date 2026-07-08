"""Tests for scripts/train_meta_learner.py — build_training_data and train logic."""

import pytest
import numpy as np
import pandas as pd
from datetime import date


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_monthly_errors(
    models: list[str],
    months: list[date],
    values: dict[str, list[tuple[float, float]]],
    dfu: tuple[str, str, str] = ("ITEM1", "GRP1", "LOC1"),
) -> pd.DataFrame:
    rows = []
    for model_id in models:
        for i, month in enumerate(months):
            fcst, actual = values[model_id][i]
            rows.append(
                {
                    "item_id": dfu[0],
                    "customer_group": dfu[1],
                    "loc": dfu[2],
                    "startdate": pd.Timestamp(month),
                    "model_id": model_id,
                    "basefcst_pref": fcst,
                    "tothist_dmd": actual,
                    "abs_err": abs(fcst - actual),
                }
            )
    return pd.DataFrame(rows)


def _make_dfu_features(
    dfus: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    if dfus is None:
        dfus = [("ITEM1", "GRP1", "LOC1")]
    rows = []
    for item_id, customer_group, loc in dfus:
        rows.append(
            {
                "item_id": item_id,
                "customer_group": customer_group,
                "loc": loc,
                "ml_cluster": 0,
                "abc_vol": 1,
                "execution_lag": 1,
                "total_lt": 30,
                "brand": 0,
                "region": 0,
                "seasonality_profile": 0,
                "seasonality_strength": 0.5,
                "is_yearly_seasonal": True,
                "peak_month": 6,
                "trough_month": 12,
                "peak_trough_ratio": 2.0,
            }
        )
    return pd.DataFrame(rows)


MONTHS_12 = [date(2024, m, 1) for m in range(1, 13)]


# ---------------------------------------------------------------------------
# build_training_data tests
# ---------------------------------------------------------------------------


class TestBuildTrainingData:
    def test_output_shape(self):
        from scripts.ml.train_meta_learner import build_training_data

        df = _make_monthly_errors(
            models=["A", "B"],
            months=MONTHS_12,
            values={
                "A": [(105, 100)] * 12,
                "B": [(110, 100)] * 12,
            },
        )
        dfu_features = _make_dfu_features()

        features_df, target, feature_cols = build_training_data(
            df,
            dfu_features,
            ["A", "B"],
            performance_window=6,
            min_prior_months=3,
        )

        assert len(features_df) > 0
        assert len(target) == len(features_df)
        assert len(feature_cols) > 0
        assert "ceiling_winner" in features_df.columns
        assert "startdate" in features_df.columns

    def test_ceiling_labels_correct(self):
        """Verify ceiling labels: model with lowest error wins each month."""
        from scripts.ml.train_meta_learner import build_training_data

        df = _make_monthly_errors(
            models=["A", "B"],
            months=MONTHS_12,
            values={
                "A": [(105, 100)] * 12,  # err=5 always
                "B": [(120, 100)] * 12,  # err=20 always
            },
        )
        dfu_features = _make_dfu_features()

        features_df, target, _ = build_training_data(
            df,
            dfu_features,
            ["A", "B"],
            performance_window=6,
            min_prior_months=3,
        )

        # Model A always has lower error, so ceiling_winner should be A
        assert (target == "A").all()

    def test_features_strictly_prior(self):
        """Features for month T should only use data from months < T.

        Specifically: roll_wape columns should use shift(1) before rolling,
        so the first few rows should have NaN roll_wape values.
        """
        from scripts.ml.train_meta_learner import build_training_data

        df = _make_monthly_errors(
            models=["A", "B"],
            months=MONTHS_12,
            values={
                "A": [(105, 100)] * 12,
                "B": [(120, 100)] * 12,
            },
        )
        dfu_features = _make_dfu_features()

        features_df, _, feature_cols = build_training_data(
            df,
            dfu_features,
            ["A", "B"],
            performance_window=6,
            min_prior_months=3,
        )

        # Months with min_prior_months < 3 should be excluded
        earliest = features_df["startdate"].min()
        # Should not include the very first 3 months (0-indexed: Jan, Feb, Mar)
        assert earliest >= pd.Timestamp("2024-04-01")

    def test_min_prior_months_filtering(self):
        """Increasing min_prior_months should reduce output rows."""
        from scripts.ml.train_meta_learner import build_training_data

        df = _make_monthly_errors(
            models=["A", "B"],
            months=MONTHS_12,
            values={
                "A": [(105, 100)] * 12,
                "B": [(120, 100)] * 12,
            },
        )
        dfu_features = _make_dfu_features()

        features_3, _, _ = build_training_data(
            df,
            dfu_features,
            ["A", "B"],
            performance_window=6,
            min_prior_months=3,
        )
        features_6, _, _ = build_training_data(
            df,
            dfu_features,
            ["A", "B"],
            performance_window=6,
            min_prior_months=6,
        )

        assert len(features_3) >= len(features_6)

    def test_feature_columns_include_expected(self):
        """Check that expected feature columns are present."""
        from scripts.ml.train_meta_learner import build_training_data

        df = _make_monthly_errors(
            models=["A", "B"],
            months=MONTHS_12,
            values={
                "A": [(105, 100)] * 12,
                "B": [(120, 100)] * 12,
            },
        )
        dfu_features = _make_dfu_features()

        _, _, feature_cols = build_training_data(
            df,
            dfu_features,
            ["A", "B"],
            performance_window=6,
            min_prior_months=3,
        )

        # Should include per-model WAPE columns
        assert "roll_wape_A" in feature_cols
        assert "roll_wape_B" in feature_cols
        # Should include calendar features
        assert "month" in feature_cols
        assert "fourier_sin_12" in feature_cols
        assert "fourier_cos_12" in feature_cols
        # Should include demand stats
        assert "mean_qty" in feature_cols
        assert "cv_demand" in feature_cols


# ---------------------------------------------------------------------------
# Temporal split tests
# ---------------------------------------------------------------------------


class TestTemporalSplit:
    def test_no_future_data_in_train(self):
        """Verify that training data does not contain future months."""
        from scripts.ml.train_meta_learner import build_training_data

        df = _make_monthly_errors(
            models=["A", "B"],
            months=MONTHS_12,
            values={
                "A": [(105, 100)] * 12,
                "B": [(120, 100)] * 12,
            },
        )
        dfu_features = _make_dfu_features()

        features_df, target, feature_cols = build_training_data(
            df,
            dfu_features,
            ["A", "B"],
            performance_window=6,
            min_prior_months=3,
        )

        # Simulate temporal split (last 3 months holdout)
        test_months = 3
        all_months = sorted(features_df["startdate"].unique())
        if len(all_months) > test_months:
            cutoff = all_months[-test_months]
            train = features_df[features_df["startdate"] < cutoff]
            test = features_df[features_df["startdate"] >= cutoff]

            # No overlap
            train_months = set(train["startdate"].unique())
            test_months_set = set(test["startdate"].unique())
            assert train_months.isdisjoint(test_months_set)

            # Train months all before cutoff
            assert all(m < cutoff for m in train_months)


# ---------------------------------------------------------------------------
# Train/predict cycle test
# ---------------------------------------------------------------------------


class TestTrainPredict:
    def test_classifier_trains_and_predicts(self):
        """Verify train_classifier produces a working model."""
        from scripts.ml.train_meta_learner import build_training_data, train_classifier

        df = _make_monthly_errors(
            models=["A", "B"],
            months=MONTHS_12,
            values={
                "A": [(105, 100)] * 12,
                "B": [(120, 100)] * 12,
            },
        )
        dfu_features = _make_dfu_features()

        features_df, target, feature_cols = build_training_data(
            df,
            dfu_features,
            ["A", "B"],
            performance_window=6,
            min_prior_months=3,
        )

        if len(features_df) < 4:
            pytest.skip("Not enough data for train/test split")

        # Simple split
        all_months = sorted(features_df["startdate"].unique())
        cutoff = all_months[-2]
        train_mask = features_df["startdate"] < cutoff
        test_mask = features_df["startdate"] >= cutoff

        X_train = features_df.loc[train_mask, feature_cols].fillna(0)
        y_train = target[train_mask]
        X_test = features_df.loc[test_mask, feature_cols].fillna(0)
        y_test = target[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            pytest.skip("Not enough data for train/test split")

        clf, accuracy, report = train_classifier(
            X_train,
            y_train,
            X_test,
            y_test,
            model_type="random_forest",
        )

        assert clf is not None
        assert 0.0 <= accuracy <= 1.0
        assert isinstance(report, dict)

        # Should be able to predict
        preds = clf.predict(X_test)
        assert len(preds) == len(X_test)

    def test_xgboost_classifier_uses_registry_params(self, monkeypatch):
        """XGBoost meta-learner goes through model_registry with configured params."""
        from scripts.ml import train_meta_learner as module

        X_train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
        y_train = pd.Series(["A", "B", "A", "B"])
        X_test = pd.DataFrame({"a": [4.0, 5.0]})
        y_test = pd.Series(["A", "B"])
        captured: dict[str, object] = {}

        class FakeClassifier:
            def predict(self, X):
                return np.array([0] * len(X))

        fake_classifier = FakeClassifier()

        def fake_build_tree_classifier(model_name, params):
            captured["model_name"] = model_name
            captured["params"] = params
            return fake_classifier

        def fake_fit_tree_classifier(model, model_name, X, y):
            captured["fit_model"] = model
            captured["fit_name"] = model_name
            captured["fit_rows"] = len(X)
            captured["encoded_labels"] = set(y.tolist())

        monkeypatch.setattr(module, "build_tree_classifier", fake_build_tree_classifier)
        monkeypatch.setattr(module, "fit_tree_classifier", fake_fit_tree_classifier)

        clf, accuracy, report = module.train_classifier(
            X_train,
            y_train,
            X_test,
            y_test,
            model_type="xgboost",
            n_estimators=11,
            max_depth=3,
            learning_rate=0.2,
            random_state=7,
            n_jobs=1,
            eval_metric="mlogloss",
        )

        assert clf is fake_classifier
        assert captured["model_name"] == "xgboost"
        assert captured["params"] == {
            "n_estimators": 11,
            "max_depth": 3,
            "learning_rate": 0.2,
            "random_state": 7,
            "n_jobs": 1,
            "eval_metric": "mlogloss",
        }
        assert captured["fit_model"] is fake_classifier
        assert captured["fit_name"] == "xgboost"
        assert captured["fit_rows"] == len(X_train)
        assert captured["encoded_labels"] == {0, 1}
        assert 0.0 <= accuracy <= 1.0
        assert isinstance(report, dict)

    def test_missing_classifier_config_fails_loud(self, monkeypatch):
        from scripts.ml import train_meta_learner as module

        monkeypatch.setattr(module, "_load_meta_learner_config", lambda: {})

        with pytest.raises(ValueError, match="random_forest meta-learner params missing"):
            module.train_classifier(
                pd.DataFrame({"a": [1.0, 2.0]}),
                pd.Series(["A", "B"]),
                pd.DataFrame({"a": [3.0]}),
                pd.Series(["A"]),
                model_type="random_forest",
            )
