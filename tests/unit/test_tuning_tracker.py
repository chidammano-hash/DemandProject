"""Tests for real-run tuning breakdown persistence."""

from unittest.mock import MagicMock, patch

import pandas as pd


def test_register_lag_breakdowns_uses_real_all_lag_predictions(tmp_path) -> None:
    from common.core.constants import FORECAST_QTY_COL
    from common.ml.tuning_tracker import register_lag_breakdowns

    csv_path = tmp_path / "backtest_predictions_all_lags.csv"
    pd.DataFrame(
        [
            {"item_id": "A", "customer_group": "G", "loc": "L", "lag": 0,
             FORECAST_QTY_COL: 90.0, "tothist_dmd": 100.0},
            {"item_id": "B", "customer_group": "G", "loc": "L", "lag": 0,
             FORECAST_QTY_COL: 110.0, "tothist_dmd": 100.0},
            {"item_id": "A", "customer_group": "G", "loc": "L", "lag": 1,
             FORECAST_QTY_COL: 120.0, "tothist_dmd": 100.0},
            {"item_id": "A", "customer_group": "G", "loc": "L", "lag": 5,
             FORECAST_QTY_COL: 100.0, "tothist_dmd": 100.0},
        ]
    ).to_csv(csv_path, index=False)

    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    connection = MagicMock()
    connection.cursor.return_value = cursor
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = False

    with (
        patch("common.ml.tuning_tracker.get_db_params", return_value={}),
        patch("common.ml.tuning_tracker.psycopg.connect", return_value=connection),
    ):
        count = register_lag_breakdowns(7, csv_path)

    assert count == 2
    params = [call.args[1] for call in cursor.execute.call_args_list]
    assert params[0] == (7, 0, 2, 2, 90.0, 10.0, 0.0)
    assert params[1] == (7, 1, 1, 1, 80.0, 20.0, 0.2)
    connection.commit.assert_called_once()
