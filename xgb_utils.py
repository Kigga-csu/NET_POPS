import json
import logging

import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV


def _build_xgb_estimator(loss_type, random_seed, n_jobs, tree_method):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for model_type=xgboost. Please install it in the current environment."
        ) from exc

    if loss_type == "mse":
        objective = "reg:squarederror"
    elif loss_type == "huber":
        objective = "reg:pseudohubererror"
    else:
        raise ValueError(f"XGBoost branch only supports loss_type in {{mse, huber}}, got: {loss_type}")

    return XGBRegressor(
        objective=objective,
        random_state=random_seed,
        n_jobs=n_jobs,
        tree_method=tree_method,
        eval_metric="rmse",
        verbosity=0,
    )


def _default_xgb_search_space():
    return {
        "n_estimators": [100, 200, 400, 800],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
        "min_child_weight": [1, 2, 5, 10],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0, 10.0],
    }


def _fit_single_xgb(model, x_all_np, y_full, train_mask, val_mask):
    model.fit(x_all_np[train_mask], y_full[train_mask])
    pred = model.predict(x_all_np)
    train_loss = mean_squared_error(y_full[train_mask], pred[train_mask])
    val_loss = mean_squared_error(y_full[val_mask], pred[val_mask]) if val_mask.any() else train_loss
    return model, pred, train_loss, val_loss


def train_xgboost_model(
    x_all_np,
    y_full,
    train_mask,
    val_mask,
    loss_type,
    random_seed=42,
    n_jobs=1,
    tree_method="hist",
    auto_tune=False,
    search_iter=20,
):
    base_model = _build_xgb_estimator(
        loss_type=loss_type,
        random_seed=random_seed,
        n_jobs=n_jobs,
        tree_method=tree_method,
    )

    history = []
    if not auto_tune:
        model, pred, train_loss, val_loss = _fit_single_xgb(base_model, x_all_np, y_full, train_mask, val_mask)
        history.append(
            {
                "iteration": 0,
                "train_mse": float(train_loss),
                "val_mse": float(val_loss),
                "params": json.dumps(model.get_params(), sort_keys=True),
            }
        )
        metadata = {
            "BEST_MONITOR_LOSS": float(val_loss),
            "XGB_AUTO_TUNE": False,
            "XGB_BEST_PARAMS_JSON": json.dumps(model.get_params(), sort_keys=True),
        }
        return model, pred, metadata, history

    test_fold = np.full(x_all_np.shape[0], -1, dtype=int)
    test_fold[val_mask] = 0
    trainable = train_mask | val_mask
    ps = PredefinedSplit(test_fold=test_fold[trainable])
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=_default_xgb_search_space(),
        n_iter=search_iter,
        scoring=scorer,
        n_jobs=1,
        cv=ps,
        verbose=0,
        random_state=random_seed,
        refit=False,
        return_train_score=True,
    )
    search.fit(x_all_np[trainable], y_full[trainable])
    results = search.cv_results_
    for idx, params in enumerate(results["params"]):
        history.append(
            {
                "iteration": idx,
                "train_mse": float(-results["mean_train_score"][idx]),
                "val_mse": float(-results["mean_test_score"][idx]),
                "rank": int(results["rank_test_score"][idx]),
                "params": json.dumps(params, sort_keys=True),
            }
        )
    best_params = search.best_params_
    logging.info("XGBoost auto-tune best params: %s", json.dumps(best_params, sort_keys=True))

    best_model = _build_xgb_estimator(
        loss_type=loss_type,
        random_seed=random_seed,
        n_jobs=n_jobs,
        tree_method=tree_method,
    )
    best_model.set_params(**best_params)
    best_model, pred, train_loss, val_loss = _fit_single_xgb(best_model, x_all_np, y_full, train_mask, val_mask)
    metadata = {
        "BEST_MONITOR_LOSS": float(val_loss),
        "XGB_AUTO_TUNE": True,
        "XGB_SEARCH_ITER": int(search_iter),
        "XGB_BEST_PARAMS_JSON": json.dumps(best_params, sort_keys=True),
    }
    return best_model, pred, metadata, history
