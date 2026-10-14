import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error
    from catboost import CatBoostRegressor
    from catboost import Pool

    import shap

    return (
        CatBoostRegressor,
        KFold,
        Pool,
        mean_absolute_error,
        mo,
        np,
        pd,
        plt,
        shap,
    )


@app.cell
def _(pd):
    df_train = pd.read_parquet("processed/train_model_feats_df.parquet")
    return (df_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model training

    We train a CatBoost model and compare to a median baseline, using five-fold cross-validation with fixed splits for a maximally stable performance estimate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the starting feature set, we remove the following raw features from the input data:

    * delivery coordinates (as this is superseded by delivery distance, we assume direction is not important)
    * date (as we assume that day of week captures any relevant calendar-specific information in a more general way)
    """)
    return


@app.cell
def _():
    train_cols = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Time_Orderd",
        "Weatherconditions",
        "Road_traffic_density",
        "Vehicle_condition",
        "Type_of_vehicle",
        "multiple_deliveries",
        "Festival",
        "City",
        "City_code",
        "Restaurant_ID",
        "Time_to_prep",
        "Delivery_distance",
        "Day_of_week",
        "Type_of_order"
    ]
    return (train_cols,)


@app.cell
def _(train_cols):
    def prepare_data(df, *, is_test_set=False):
        all_cols = train_cols.copy()
        if not is_test_set:
            all_cols += ["time_taken_min"]

        df_feats = df[all_cols].copy()

        for _col in df_feats.select_dtypes(include="category"):
            df_feats[_col] = df_feats[_col].astype("string").fillna("__missing__")

        if "Time_Orderd" in df_feats:
            df_feats["Time_Orderd"] = df_feats["Time_Orderd"].dt.total_seconds() / 60  # minutes since mindnight

        if "Vehicle_condition" in df_feats:
            # treat as categorical as unsure of meaning
            df_feats["Vehicle_condition"] = df_feats["Vehicle_condition"].astype("string").fillna("__missing__")

        return df_feats

    return (prepare_data,)


@app.cell
def _(df_train, prepare_data):
    df_train_feats = prepare_data(df_train)
    return (df_train_feats,)


@app.cell
def _(df_train_feats):
    df_train_feats
    return


@app.cell
def _(KFold):
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    return (cv,)


@app.cell
def _(df_train_feats, train_cols):
    X = df_train_feats[train_cols].copy()
    y = df_train_feats["time_taken_min"].astype(float)
    return X, y


@app.cell
def _(X, cv, mean_absolute_error, np, y):
    baseline_scores = []

    for train_idx, val_idx in cv.split(X):
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        prediction = y_train.median()

        baseline_scores.append(
            mean_absolute_error(
                y_val,
                [prediction] * len(y_val),
            )
        )

    print(f"Median baseline MAE: {np.mean(baseline_scores):.3f}")
    return


@app.cell
def _(CatBoostRegressor):
    def prepare_catboost_model(X, iterations=3000):
        model = CatBoostRegressor(
            iterations=iterations,
            depth=6,
            learning_rate=0.05,
            loss_function="MAE",
            eval_metric="MAE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )

        cat_features = X.select_dtypes(include=["string", "object"]).columns.tolist()

        return model, cat_features


    return (prepare_catboost_model,)


@app.cell
def _(mean_absolute_error, np, prepare_catboost_model):
    def cross_validate_catboost(X, y, cv):
        scores = []
        oof_predictions = np.full(len(y), np.nan)
        models = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X), start=1):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model, cat_features = prepare_catboost_model(X)

            model.fit(
                X_train,
                y_train,
                cat_features=cat_features,
                eval_set=(X_val, y_val),
                early_stopping_rounds=100,
            )

            predictions = model.predict(X_val)

            mae = mean_absolute_error(y_val, predictions)
            scores.append(mae)

            oof_predictions[val_idx] = predictions
            models.append(model)

            print(
                f"Fold {fold}: "
                f"MAE = {mae:.3f}, "
                f"best iteration = {model.best_iteration_}"
            )

        print(
            f"\nCV MAE: {np.mean(scores):.3f} "
            f"± {np.std(scores):.3f}"
        )

        return scores, oof_predictions, models

    return (cross_validate_catboost,)


@app.cell
def _(X, cross_validate_catboost, cv, y):
    scores, oof_predictions, models = cross_validate_catboost(
        X, y, cv
    )
    return models, oof_predictions, scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sanity check results using residuals
    """)
    return


@app.cell
def _(oof_predictions, plt, y):
    plt.figure(figsize=(6, 6))
    plt.scatter(y, oof_predictions, alpha=0.1, s=5)

    limits = [
        min(y.min(), oof_predictions.min()),
        max(y.max(), oof_predictions.max()),
    ]

    plt.plot(limits, limits, "r--")

    plt.xlabel("Actual delivery time (minutes)")
    plt.ylabel("Predicted delivery time (minutes)")
    plt.title("Actual vs. predicted (OOF)")
    plt.axis("equal")
    plt.show()
    return


@app.cell
def _(mean_absolute_error, np, oof_predictions, scores, y):
    print(f"OOF MAE: {mean_absolute_error(y, oof_predictions):.3f}")
    print(f"CV MAE:  {np.mean(scores):.3f}")
    return


@app.cell
def _(np, oof_predictions, plt, y):
    # +ve -> underprediction, -ve -> overprediction
    residuals = y.to_numpy() - oof_predictions

    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=60)
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residual (minutes)")
    plt.ylabel("Count")
    plt.title("Distribution of out-of-fold residuals")
    plt.show()

    print(f"Mean residual:   {residuals.mean():.3f}")
    print(f"Median residual: {np.median(residuals):.3f}")
    print(f"Residual std:    {residuals.std():.3f}")
    return (residuals,)


@app.cell
def _(oof_predictions, plt, residuals):
    plt.figure(figsize=(8, 4))
    plt.scatter(oof_predictions, residuals, alpha=0.1, s=5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted delivery time (minutes)")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Out-of-fold residuals vs. predictions")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check the banding pattern is a result of discrete target values in the data rather than a model failure by plotting for a single value of the target
    """)
    return


@app.cell
def _(oof_predictions, plt, residuals, y):
    mask = y == 30

    plt.scatter(
        oof_predictions[mask],
        residuals[mask],
        alpha=0.3,
        s=10,
    )

    plt.xlabel("Predicted delivery time")
    plt.ylabel("Residual")
    plt.title("Residuals for actual delivery time = 30 min")
    plt.show()
    return


@app.cell
def _(oof_predictions, pd, residuals):
    diagnostics = pd.DataFrame({
        "predicted": oof_predictions,
        "residual": residuals,
    })

    diagnostics["prediction_bin"] = pd.qcut(
        diagnostics["predicted"],
        q=10,
        duplicates="drop",
    )

    diagnostics.groupby(
        "prediction_bin", observed=True
    )["residual"].agg(
        count="size",
        bias="mean",
        mae=lambda x: x.abs().mean(),
    )
    return (diagnostics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check whether residuals vary systematically with predicted duration
    """)
    return


@app.cell
def _(diagnostics, plt):
    diagnostics.groupby(
        "prediction_bin", observed=True
    )["residual"].agg(["mean", "median"]).plot.bar(
        figsize=(10, 4)
    )

    plt.axhline(0, color="red", linestyle="--")
    plt.ylabel("Residual (minutes)")
    plt.title("Residual bias by predicted delivery time")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Interesting. It looks like longer delivery times tend to suffer from underprediction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check feature importances
    """)
    return


@app.cell
def _(X, models, pd, plt):
    importances = pd.DataFrame(
        {
            f"fold_{i+1}": model.get_feature_importance()
            for i, model in enumerate(models)
        },
        index=X.columns,
    )

    mean_importance = importances.mean(axis=1).sort_values()

    mean_importance.plot.barh(figsize=(9, 6))

    plt.xlabel("Mean feature importance (%)")
    plt.title("CatBoost feature importance across CV folds")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ablations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Geographic features

    Looking at the feature importances, geographic features (`Restaurant_ID`, `Restaurant_latitude`, `Restaurant_longitude`, `City_code`) appear to have very little importance. What happens if we remove these features and retrain?
    """)
    return


@app.cell
def _(X, cross_validate_catboost, cv, y):
    location_features = [
        "City_code",
        "Restaurant_ID",
        "Restaurant_latitude",
        "Restaurant_longitude",
    ]

    X_no_location = X.drop(columns=location_features)

    scores_no_location, _, _ = cross_validate_catboost(
        X_no_location, y, cv
    )
    return X_no_location, scores_no_location


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This result is marginally better on average than the previous one, bringing the overall CV MAE from:

    3.169 ± 0.037 -> 3.155 ± 0.027

    Therefore, take this as the **new best model** for further ablations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Other low importance features

    `Time_to_prep`, `Day_of_week`, `Type_of_vehicle` appear to contribute little to the model.

    `Time_to_prep` being unimportant would be consistent with delivery time being independent of prep time, although this is not definitive evidence for this. Nonetheless, irrespective of reason, being able to omit it would be useful in production where actual prep time would not be available immediately after an order has been placed.

    `Day_of_week` perhaps is already captured in `Road_traffic_density` (and this also might apply to `Time_Ordered` - we can try removing that next)

    `Type_of_order` makes sense, as looking at its valid values (snack, meal, drinks, buffet - all 10k each) this appears to be a non-informative field.


    `Type_of_vehicle` is a little stranger on the surface, but again looking at its values (motorcycle, scooter, electric scooter) there isn't much actual variation here.

    Let's try removing them and re-training to see if the performance drops.
    """)
    return


@app.cell
def _(X_no_location, cross_validate_catboost, cv, y):
    low_importance_features = [
        "Day_of_week",
        "Time_to_prep",
        "Type_of_vehicle",
        "Type_of_order"
    ]

    X_reduced = X_no_location.drop(
        columns=low_importance_features
    )

    scores_reduced, _, _ = cross_validate_catboost(
        X_reduced, y, cv
    )
    return X_reduced, scores_reduced


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's also try removing `Time_Ordered`:
    """)
    return


@app.cell
def _(X_reduced, cross_validate_catboost, cv, np, scores_reduced, y):
    X_no_time = X_reduced.drop(columns=["Time_Orderd"])

    scores_no_time, oof_no_time, models_no_time = cross_validate_catboost(
        X_no_time, y, cv
    )

    print(
        f"Change in MAE: "
        f"{np.mean(scores_no_time) - np.mean(scores_reduced):+.4f}"
    )
    return X_no_time, models_no_time, oof_no_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Taking both as a set, we now have:

    | Model | Features | CV MAE |
    |---|---:|---:|
    | Full model | 18 | 3.169 |
    | No geographic features | 14 | 3.155 |
    | Further reduced | 10 | 3.144 |
    | Without `Time_Orderd` | 9 | **3.142** |

    We observed a further small improvement after removing the unimportant features.

    - The geographic features, `Day_of_week` and `Time_Orderd` appear to provide little additional predictive information in these experiments. This could be due to feature redundancy (e.g. traffic density capturing signal that we'd expect restaurant location and day / time ordered to be proxies for), although this is just a hypothesis.
    - `Time_to_prep` contributing little would make sense if preparation and delivery durations were largely independent, although we haven't established that this is the explanation.
    - `Type_of_vehicle` is a little more counter-intuitive for real data, but looking at the possible options there actually isn't that much variation (motorcycle, scooter, electric scooter, plus bicycles which were removed during data cleaning).

    > Reducing the feature set from 18 to 9 features had negligible impact on predictive performance (five-fold CV MAE: 3.169 → 3.142 minutes), indicating that most predictive information is concentrated in a relatively small subset of features.

    *The redundancy hypothesis for the time features is checked after submission in Appendix B at the end of this notebook.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### How much does traffic density contribute to MAE?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If traffic captures information that geographic features might otherwise supply, does removing traffic make geography useful?
    """)
    return


@app.cell
def _(X, X_no_location, cross_validate_catboost, cv, y):
    traffic_feature = ["Road_traffic_density"]

    X_no_traffic = X.drop(columns=traffic_feature)
    X_no_location_or_traffic = X_no_location.drop(
        columns=traffic_feature
    )

    scores_no_traffic, _, _ = cross_validate_catboost(
        X_no_traffic, y, cv
    )

    scores_no_location_or_traffic, _, _ = cross_validate_catboost(
        X_no_location_or_traffic, y, cv
    )
    return scores_no_location_or_traffic, scores_no_traffic


@app.cell
def _(
    np,
    pd,
    scores,
    scores_no_location,
    scores_no_location_or_traffic,
    scores_no_traffic,
):
    pd.DataFrame({
        "With traffic": {
            "With location": np.mean(scores),
            "Without location": np.mean(scores_no_location),
        },
        "Without traffic": {
            "With location": np.mean(scores_no_traffic),
            "Without location": np.mean(scores_no_location_or_traffic),
        },
    }).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Observations:

    1) Road traffic provides useful information, but it's contribution is surprisingly modest:

    * 5.6 seconds when geographic features are retained
    * 4.5 seconds when geographic features are removed

    (around 2-3% deterioration in predictive performance)

    2) Geographic features actually perform worse when road traffic is absent

    * With geographic features: 3.262
    * Without geographic features: 3.230

    Removing geographic features improves MAE by 0.032 minutes, or approximately 1.9 seconds.

    It suggests that the additional geographic features aren't providing a useful substitute for observed traffic density.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sliced Error Analysis of best model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The overall MAE could hide variation in performance across different sub-populations. Let's investigate the actual behaviour of the model by slicing over traffic and distance (the two most important features).
    """)
    return


@app.cell
def _(X_no_time, models_no_time, oof_no_time):
    X_best = X_no_time
    oof_best = oof_no_time
    models_best = models_no_time
    return X_best, models_best, oof_best


@app.cell
def _(pd):
    def error_by_slice(group, y, predictions):
        df = pd.DataFrame({
            "group": group.to_numpy(),
            "error": y.to_numpy() - predictions
        })

        return (
            df.groupby("group", dropna=False)["error"]
            .agg(
                count="size",
                mae=lambda x: x.abs().mean(),
                bias="mean"
            )
            .round(3)
        )

    return (error_by_slice,)


@app.cell
def _(X_best, error_by_slice, oof_best, y):
    traffic_order = ["Low", "Medium", "High", "Jam", "__missing__"]
    error_by_slice(X_best["Road_traffic_density"], y, oof_best).reindex(traffic_order)
    return


@app.cell
def _(X_best, error_by_slice, np, oof_no_time, pd, y):
    distance_bins = pd.cut(
        X_best["Delivery_distance"].astype(float),
        bins=[0, 5, 10, 15, np.inf],
        labels=["0–5 km", "5–10 km", "10–15 km", ">15 km"],
        include_lowest=True,
    ).astype("string").fillna("Missing")

    error_by_slice(
        distance_bins,
        y,
        oof_no_time,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Combined plot
    """)
    return


@app.cell
def _(error_by_slice, mean_absolute_error, np, pd, plt):
    def plot_combined(X_best, oof_best, y):
        traffic_order = [
            "Low", "Medium", "High", "Jam", "__missing__"
        ]

        distance_order = [
            "0–5 km", "5–10 km", "10–15 km", ">15 km", "Missing"
        ]

        distance_bins = pd.cut(
            X_best["Delivery_distance"].astype(float),
            bins=[0, 5, 10, 15, np.inf],
            labels=["0–5 km", "5–10 km", "10–15 km", ">15 km"],
            include_lowest=True,
        ).astype("string").fillna("Missing")
        overall_mae = mean_absolute_error(y, oof_best)

        traffic_stats = error_by_slice(
            X_best["Road_traffic_density"], y, oof_best
        ).reindex(traffic_order)

        distance_stats = error_by_slice(
            distance_bins, y, oof_best
        ).reindex(distance_order)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        for ax, stats, title in [
            (axes[0], traffic_stats, "Error by traffic density"),
            (axes[1], distance_stats, "Error by delivery distance"),
        ]:
            bars = ax.bar(stats.index, stats["mae"])

            ax.axhline(
                overall_mae,
                color="red",
                linestyle="--",
                label="Overall MAE",
            )

            for bar, count in zip(bars, stats["count"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"n={count:,}",
                    ha="center",
                    fontsize=8,
                )

            ax.set_title(title)
            ax.set_ylabel("MAE (minutes)")
            ax.tick_params(axis="x", rotation=30)
            ax.legend()

        plt.tight_layout()
        plt.show()

    return (plot_combined,)


@app.cell
def _(X_best, oof_best, plot_combined, y):
    plot_combined(X_best, oof_best, y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Verify missing traffic density
    """)
    return


@app.cell
def _(X_best):
    X_best[X_best["Road_traffic_density"] == "__missing__"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It looks like for those cases with missing `Road_traffic_density`, many other fields are also missing (e.g. _always_ delivery person age and ratings) and so it makes sense why the MAE would be incredibly high for these cases
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Dig into distances

    The increased error > 10km also aligns with the discontinuity we see in the SHAP values and underlying distribution (see section below)
    """)
    return


@app.cell
def _(mean_absolute_error, np, pd, plt):
    def plot_distances_deep_dive(X_best, oof_best, y):
        residual = y.to_numpy() - oof_best

        df_distance = pd.DataFrame({
            "distance": X_best["Delivery_distance"].to_numpy(),
            "error": residual,
        })

        df_distance["distance_bin"] = pd.cut(
            df_distance["distance"],
            bins=np.arange(0, 24, 2),
        )

        stats = (
            df_distance
            .groupby("distance_bin", observed=True)
            ["error"]
            .agg(
                count="size",
                mae=lambda x: x.abs().mean(),
                bias="mean",
            )
        )

        midpoints = [interval.mid for interval in stats.index]

        plt.figure(figsize=(9, 4))

        plt.plot(midpoints, stats["mae"], marker="o", label="MAE")
        plt.plot(midpoints, stats["bias"], marker="o", label="Bias")

        plt.axvline(10, color="red", linestyle="--", label="10 km threshold")
        plt.axhline(mean_absolute_error(y, oof_best), color="gray", linestyle=":", label="Overall MAE")

        plt.xlabel("Delivery distance (km)")
        plt.ylabel("Error (minutes)")
        plt.title("Prediction error as a function of delivery distance")
        plt.legend()
        plt.show()

    return (plot_distances_deep_dive,)


@app.cell
def _(X_best, oof_best, plot_distances_deep_dive, y):
    plot_distances_deep_dive(X_best, oof_best, y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SHAP Analysis of base model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have reviewed the model's overall behaviour, let's use SHAP analysis to understand the model's learnt relationships better. Unlike the feature importance, this looks at the specific contribution in minutes that different features make to the final prediction.
    """)
    return


@app.cell
def _(Pool, np):
    def calculate_shap_values(X, models, cv):
        cat_features = X.select_dtypes(
            include=["string", "object"]
        ).columns.tolist()

        shap_values = np.zeros((len(X), X.shape[1]))  # (n_samples, n_features)

        for fold, (_, _val_idx) in enumerate(cv.split(X)):
            X_val = X.iloc[_val_idx]

            values = models[fold].get_feature_importance(
                Pool(X_val, cat_features=cat_features),
                type="ShapValues",
            )

            # Last column contains the baseline prediction
            shap_values[_val_idx] = values[:, :-1]

        return shap_values

    return (calculate_shap_values,)


@app.cell
def _(np, shap):
    def prepare_shap_values(X, shap_values):
        X_plot = X.copy()

        for col in X_plot.select_dtypes(include="number").columns:
            X_plot[col] = X_plot[col].to_numpy(
                dtype="float64",
                na_value=np.nan,
            )

        explanation = shap.Explanation(
            values=shap_values,
            data=X_plot.to_numpy(),
            feature_names=X.columns.tolist(),
        )

        return explanation

    return (prepare_shap_values,)


@app.cell
def _(X, calculate_shap_values, cv, models, prepare_shap_values):
    shap_values = calculate_shap_values(X, models, cv)
    explanation = prepare_shap_values(X, shap_values)
    return explanation, shap_values


@app.cell
def _(explanation, shap):
    shap.plots.beeswarm(explanation)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It looks like `Delivery_person_Age`, `Delivery_distance` and `Delivery_person_Ratings` all have an interesting bimodal distribution. Let's investigate by plotting each observation's SHAP contribution against its feature value:
    """)
    return


@app.cell
def _(explanation, shap):
    for _feature in [
        "Delivery_distance",
        "Delivery_person_Age",
        "Delivery_person_Ratings"
    ]:
        shap.plots.scatter(explanation[:, _feature])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Investigate whether sharp thresholds exist in original data distribution
    """)
    return


@app.cell
def _(df_train_feats, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, _feature, threshold in [
        (axes[0], "Delivery_person_Age", 30),
        (axes[1], "Delivery_person_Ratings", 4.5),
    ]:
        stats = (
            df_train_feats.groupby(_feature)["time_taken_min"]
            .agg(["mean", "count"])
            .sort_index()
        )

        ax.plot(stats.index, stats["mean"], marker=".")
        ax.axvline(threshold, color="red", linestyle="--")
        ax.set_xlabel(_feature)
        ax.set_ylabel("Mean delivery time (minutes)")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The discontinuities also exist in the observed targets, so they are not solely artefacts of the fitted model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Let's also check distance

    Given the findings in the sliced error analysis section
    """)
    return


@app.cell
def _(np, pd, plt, y):
    def check_distance_discontinuity(X_best):
        distance = X_best["Delivery_distance"].astype(float)

        distance_bins = pd.cut(
            distance,
            bins=np.arange(0, 23, 1),
        )

        stats = (
            pd.DataFrame({
                "distance_bin": distance_bins,
                "target": y,
            })
            .groupby("distance_bin", observed=True)["target"]
            .agg(["mean", "std", "count"])
        )

        stats["midpoint"] = [interval.mid for interval in stats.index]
        stats["ci95"] = 1.96 * stats["std"] / np.sqrt(stats["count"])

        # Avoid displaying unreliable means for sparsely populated bins.
        stats = stats[stats["count"] >= 30]

        plt.figure(figsize=(9, 4))

        plt.errorbar(
            stats["midpoint"],
            stats["mean"],
            yerr=stats["ci95"],
            fmt="o-",
            capsize=2,
        )

        plt.axvline(10, color="red", linestyle="--")

        plt.xlabel("Delivery distance (km)")
        plt.ylabel("Mean delivery time (minutes)")
        plt.title("Observed delivery time vs. delivery distance")
        plt.grid(axis="y", alpha=0.2)
        plt.show()

    return (check_distance_discontinuity,)


@app.cell
def _(X_best, check_distance_discontinuity):
    check_distance_discontinuity(X_best)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Prepare a pub ready figure for the report
    """)
    return


@app.cell
def _(np, pd, plt):
    from pathlib import Path


    def age_rating_delivery_time_discontinuities_pub_ready(df_train_feats):

        def grouped_target_stats(df, feature):
            stats = (
                df.groupby(feature, observed=True)["time_taken_min"]
                .agg(["mean", "std", "count"])
                .sort_index()
            )

            stats["ci95"] = (
                1.96 * stats["std"] / np.sqrt(stats["count"])
            )

            return stats

        # --- Age and ratings ---

        age_stats = grouped_target_stats(
            df_train_feats, "Delivery_person_Age"
        )

        rating_stats = grouped_target_stats(
            df_train_feats, "Delivery_person_Ratings"
        )

        # --- Distance: 1-km bins ---

        distance_bins = pd.cut(
            df_train_feats["Delivery_distance"].astype(float),
            bins=np.arange(0, 23, 1),
            include_lowest=True,
        )

        distance_stats = grouped_target_stats(
            df_train_feats.assign(distance_bin=distance_bins),
            "distance_bin",
        )

        distance_stats["midpoint"] = [
            interval.mid for interval in distance_stats.index
        ]

        # Exclude sparsely populated bins
        distance_stats_plot = distance_stats[
            distance_stats["count"] >= 30
        ]

        # --- Calculate observed discontinuities ---

        age_jump = (
            age_stats.loc[30, "mean"]
            - age_stats.loc[29, "mean"]
        )

        rating_jump = (
            rating_stats.loc[4.5, "mean"]
            - rating_stats.loc[4.4, "mean"]
        )

        distance_jump = (
            distance_stats.loc[pd.Interval(10, 11), "mean"]
            - distance_stats.loc[pd.Interval(9, 10), "mean"]
        )

        # --- Figure ---

        fig, axes = plt.subplots(
            1, 3,
            figsize=(9, 3.0), # (12, 3.5),
            sharey=True,
            layout="constrained",
        )

        panels = [
            {
                "ax": axes[0],
                "x": age_stats.index.to_numpy(dtype=float),
                "stats": age_stats,
                "xlabel": "Courier age (years)",
                "title": "(a) Courier age",
                "threshold": 29.5,
                "annotation": f"29 → 30: {age_jump:+.1f} min",
            },
            {
                "ax": axes[1],
                "x": rating_stats.index.to_numpy(dtype=float),
                "stats": rating_stats,
                "xlabel": "Courier rating",
                "title": "(b) Courier rating",
                "threshold": 4.45,
                "annotation": f"4.4 → 4.5: {rating_jump:+.1f} min",
            },
            {
                "ax": axes[2],
                "x": distance_stats_plot["midpoint"].to_numpy(),
                "stats": distance_stats_plot,
                "xlabel": "Delivery distance (km)",
                "title": "(c) Delivery distance",
                "threshold": 10,
                "annotation": f"9–10 → 10–11 km: {distance_jump:+.1f} min",
            },
        ]

        for panel in panels:
            ax = panel["ax"]
            stats = panel["stats"]

            ax.errorbar(
                panel["x"],
                stats["mean"].to_numpy(),
                yerr=stats["ci95"].to_numpy(),
                fmt="o-",
                markersize=3.5,
                linewidth=1.4,
                elinewidth=0.8,
                capsize=2,
            )

            ax.axvline(
                panel["threshold"],
                color="0.45",
                linestyle="--",
                linewidth=1,
            )

            ax.set_title(panel["title"], fontsize=11)
            ax.set_xlabel(panel["xlabel"], fontsize=10)

            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.tick_params(labelsize=9)

            ax.text(
                0.04, 0.96,
                panel["annotation"],
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                    pad=2,
                ),
            )

        axes[0].set_ylabel(
            "Mean delivery time (min)",
            fontsize=10,
        )

        # Consistent scales across panels
        axes[0].set_ylim(20, 40)

        axes[0].set_xlim(19.5, 39.5)
        axes[1].set_xlim(2.35, 5.05)
        axes[2].set_xlim(0.5, 21.5)

        # --- Export ---

        Path("figures").mkdir(exist_ok=True)

        fig.savefig(
            "figures/delivery_time_discontinuities.pdf",
            bbox_inches="tight",
        )

        fig.savefig(
            "figures/delivery_time_discontinuities.png",
            dpi=300,
            bbox_inches="tight",
        )

        # --- Report supporting statistics ---

        print("Age threshold:")
        print(age_stats.loc[[29, 30], ["mean", "count", "ci95"]])

        print("\nRating threshold:")
        print(rating_stats.loc[[4.4, 4.5], ["mean", "count", "ci95"]])

        print("\nDistance threshold:")
        print(
            distance_stats.loc[
                [pd.Interval(9, 10), pd.Interval(10, 11)],
                ["mean", "count", "ci95"],
            ]
        )

        plt.show()

        return age_stats, rating_stats, distance_stats

    return (age_rating_delivery_time_discontinuities_pub_ready,)


@app.cell
def _(age_rating_delivery_time_discontinuities_pub_ready, df_train_feats):
    _, _, _ = age_rating_delivery_time_discontinuities_pub_ready(df_train_feats)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Visualise the categorical SHAP contributions
    """)
    return


@app.cell
def _(pd, plt):
    def plot_categorical_feature_contributions(X, shap_values, feature_name):
        idx = X.columns.get_loc(feature_name)

        df_plot = pd.DataFrame({
            "category": X[feature_name].astype(str),
            "shap": shap_values[:, idx],
        })

        order = (
            df_plot.groupby("category")["shap"]
            .median()
            .sort_values()
            .index
        )

        plt.figure(figsize=(9, 4))

        plt.boxplot(
            [
                df_plot.loc[df_plot["category"] == c, "shap"]
                for c in order
            ],
            tick_labels=order,
            showfliers=False,
        )

        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel(feature_name)
        plt.ylabel("SHAP contribution (minutes)")
        plt.show()

    return (plot_categorical_feature_contributions,)


@app.cell
def _(X, plot_categorical_feature_contributions, shap_values):
    plot_categorical_feature_contributions(X, shap_values, "Road_traffic_density")
    plot_categorical_feature_contributions(X, shap_values, "Weatherconditions")
    plot_categorical_feature_contributions(X, shap_values, "Vehicle_condition")
    plot_categorical_feature_contributions(X, shap_values, "Type_of_vehicle")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SHAP Analysis of best model
    """)
    return


@app.cell
def _(
    X_best,
    calculate_shap_values,
    cv,
    models_best,
    prepare_shap_values,
    shap,
):
    shap_values_best = calculate_shap_values(X_best, models_best, cv)
    explanation_best = prepare_shap_values(X_best, shap_values_best)
    shap.plots.beeswarm(explanation_best)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The plots are broadly similar: reducing the number of features doesn't materially change the model's behaviour
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final model training
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Re-train best model on all data
    """)
    return


@app.cell
def _(np, prepare_catboost_model):
    def train_final_model(X, models, y):
        best_iterations = [
            model.best_iteration_ + 1
            for model in models
        ]

        final_iterations = int(np.median(best_iterations))

        print(f"Final iterations: {final_iterations}")

        model, cat_features = prepare_catboost_model(X, iterations=final_iterations)

        model.fit(
            X,
            y,
            cat_features=cat_features,
        )

        return model

    return (train_final_model,)


@app.cell
def _(X_best, models_best, train_final_model, y):
    final_model = train_final_model(X_best, models_best, y)
    return (final_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run over test data
    """)
    return


@app.cell
def _(pd):
    df_test = pd.read_parquet("processed/test_model_feats_df.parquet")
    return (df_test,)


@app.cell
def _(df_test, prepare_data):
    df_test_feats = prepare_data(df_test, is_test_set=True)
    return (df_test_feats,)


@app.cell
def _(X_best, df_test_feats, final_model):
    X_test = df_test_feats[X_best.columns].copy()
    assert X_test.columns.tolist() == final_model.feature_names_
    return (X_test,)


@app.cell
def _(X_test, final_model):
    predictions = final_model.predict(X_test)
    return (predictions,)


@app.cell
def _(pd, predictions, y):
    # sanity check distributions
    pd.DataFrame({
        "Train targets": y.describe(),
        "Test predictions": pd.Series(predictions).describe(),
    })
    return


@app.cell
def _(X_best, X_test, df_test, np, pd, predictions):
    raw_test_ids = pd.read_csv("dataset/test.csv")["ID"]
    sample = pd.read_csv("dataset/Sample_Submission.csv")

    # Preserve literal identifiers while checking positional prediction alignment.
    assert raw_test_ids.str.strip().tolist() == df_test["ID"].tolist()
    assert len(predictions) == len(raw_test_ids)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()
    assert X_test.columns.tolist() == X_best.columns.tolist()
    assert X_test.dtypes.equals(X_best.dtypes)

    submission = pd.DataFrame({
        "ID": raw_test_ids.to_numpy(),
        "Time_taken (min)": predictions,
    })

    assert len(submission) == len(df_test)
    assert submission["ID"].is_unique

    assert submission.columns.tolist() == sample.columns.tolist()
    assert submission["ID"].tolist() == sample["ID"].tolist()

    submission.to_csv("processed/submission.csv", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix A: (post-submission) recovering delivery distance for (0, 0) restaurants
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Was masking the delivery distance too conservative?

    During exploration we found that ~8% of rows have restaurant coordinates of (0, 0), with delivery coordinates set to small fixed values such as (0.03, 0.03) or (0.11, 0.11). We treated these as missing location data and masked all four coordinate fields, which leaves `Delivery_distance` missing for these rows. The sliced error analysis above shows this slice has markedly higher MAE than any other.

    Two other observations from exploration suggest this may have been too conservative. The paired delivery values in the (0, 0) rows look like a fixed offset from the (missing) restaurant location, and the distribution of `Delivery_distance` for the remaining rows showed equally sized peaks at regularly spaced values. If the same regular offset structure applies to the rows with valid restaurant coordinates, then the offset carries the distance information for *all* rows, and the (0, 0) rows should not have been discarded just because the origin is missing: the offset is intact, only the origin is unknown.

    We check this in three steps: (1) whether every row has the same offset structure, (2) whether the regularly spaced distance peaks are exactly these offsets, and (3) whether the target depends on the offset in the same way for the (0, 0) rows as for the rest.
    """)
    return


@app.cell
def _(X_best, pd):
    # Raw, pre-masking coordinates aligned to the modelling rows by index
    zr_raw = pd.read_parquet("processed/train_formatted_df.parquet").loc[
        X_best.index,
        [
            "Restaurant_latitude",
            "Restaurant_longitude",
            "Delivery_location_latitude",
            "Delivery_location_longitude",
        ],
    ]

    # Sign-corrected restaurant coordinates, as in explore.py
    zr_raw["Restaurant_latitude"] = zr_raw["Restaurant_latitude"].abs()
    zr_raw["Restaurant_longitude"] = zr_raw["Restaurant_longitude"].abs()

    zr_is_zero = (
        (zr_raw["Restaurant_latitude"] == 0)
        & (zr_raw["Restaurant_longitude"] == 0)
    )

    # The rows with missing distance are exactly the (0, 0) rows
    assert (zr_is_zero == X_best["Delivery_distance"].isna()).all()

    _dlat = zr_raw["Delivery_location_latitude"] - zr_raw["Restaurant_latitude"]
    _dlon = zr_raw["Delivery_location_longitude"] - zr_raw["Restaurant_longitude"]

    # Tolerances cover the 6 d.p. rounding of coordinates in the CSV
    _equal_offsets = (_dlat - _dlon).abs().lt(1e-3)
    zr_offset_k = (_dlat * 100).round()
    _on_grid = (_dlat * 100 - zr_offset_k).abs().lt(1e-2)

    print(f"Rows with equal lat / lon offsets:       {_equal_offsets.mean():.1%}")
    print(f"Rows with offsets on a 0.01 degree grid: {_on_grid.mean():.1%}")

    assert _equal_offsets.all() and _on_grid.all()

    zr_offset_k = zr_offset_k.astype(int)

    pd.crosstab(
        zr_offset_k.rename("Offset k (× 0.01°)"),
        zr_is_zero.map(
            {True: "(0, 0) restaurant rows", False: "Rows with coordinates"}
        ).rename("Population"),
    )
    return zr_is_zero, zr_offset_k, zr_raw


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The offset structure holds for every row, not only the (0, 0) rows. Both populations are drawn from the same set of twelve offsets, `k ∈ {1–9, 11, 13, 14}`, in roughly equal proportions (`k = 14` is a small group that happens not to occur in the (0, 0) rows).
    """)
    return


@app.cell
def _(X_best, np, zr_is_zero, zr_offset_k, zr_raw):
    def _haversine_km(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = (np.radians(_c) for _c in (lat1, lon1, lat2, lon2))
        _a = (
            np.sin((lat2 - lat1) / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
        _a = np.clip(_a, 0, 1)
        return 6371.0088 * 2 * np.arctan2(np.sqrt(_a), np.sqrt(1 - _a))

    # Haversine on the raw coordinates for every row, i.e. measured from (0, 0)
    # for the masked rows
    zr_distance_unmasked = _haversine_km(
        zr_raw["Restaurant_latitude"],
        zr_raw["Restaurant_longitude"],
        zr_raw["Delivery_location_latitude"],
        zr_raw["Delivery_location_longitude"],
    )

    # Where the distance was not masked, the recomputation must match explore.py
    assert np.allclose(
        zr_distance_unmasked[~zr_is_zero],
        X_best.loc[~zr_is_zero, "Delivery_distance"],
    )

    # Distance clusters for the rows with coordinates, one per offset value
    zr_clusters = (
        X_best.loc[~zr_is_zero, "Delivery_distance"]
        .groupby(zr_offset_k[~zr_is_zero])
        .agg(["count", "min", "median", "max"])
    )

    # The same offset measured from (0, 0), as used for the masked rows below
    zr_clusters["from (0, 0)"] = _haversine_km(
        0, 0, zr_clusters.index * 0.01, zr_clusters.index * 0.01
    )

    # Clusters must not overlap, and for each k present in the masked rows the
    # (0, 0)-origin value must fall below the next cluster's minimum
    _next_min = zr_clusters["min"].shift(-1)
    assert (zr_clusters["max"] < _next_min.fillna(np.inf)).all()

    _k_masked = np.sort(zr_offset_k[zr_is_zero].unique())
    assert (
        zr_clusters.loc[_k_masked, "from (0, 0)"] < _next_min.loc[_k_masked]
    ).all()

    zr_clusters.round(2)
    return (zr_distance_unmasked,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The twelve regularly spaced peaks in the `Delivery_distance` distribution are exactly the twelve offset values. Within each `k` the spread (e.g. 4.40–4.68 km for `k = 3`) comes only from the restaurant latitude, which enters through `cos(φ)` on the longitude term across the dataset's 10°–31° range, and the clusters never overlap. Measured from (0, 0) instead of the true origin, the same offset lands just above its cluster's maximum and below the next cluster's minimum, so an origin-free distance for the masked rows would still be separable from its neighbours by a tree model.
    """)
    return


@app.cell
def _(np, plt, y, zr_is_zero, zr_offset_k):
    _fig, _ax = plt.subplots(figsize=(8, 4))

    for _mask, _label in [
        (~zr_is_zero, "Rows with coordinates"),
        (zr_is_zero, "(0, 0) restaurant rows"),
    ]:
        _stats = y[_mask].groupby(zr_offset_k[_mask]).agg(["mean", "std", "count"])

        _ax.errorbar(
            _stats.index,
            _stats["mean"],
            yerr=1.96 * _stats["std"] / np.sqrt(_stats["count"]),
            fmt="o-",
            capsize=2,
            label=_label,
        )

    _ax.set_xlabel("Offset k (× 0.01°)")
    _ax.set_ylabel("Mean delivery time (min)")
    _ax.set_title("Observed delivery time by offset, with and without restaurant coordinates")
    _ax.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The target depends on the offset in the same way in both populations, including the step of around +6 minutes at `k ≥ 7` (the 10 km discontinuity identified above). The (0, 0) rows therefore carry the full distance signal; what they lack is only the origin needed to express it in kilometres.

    Before recovering it from the offset, check whether the origin itself can be looked up from other orders at the same restaurant:
    """)
    return


@app.cell
def _(df_train, zr_is_zero):
    _zero_share = zr_is_zero.groupby(df_train["Restaurant_ID"]).mean()
    _affected = _zero_share[_zero_share > 0]

    print(f"Restaurants with any (0, 0) rows: {len(_affected)}")
    print(f"...of which (0, 0) in every row:  {(_affected == 1).sum()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Recovering the distance

    The affected rows belong to 50 restaurants that are (0, 0) in every one of their orders, so the origin cannot be imputed from the data. The offset, however, needs no origin. We therefore test two ways of recovering the distance signal, keeping everything else identical to the final nine-feature model (same features, hyperparameters and CV folds):

    * **Variant B**: use the haversine distance computed on the raw coordinates *without* masking, i.e. measured from (0, 0) for the affected rows. As shown above, this places each masked row just above its true cluster and below the next one.
    * **Variant C**: replace `Delivery_distance` by the offset `k` itself for all rows, which needs no origin at all.
    """)
    return


@app.cell
def _(
    X_best,
    cross_validate_catboost,
    cv,
    y,
    zr_distance_unmasked,
    zr_offset_k,
):
    X_zr_haversine = X_best.assign(Delivery_distance=zr_distance_unmasked)
    X_zr_offset = X_best.assign(Delivery_distance=zr_offset_k.astype(float))

    print("Variant B: haversine with (0, 0) origin for the masked rows")
    scores_zr_haversine, oof_zr_haversine, _ = cross_validate_catboost(
        X_zr_haversine, y, cv
    )

    print("\nVariant C: offset k in place of distance")
    scores_zr_offset, oof_zr_offset, _ = cross_validate_catboost(
        X_zr_offset, y, cv
    )
    return oof_zr_haversine, oof_zr_offset


@app.cell
def _(
    error_by_slice,
    mean_absolute_error,
    np,
    oof_best,
    oof_zr_haversine,
    oof_zr_offset,
    pd,
    y,
    zr_is_zero,
):
    _slice = pd.Series(
        np.where(zr_is_zero, "(0, 0) restaurant rows", "Rows with coordinates"),
        index=y.index,
    )

    _variants = {
        "A: distance masked (final model)": oof_best,
        "B: haversine, (0, 0) origin": oof_zr_haversine,
        "C: offset k": oof_zr_offset,
    }

    zr_results = pd.concat(
        {
            _name: error_by_slice(_slice, y, _oof)["mae"]
            for _name, _oof in _variants.items()
        },
        axis=1,
    ).T

    zr_results.insert(
        0,
        "Overall",
        [mean_absolute_error(y, _oof) for _oof in _variants.values()],
    )

    zr_results.columns.name = "OOF MAE (min)"
    zr_results.round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Results from the recorded run (five-fold CV, same folds as above):

    | Variant | Overall | (0, 0) restaurant rows (8%) | Rows with coordinates (92%) |
    |---|---:|---:|---:|
    | A: distance masked (final model) | 3.143 | 3.877 | 3.080 |
    | B: haversine, (0, 0) origin | 3.070 | 3.021 | 3.075 |
    | C: offset `k` | **3.057** | 2.983 | 3.063 |

    Observations:

    * Recovering the distance for the (0, 0) rows brings their MAE down from 3.9 to 3.0 minutes, in line with the rest of the data, and the overall OOF MAE from 3.14 to 3.06–3.07. This is larger than the entire range covered by the feature ablations above (3.169 → 3.142). Rows with valid coordinates are unaffected, as expected.
    * B and C are within fold-to-fold noise of each other (~0.03), so the equator-origin approximation in B costs nothing a tree model cares about. The gain comes from not discarding the 8%, not from the form of the feature.
    * The equivalent fix on real data, where the origin would matter, is to impute the restaurant location from master data rather than from other orders.
    * The final model and submission above are left unchanged; this section records a check carried out after submission.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix B: (post-submission) why dropping the time features cost nothing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The ablations above removed `Time_Orderd` and `Day_of_week` with no loss in CV MAE, and we hypothesised that `Road_traffic_density` was already capturing whatever they contribute. This appendix checks that hypothesis directly.

    Two kinds of relationship are possible, and worth keeping distinct: an **exact mapping**, where one field can be written down from another, and a **shift in distribution**, where one field changes the mix of another without determining it.
    """)
    return


@app.cell
def _(df_train, pd):
    _ok = df_train["Time_Orderd"].notna() & df_train["Road_traffic_density"].notna()

    tf_minute = (
        df_train.loc[_ok, "Time_Orderd"].dt.total_seconds() // 60
    ).astype(int)
    tf_traffic = df_train.loc[_ok, "Road_traffic_density"].astype(str)

    pd.crosstab(
        (tf_minute // 60).rename("Order hour"),
        tf_traffic.rename("Road_traffic_density"),
    )[["Low", "High", "Medium", "Jam"]]
    return tf_minute, tf_traffic


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Traffic falls into bands of the day with almost no overlap, and the few mixed hours are the ones containing a boundary. Is the mapping exact at the resolution the data provides, and does it hold outside the training set?
    """)
    return


@app.cell
def _(df_test, pd, tf_minute, tf_traffic):
    # One traffic class per distinct order minute?
    _classes_per_minute = tf_traffic.groupby(tf_minute).nunique()
    assert (_classes_per_minute == 1).all()

    tf_mapping = tf_traffic.groupby(tf_minute).first()

    # Does the minute -> traffic mapping learnt from train reproduce the test set?
    _ok_test = (
        df_test["Time_Orderd"].notna() & df_test["Road_traffic_density"].notna()
    )
    _test_minute = (
        df_test.loc[_ok_test, "Time_Orderd"].dt.total_seconds() // 60
    ).astype(int)
    _test_traffic = df_test.loc[_ok_test, "Road_traffic_density"].astype(str)
    _test_match = tf_mapping.reindex(_test_minute).to_numpy() == _test_traffic.to_numpy()
    assert _test_match.all()

    print(
        f"Distinct order minutes: {len(tf_mapping)}; "
        f"traffic classes per minute: {_classes_per_minute.max()}"
    )
    print(f"Test rows reproduced by the train mapping: {_test_match.mean():.1%}")

    # Contiguous bands of the mapping
    _bands = []
    for _m, _c in tf_mapping.items():
        if _bands and _bands[-1][2] == _c:
            _bands[-1][1] = _m
        else:
            _bands.append([_m, _m, _c])

    pd.DataFrame(
        [
            {
                "From": f"{_s // 60:02d}:{_s % 60:02d}",
                "To": f"{_e // 60:02d}:{_e % 60:02d}",
                "Road_traffic_density": _c,
                "Orders": int(tf_minute.between(_s, _e).sum()),
            }
            for _s, _e, _c in _bands
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Traffic is an **exact mapping** from order time: five bands of the clock, one class per minute, no exceptions in training, and the same mapping reproduces every test row. `Time_Orderd` therefore carries nothing *about traffic* that the traffic class doesn't already say.

    Order time could still matter through something else. Within a traffic band, does delivery time vary with the hour, and if so, what carries it?
    """)
    return


@app.cell
def _(df_train, pd, tf_minute, tf_traffic, y):
    tf_hour = (tf_minute // 60).rename("Order hour")
    _rows = tf_hour.index
    _band_order = ["Low", "High", "Medium", "Jam"]

    tf_within_band = (
        pd.DataFrame(
            {
                "y": y.loc[_rows],
                "distance": df_train.loc[_rows, "Delivery_distance"],
            }
        )
        .groupby([tf_traffic.rename("Road_traffic_density"), tf_hour])
        .agg(
            Orders=("y", "size"),
            **{
                "Mean delivery time (min)": ("y", "mean"),
                "Mean delivery distance (km)": ("distance", "mean"),
            },
        )
        .round(1)
    )

    # Hide the slivers of orders either side of a band boundary
    tf_within_band = tf_within_band[tf_within_band["Orders"] >= 300]

    tf_within_band.reindex(_band_order, level=0)
    return (tf_hour,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Within every band, delivery time still varies with the hour by 3–5 minutes (Low: 19.6 at 08–10 against 22.4 at 22–23; Medium: 22.9 at 15–16 against 27.4 at 17–18), so the clock carries something beyond the traffic class. The same table shows what: delivery distance shifts with the hour, from around 2 km in the morning to 6 km around midday and 12 km in the evening. This is a **shift in distribution**, not a mapping: any given hour contains a spread of distances. Since the model keeps the actual distance, the hour's guess about it is superseded.

    To check that nothing is left, group orders by the five step effects identified above (traffic band, the 10 km distance threshold, number of deliveries, courier age at 30 and rating at 4.5), subtract each group's mean, and look at what the hour still explains in the residual. The other four features are left out to keep the groups populated; any imbalance in them across hours shows up here as hour effect, so this is an upper bound on what remains after the full model.
    """)
    return


@app.cell
def _(df_train, np, pd, tf_hour, tf_traffic, y):
    _rows = tf_hour.index

    def _threshold(series, value):
        _s = series.astype(float)
        return np.where(
            _s.isna(), "missing", np.where(_s >= value, f">= {value}", f"< {value}")
        )

    # Group by the five step effects identified in the SHAP analysis above
    tf_structure = pd.DataFrame(
        {
            "traffic": tf_traffic,
            "distance": _threshold(df_train.loc[_rows, "Delivery_distance"], 10),
            "multiple": df_train.loc[_rows, "multiple_deliveries"].astype(str),
            "age": _threshold(df_train.loc[_rows, "Delivery_person_Age"], 30),
            "rating": _threshold(df_train.loc[_rows, "Delivery_person_Ratings"], 4.5),
        },
        index=_rows,
    )

    tf_residual = y.loc[_rows] - y.loc[_rows].groupby(
        [tf_structure[_c] for _c in tf_structure]
    ).transform("mean")

    _raw = y.loc[_rows].groupby(tf_hour).mean()
    _res = tf_residual.groupby(tf_hour).mean()

    print(f"Spread of mean delivery time across hours, raw:       {_raw.max() - _raw.min():.1f} min")
    print(f"Spread of mean residual across hours, after control:  {_res.max() - _res.min():.1f} min")
    print(f"Residual noise remaining (std):                        {tf_residual.std():.1f} min")

    pd.DataFrame(
        {"Raw mean (min)": _raw, "Residual after control (min)": _res}
    ).round(2)
    return tf_residual, tf_structure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After controlling for the retained structure the hour explains nothing: the spread across hours drops from about 12 minutes to about half a minute, which is noise at these sample sizes. Time of day matters only through traffic (an exact mapping) and distance (a shift in distribution), and the model keeps both.

    `Day_of_week` is the same question. Its raw means differ by up to 2.5 minutes between days, which looks like signal, but the traffic and distance mix also differs between days:
    """)
    return


@app.cell
def _(df_train, pd, tf_hour, tf_residual, tf_structure, y):
    _rows = tf_hour.index
    _dow = df_train.loc[_rows, "Day_of_week"].rename("Day_of_week")
    _names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    tf_by_day = pd.DataFrame(
        {
            "Orders": _dow.value_counts(),
            "Raw mean delivery time (min)": y.loc[_rows].groupby(_dow).mean(),
            "Share in Jam traffic": (tf_structure["traffic"] == "Jam").groupby(_dow).mean(),
            "Share over 10 km": (tf_structure["distance"] == ">= 10").groupby(_dow).mean(),
            "Residual after control (min)": tf_residual.groupby(_dow).mean(),
        }
    ).round(2)

    tf_by_day.index = [_names[int(_d)] for _d in tf_by_day.index]
    tf_by_day
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The day-of-week differences are composition: Wednesday has twice Thursday's share of long deliveries and a third more Jam-hour orders, and once the retained structure is controlled for, the day explains nothing (residuals within ±0.1 min).

    Time and day of order are simply different ways of slicing the same population into groups with different mixes of traffic and distance. Once an order's actual traffic and distance are known, the slice it came from is irrelevant, which is why removing both features left the CV MAE unchanged. This does not make delivery time predictable from traffic and distance alone: around 6 minutes of spread remains after controlling for them and the courier thresholds.

    In a real operation these would be three distinct signals: the order time is known when the order is placed, the traffic is a later and noisier observation, and the distance is independent of the clock. The redundancy here is a property of this dataset, not a reason to drop time features in general.

    The final model and submission above are left unchanged; this section records a check carried out after submission.
    """)
    return


if __name__ == "__main__":
    app.run()
