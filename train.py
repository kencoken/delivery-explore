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

    > Reducing the feature set from 18 to 9 features had negligible impact on predictive performance (five-fold CV MAE: 3.167 → 3.142 minutes), indicating that most predictive information is concentrated in a relatively small subset of features.
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


if __name__ == "__main__":
    app.run()
