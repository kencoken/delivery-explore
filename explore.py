import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt

    return mo, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore dataset
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("dataset/train.csv")
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Find missing values
    """)
    return


@app.cell
def _(df):
    df.isna().sum()
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(pd):
    def clean_df(df):
        cleaned_df = df.copy()
        string_cols = cleaned_df.select_dtypes(include="str").columns
        cleaned_df[string_cols] = cleaned_df[string_cols].apply(
            lambda s: s.str.strip()
        )
        cleaned_df = cleaned_df.replace(
            r".*NaN.*",
            pd.NA,
            regex=True
        )

        return cleaned_df

    return (clean_df,)


@app.cell
def _(clean_df, df):
    cleaned_df = clean_df(df)
    return (cleaned_df,)


@app.cell
def _(cleaned_df):
    cleaned_df.isna().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Convert to sensible types
    """)
    return


@app.cell
def _(pd):
    def format_df(cleaned_df, *, is_test_set=False):
        formatted_df = cleaned_df.copy()

        formatted_df["Delivery_person_Age"] = (
            pd.to_numeric(
                formatted_df["Delivery_person_Age"],
                errors="raise",
            )
            .astype("Int64")
        )

        formatted_df["Delivery_person_Ratings"] = (
            pd.to_numeric(
                formatted_df["Delivery_person_Ratings"],
                errors="raise",
            )
            .astype("Float64")
        )

        formatted_df["multiple_deliveries"] = (
            pd.to_numeric(
                formatted_df["multiple_deliveries"],
                errors="raise",
            )
            .astype("Int64")
        )


        # --- Date / time ---

        formatted_df["Order_Date"] = pd.to_datetime(
            formatted_df["Order_Date"],
            format="%d-%m-%Y",
            errors="raise",
        )

        formatted_df["Time_Orderd"] = pd.to_timedelta(
            formatted_df["Time_Orderd"],
            errors="raise",
        )

        formatted_df["Time_Order_picked"] = pd.to_timedelta(
            formatted_df["Time_Order_picked"],
            errors="raise",
        )


        # --- Target ---

        if not is_test_set:
            formatted_df["time_taken_min"] = (
                formatted_df.pop("Time_taken(min)")
                .str.removeprefix("(min)")
                .str.strip()
                .pipe(pd.to_numeric, errors="raise")
                .astype("Int64")
            )


        # --- Categories ---

        formatted_df["Weatherconditions"] = (
            formatted_df["Weatherconditions"]
            .str.removeprefix("conditions ")
        )

        categorical_cols = [
            "Weatherconditions",
            "Road_traffic_density",
            "Vehicle_condition",
            "Type_of_order",
            "Type_of_vehicle",
            "Festival",
            "City",
        ]

        formatted_df[categorical_cols] = (
            formatted_df[categorical_cols].astype("category")
        )

        return formatted_df

    return (format_df,)


@app.cell
def _(cleaned_df, format_df):
    formatted_df = format_df(cleaned_df)
    return (formatted_df,)


@app.cell
def _(formatted_df):
    formatted_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Explore value distributions
    """)
    return


@app.cell
def _(formatted_df):
    formatted_df["Delivery_person_Age"].value_counts().sort_index()
    return


@app.cell
def _(plt):
    def plot_counts(df, column):
        counts = df[column].value_counts(dropna=False).sort_index()

        ax = counts.plot.bar()

        # labels = [
        #     f"{count:,}\n({count / len(df):.1%})"
        #     for count in counts
        # ]
        # ax.bar_label(ax.containers[0], labels=labels, padding=3)

        labels = [
            f"{count:,} ({count / len(df):.1%})"
            for count in counts
        ]

        ax.bar_label(
            ax.containers[0],
            labels=labels,
            padding=3,
            rotation=90,
        )

        ax.margins(y=0.45)

        plt.xlabel(column)
        plt.ylabel("Count")
        plt.title(f"Distribution of {column}")
        plt.show()

    return (plot_counts,)


@app.cell
def _(formatted_df, plot_counts):
    plot_counts(formatted_df, "Delivery_person_Age")
    return


@app.cell
def _(formatted_df, plot_counts):
    plot_counts(formatted_df, "Vehicle_condition")
    return


@app.cell
def _(formatted_df, plot_counts):
    plot_counts(formatted_df, "Delivery_person_Ratings")
    return


@app.cell
def _(formatted_df, plot_counts):
    for col in ["Type_of_order", "Type_of_vehicle", "City", "time_taken_min", "Festival", "Type_of_order", "Road_traffic_density", "Weatherconditions"]:
        plot_counts(formatted_df, col)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Dig into bicycle vehicle type
    """)
    return


@app.cell
def _(formatted_df):
    bicycle_df = formatted_df[formatted_df["Type_of_vehicle"] == "bicycle"]
    return (bicycle_df,)


@app.cell
def _(bicycle_df):
    bicycle_df
    return


@app.cell
def _(bicycle_df):
    print(bicycle_df["Delivery_person_Age"].value_counts())
    print(bicycle_df["Delivery_person_Ratings"].value_counts())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Dig into Delivery_person_Age
    """)
    return


@app.cell
def _(formatted_df):
    strange_age_df = formatted_df[formatted_df["Delivery_person_Age"].isin([15, 50])]
    return (strange_age_df,)


@app.cell
def _(strange_age_df):
    strange_age_df
    return


@app.cell
def _(strange_age_df):
    print(strange_age_df["Delivery_person_Ratings"].value_counts())
    print(strange_age_df["Vehicle_condition"].value_counts())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Dig into vehicle_condition == 3
    """)
    return


@app.cell
def _(formatted_df):
    vehicle_cond_3_df = formatted_df[formatted_df["Vehicle_condition"] == 3]
    return (vehicle_cond_3_df,)


@app.cell
def _(vehicle_cond_3_df):
    vehicle_cond_3_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Explore `Delivery_person_ID`
    """)
    return


@app.function
def split_delivery_person_id(df):
    return df["Delivery_person_ID"].str.extract(
        r"^(?P<p>.*?)RES(?P<res>\d+)DEL(?P<del>\d+)$"
    )


@app.cell
def _(formatted_df):
    delivery_ids = split_delivery_person_id(formatted_df)
    return (delivery_ids,)


@app.cell
def _(delivery_ids):
    delivery_ids
    return


@app.cell
def _(delivery_ids):
    print(delivery_ids["p"].value_counts().to_string())
    print(delivery_ids["res"].value_counts().to_string())
    print(delivery_ids["del"].value_counts().to_string())
    return


@app.cell
def _(delivery_ids):
    delivery_ids.head()
    return


@app.cell
def _(delivery_ids, formatted_df):
    # are 'res' codes related to restaurants? no, lat / long vary
    formatted_df[delivery_ids["res"] == "13"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    no, lat / long vary
    """)
    return


@app.cell
def _(delivery_ids, formatted_df):
    # how about 'p' and 'res' pairs, as 'p' seems to relate to city and perhaps 'res' is reused between cities? no, lat / long vary
    formatted_df[(delivery_ids["p"] == "INDO") & (delivery_ids["res"] == "13")]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    yes, with this lat / long are singular, except interestingly enough sometimes the signt of lat / long are flipped
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This also suggests that `p` does relate to city and `res` relates to restaurant when combined with city
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Explore flipped lat / long
    """)
    return


@app.cell
def _(formatted_df):
    formatted_df[
        (formatted_df["Restaurant_latitude"] < 0) |
        (formatted_df["Restaurant_longitude"] < 0)
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Looks like flipped signs can just be reversed when they occur - do this cleaning
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Explore latitude / longitude
    """)
    return


@app.cell
def _(pd, plt):
    def plot_coordinates(df, coord_label, *, take_abs=False, plot_frequency=False):
        label_lat = f'{coord_label}_latitude'
        label_long = f'{coord_label}_longitude'
        s_lat = df[label_lat]
        s_long = df[label_long]

        if take_abs:
            s_lat = s_lat.abs()
            s_long = s_long.abs()

        if plot_frequency:
            coords = pd.DataFrame({
                label_lat: s_lat,
                label_long: s_long,
            })
            counts = (
                coords
                .groupby([label_lat, label_long])
                .size()
                .reset_index(name="count")
            )
    
            plot_lat = counts[label_lat]
            plot_long = counts[label_long]
            point_size = counts["count"]
        else:
            plot_lat = s_lat
            plot_long = s_long
            point_size = 10

        plt.scatter(
            plot_long,
            plot_lat,
            alpha=0.2,
            s=point_size,
        )

        plt.xlabel(f'{coord_label} Longitude')
        plt.ylabel(f'{coord_label} Latitude')
        plt.title("Locations")
        plt.show()

        zero_coords = (
            (df[f"{coord_label}_latitude"] == 0)
            & (df[f"{coord_label}_longitude"] == 0)
        )

        print(f"{zero_coords.mean() * 100.0}% of data (0, 0)")

    return (plot_coordinates,)


@app.cell
def _(formatted_df, plot_coordinates):
    # remove flipped sign locations, as we'll fix those later
    plot_coordinates(formatted_df, "Restaurant", take_abs=True)
    return


@app.cell
def _(formatted_df, plot_coordinates):
    plot_coordinates(formatted_df, "Delivery_location")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    interesting... the (0, 0) points in the restaurant field relate to the 'near zero' points in the delivery location field?
    """)
    return


@app.cell
def _(formatted_df):
    restaurant_loc_is_zero_df = formatted_df[(formatted_df["Restaurant_longitude"] == 0.0) & (formatted_df["Restaurant_latitude"] == 0.0)]

    restaurant_loc_is_not_zero_df = formatted_df[(formatted_df["Restaurant_longitude"] != 0.0) & (formatted_df["Restaurant_latitude"] != 0.0)]
    return restaurant_loc_is_not_zero_df, restaurant_loc_is_zero_df


@app.cell
def _(
    plot_coordinates,
    restaurant_loc_is_not_zero_df,
    restaurant_loc_is_zero_df,
):
    plot_coordinates(restaurant_loc_is_not_zero_df, "Delivery_location")
    plot_coordinates(restaurant_loc_is_zero_df, "Delivery_location")
    return


@app.cell
def _(plot_coordinates, restaurant_loc_is_zero_df):
    plot_coordinates(restaurant_loc_is_zero_df, "Delivery_location", plot_frequency=True)
    return


@app.cell
def _(restaurant_loc_is_zero_df):
    restaurant_loc_is_zero_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Observations so far
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * Distribution of `time_taken_min` looks suspiciously synthetically generated
    * Distribution of `type_of_order` is suspiciously balanced - either dataset was sampled like this, or again looks synthetically generated
    * Same applies to a strangly balanced `Weatherconditions`
    * `Type_of_vehicle == 'bicycle'` contains only NaN or invalid value `Delivery_person_Age` or `Delivery_person_Ratings` -> drop it in train, replace it with `NaN` in test
    * `Delivery_person_Age` values `(15, 50)` also aligns with `vehicle_condition == 3` and a `Delivery_person_Ratings` of `(1, 6)` -> mark as `NaN`
    * `Vehicle_condition == 3` aligns with `Delivery_person_age == (NaN, 15, 50)` and `Delivery_person_Ratings == (NaN, 1, 6)` and `Time_Ordered` is `NaN` -> mark as `NaN` (could also make argument for dropping it)
    * Mark `Delivery_person_Ratings == (1, 6)` as `NaN`
    * Sometimes `Restaurant_latitude` and `Restaurant_longitude` are negative - but can be flipped to produce a valid value
    * Sometimes `Restaurant_latitude / longitude` are `(0,0)` (~8% of the data). In these cases, `Delivery_location_latitude / longitude` always have a fixed paired value e.g. (0.03, 0.03) or (0.11, 0.11) which strongly suggests a synthetically generated offset. Let's therefore mark all Restaurant and Delivery location fields to NaN to indicate missing values in all rows where `Restaurant_latitude / longitude` are `(0,0)`
    * `Delivery_person_ID` doesn't actually seem to refer to a delivery person - but instead, the first two parts relate to a city + restaurant, and the final part relate to something else (could be investigated further). This means it shouldn't be treated as a delivery person identifier, but 'city' is a likely a useful feature for modelling -> extract the first part of `Delivery_person_ID` as a new column 'city_code' (to distinguish it from the existing 'City' field that actually encodes things like Urban / Metropolitan / Semi-Urban), extract the first and second part combined of `Delivery_person_ID` as `restaurant_ID` (we can include / exclude this on the basis of whether we want to learn patterns related to specific restaurants or not), and remember not to use `Delivery_person_ID` for modelling
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Apply cleanup from observations
    """)
    return


@app.cell
def _(pd):
    def prepare_model_df(formatted_df, *, is_test_set=False):
        model_df = formatted_df.copy()

        # --- Drop bicycle rows as corrupt (or mask them out in the test set) ---

        if not is_test_set:
            model_df = model_df[model_df["Type_of_vehicle"] != "bicycle"].copy()
        else:
            model_df["Type_of_vehicle"] = model_df["Type_of_vehicle"].mask(model_df["Type_of_vehicle"] == "bicycle")


        # --- Replace invalid values with missing ---

        model_df["Delivery_person_Age"] = (
            model_df["Delivery_person_Age"]
            .mask(model_df["Delivery_person_Age"].isin([15, 50]))
        )

        model_df["Delivery_person_Ratings"] = (
            model_df["Delivery_person_Ratings"]
            .mask(model_df["Delivery_person_Ratings"].isin([1, 6]))
        )

        model_df["Vehicle_condition"] = (
            model_df["Vehicle_condition"]
            .mask(model_df["Vehicle_condition"] == 3)
        )


        # --- Correct flipped restaurant coordinates ---

        model_df["Restaurant_latitude"] = (
            model_df["Restaurant_latitude"].abs()
        )

        model_df["Restaurant_longitude"] = (
            model_df["Restaurant_longitude"].abs()
        )

        # --- (0, 0) restaurant coordinates indicate missing/synthetic location data - mark these as invalid ---
        invalid_location = (
            (model_df["Restaurant_latitude"] == 0)
            & (model_df["Restaurant_longitude"] == 0)
        )

        location_cols = [
            "Restaurant_latitude",
            "Restaurant_longitude",
            "Delivery_location_latitude",
            "Delivery_location_longitude",
        ]

        model_df.loc[invalid_location, location_cols] = pd.NA


        # --- Extract useful information from Delivery_person_ID ---

        id_parts = split_delivery_person_id(model_df)

        # Fail loudly if the assumed ID format doesn't hold for every row
        assert id_parts.notna().all().all()

        model_df["City_code"] = id_parts["p"]

        model_df["Restaurant_ID"] = (
            id_parts["p"]
            + "_RES"
            + id_parts["res"]
        )


        # Drop Delivery_person_ID as it isn't useful for modelling
        model_df = model_df.drop(columns="Delivery_person_ID")

        # Drop unused categorical labels
        for _col in model_df.select_dtypes(include="category"):
            model_df[_col] = model_df[_col].cat.remove_unused_categories()

        return model_df

    return (prepare_model_df,)


@app.cell
def _(formatted_df, prepare_model_df):
    model_df = prepare_model_df(formatted_df)
    return (model_df,)


@app.cell
def _(model_df):
    model_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Derived features
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now add the following derived features for modelling:

    * `Time_to_prep` as `Time_Order_picked - Time_Ordered` accounting for orders placed around midnight
    * `Delivery_distance` as the Haverstine distance of restaurant lat / long and delivery lat / long
    * `Day_of_week` from the `Order_Date` (0 Monday - 6 Sunday)
    """)
    return


@app.cell
def _(pd):
    def add_time_to_prep(df):
        df = df.copy()

        prep_time = (
            df["Time_Order_picked"] - df["Time_Orderd"]
        ) % pd.Timedelta(days=1)

        df["Time_to_prep"] = prep_time.dt.total_seconds() / 60

        return df

    import numpy as np


    def add_delivery_distance(df):
        df = df.copy()

        lat1 = np.radians(df["Restaurant_latitude"])
        lon1 = np.radians(df["Restaurant_longitude"])
        lat2 = np.radians(df["Delivery_location_latitude"])
        lon2 = np.radians(df["Delivery_location_longitude"])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1)
            * np.cos(lat2)
            * np.sin(dlon / 2) ** 2
        )

        # Guard against tiny floating-point excursions outside [0, 1]
        a = np.clip(a, 0, 1)

        c = 2 * np.arctan2(
            np.sqrt(a),
            np.sqrt(1 - a),
        )

        earth_radius_km = 6371.0088

        df["Delivery_distance"] = earth_radius_km * c

        return df

    def add_day_of_week(df):
        df = df.copy()

        df["Day_of_week"] = df["Order_Date"].dt.dayofweek
        return df

    return add_day_of_week, add_delivery_distance, add_time_to_prep


@app.cell
def _(add_day_of_week, add_delivery_distance, add_time_to_prep):
    def prepare_model_feats_df(model_df):
        model_feats_df = add_time_to_prep(model_df)
        model_feats_df = add_delivery_distance(model_feats_df)
        model_feats_df = add_day_of_week(model_feats_df)

        return model_feats_df

    return (prepare_model_feats_df,)


@app.cell
def _(model_df, prepare_model_feats_df):
    model_feats_df = prepare_model_feats_df(model_df)
    return (model_feats_df,)


@app.cell
def _(plt):
    def plot_hist(df, column):
        series = df[column]

        series.hist(bins=40)
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.title(f"Distribution of {column}")
        plt.show()

        print(series.describe())

    return (plot_hist,)


@app.cell
def _(model_feats_df, plot_counts, plot_hist):
    plot_hist(model_feats_df, "Time_to_prep")
    plot_hist(model_feats_df, "Delivery_distance")
    plot_counts(model_feats_df, "Day_of_week")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### More observations

    * The values and distribution of both `Time_to_prep` (falls into only three possible values) and `Delivery_distance` (equally sized peaks at fixed values up to 7.5km) strongly suggests, again, synthetic generation of these fields
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Store results
    """)
    return


@app.cell
def _(formatted_df, model_df, model_feats_df):
    formatted_df.to_parquet("processed/train_formatted_df.parquet")
    model_df.to_parquet("processed/train_model_df.parquet")
    model_feats_df.to_parquet("processed/train_model_feats_df.parquet")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Apply same transformations to test data
    """)
    return


@app.cell
def _(clean_df, format_df, pd, prepare_model_df, prepare_model_feats_df):
    df_test = pd.read_csv("dataset/test.csv")
    cleaned_df_test = clean_df(df_test)
    formatted_df_test = format_df(cleaned_df_test, is_test_set=True)
    model_df_test = prepare_model_df(formatted_df_test, is_test_set=True)
    model_feats_df_test = prepare_model_feats_df(model_df_test)
    return df_test, formatted_df_test, model_df_test, model_feats_df_test


@app.cell
def _(formatted_df_test, model_df_test, model_feats_df_test):
    formatted_df_test.to_parquet("processed/test_formatted_df.parquet")
    model_df_test.to_parquet("processed/test_model_df.parquet")
    model_feats_df_test.to_parquet("processed/test_model_feats_df.parquet")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Sanity checks to ensure the data is as expected
    """)
    return


@app.cell
def _(cleaned_df, df_test, model_feats_df, model_feats_df_test):
    ## Sanity checks

    assert len(model_feats_df_test) == len(df_test)

    assert len(model_feats_df) == (
        len(cleaned_df) - cleaned_df["Type_of_vehicle"].eq("bicycle").sum()
    )

    assert (
        model_feats_df.drop(columns="time_taken_min").columns.tolist()
        == model_feats_df_test.columns.tolist()
    )

    for dataset in [model_feats_df, model_feats_df_test]:
        assert dataset["Delivery_distance"].dropna().ge(0).all()
        assert dataset["Time_to_prep"].dropna().between(0, 1440).all()
        assert dataset["Day_of_week"].dropna().between(0, 6).all()
    return


if __name__ == "__main__":
    app.run()
