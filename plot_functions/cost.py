import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import warnings


def calculate_staff_cumsum_cost(df: pd.DataFrame, initial_cost: int = 0):
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values(by=["start"])
    df["total-effect"] = df["monthly-rate"] * (df["end"] - df["start"]).dt.days / 30

    start_rate_map = df.groupby("start")["monthly-rate"].sum().to_dict()
    end_rate_map = df.groupby("end")["monthly-rate"].sum().to_dict()

    edges = sorted(set(df["start"]).union(set(df["end"])))

    values, timestamps, sum, rate, previous_edge = [], [], initial_cost, 0, edges[0]
    for edge in edges:
        duration = (edge - previous_edge).days / 30
        sum += rate * duration
        values.append(sum)
        timestamps.append(edge)

        if edge in end_rate_map:
            rate -= end_rate_map[edge]

        if edge in start_rate_map:
            rate += start_rate_map[edge]

        previous_edge = edge

    values.append(initial_cost)
    timestamps.append(timestamps[-1])

    return values, timestamps


def plot_staff_cumsum_cost(
    df: pd.DataFrame,
    initial_cost: int = 0,
    ax: plt.Axes = None,
    plot_today: bool = True,
    color: str = None,
    label="Total cost chart",
    legend: bool = True,
):
    now = datetime.now()
    values, timestamps = calculate_staff_cumsum_cost(df, initial_cost)

    if not ax:
        fig, ax = plt.subplots()

    plt.xticks(rotation=60)

    if plot_today:
        ax.axvline(
            x=now,
            color="black",
            label=f"today ({now.strftime('%Y-%m-%d')})",
            linestyle="dashed",
        )
    if legend:
        ax.legend(loc="upper right")
        ax.plot(timestamps, values, label=label, color=color)
    else:
        ax.plot(timestamps, values, color=color)

    return ax, values, timestamps


def calculate_staffs_proportional_cost(
    participation_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    project_name: str,
    start: datetime | None = None,
    end: datetime | None = None,
):
    project = budget_df[budget_df["name"] == project_name].iloc[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = project["start"] if start is None else np.datetime64(start)
        end = project["end"] if end is None else np.datetime64(end)

    proportional_cost = cost_df[
        (cost_df["start"] < end) & (cost_df["end"] > start)
    ].copy()
    proportional_cost["start"] = proportional_cost["start"].apply(
        lambda x: max(x, start)
    )
    proportional_cost["end"] = proportional_cost["end"].apply(lambda x: min(x, end))

    for idx, record in proportional_cost.iterrows():
        participation_record = participation_df[
            (participation_df["staff"] == record["staff"])
            & (participation_df["name"] == project_name)
        ]
        if not participation_record.empty:
            rate = participation_record.iloc[0]["participation"]
            proportional_cost.at[idx, "monthly-rate"] = record["monthly-rate"] * rate
        else:
            proportional_cost.at[idx, "monthly-rate"] = 0

    return proportional_cost


def plot_staff_proportional_cost(
    participation_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    project_name: str,
    initial_cost: int = 0,
    start: datetime | None = None,
    end: datetime | None = None,
    ax: plt.Axes | None = None,
    plot_today: bool = True,
    color: str | None = None,
    legend: bool = True,
):
    proportional_cost = calculate_staffs_proportional_cost(
        participation_df, cost_df, budget_df, project_name, start, end
    )
    return plot_staff_cumsum_cost(
        proportional_cost,
        initial_cost,
        ax,
        plot_today,
        color,
        label="Proportional cost chart",
        legend=legend,
    )


def calculate_projects_cumsum_cost(
    participation_df: pd.DataFrame, cost_df: pd.DataFrame, budget_df: pd.DataFrame
):
    time_grid = set()
    series_list = []
    last_drop_timestamps = []
    for idx, record in budget_df.iterrows():
        proportional_cost = calculate_staffs_proportional_cost(
            participation_df,
            cost_df,
            budget_df,
            record["name"],
            record["start"],
            record["end"],
        )
        values, timestamps = calculate_staff_cumsum_cost(proportional_cost)

        values.pop()
        last_drop_timestamps.append(timestamps.pop())

        time_grid = time_grid.union(set(timestamps))
        series_list.append(pd.Series(values, timestamps))

    time_grid = sorted(time_grid)

    # I reindex each series in order to make sure their all series have same index.
    # Then, I fill None values between values with forward fill method.
    for idx, s in enumerate(series_list):
        s = s.reindex(time_grid)
        first_valid = s.first_valid_index()
        last_valid = s.last_valid_index()
        s.loc[first_valid:last_valid] = s.loc[first_valid:last_valid].ffill()
        s = s.fillna(0).sort_index()

        for drop in last_drop_timestamps:
            i = s.index.get_loc(drop)
            if i < len(s) - 1:
                if s.iloc[i] and s.iloc[i + 1] == 0:
                    ser = pd.Series([0], index=[drop])
                else:
                    ser = pd.Series([s.iloc[i]], index=[drop])
            else:
                ser = pd.Series([0], index=[drop])

            s = pd.concat([s.iloc[: i + 1], ser, s.iloc[i + 1 :]])

        series_list[idx] = s.sort_index()

    return series_list, sorted(time_grid + last_drop_timestamps)
