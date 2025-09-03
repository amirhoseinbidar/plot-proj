from enum import Enum, auto
from typing import Literal
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import warnings


def calculate_staff_cumsum_cost(
    df: pd.DataFrame, initial_cost: int = 0
) -> tuple[list, list]:
    """Calculates cumulative staff cost over time based on start and end dates and monthly rates.

    This function processes a DataFrame containing staff cost information and calculates the
    cumulative cost over time, taking into account varying monthly rates and time periods.

    Args:
        df (pd.DataFrame): DataFrame containing columns:
            - start: Start date of the cost period
            - end: End date of the cost period
            - monthly-rate: Monthly cost rate
        initial_cost (int, optional): Initial cost value to start calculations from. Defaults to 0.

    Returns:
        tuple: A tuple containing two lists:
            - values (list): Cumulative cost values at each timestamp
            - timestamps (list): Corresponding timestamps for each cost value
    """
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


class WP_Enum(Enum):
    FILTER_ALL = auto()


def calculate_staffs_proportional_cost(
    participation_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    project_name: str,
    work_package_id: int | None | WP_Enum = WP_Enum.FILTER_ALL,
    start: datetime | None = None,
    end: datetime | None = None,
):
    """Calculate the proportional cost of staffs for a specific project within a time period.

    This function calculates the cost of staff members assigned to a project, taking into account
    their participation rate and the time period of involvement. The cost is adjusted proportionally
    based on the staff's participation percentage in the project.

    Args:
        participation_df (pd.DataFrame): DataFrame containing staff participation data.
        cost_df (pd.DataFrame): DataFrame containing staff cost data.
        budget_df (pd.DataFrame): DataFrame containing project budget data.
        project_name (str): Name of the project to calculate costs for.
        start (datetime | None, optional): Start date for cost calculation. If None,
            uses project start date. Defaults to None.
        end (datetime | None, optional): End date for cost calculation. If None,
            uses project end date. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing proportional cost data for each staff member,
            with adjusted monthly rates based on their participation in the project.

    Note:
        The function adjusts the monthly rate of each staff member based on their
        participation percentage in the project. If a staff member is not found in
        the participation records for the project, their monthly rate is set to 0.
    """
    project = budget_df[budget_df["project"] == project_name]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = project["start"].min() if start is None else np.datetime64(start)
        end = project["end"].max() if end is None else np.datetime64(end)

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
            & (participation_df["project"] == project_name)
        ]
        if work_package_id != WP_Enum.FILTER_ALL:
            if work_package_id is None:
                participation_record = participation_record[
                    participation_df["work-package"].isna()
                ]
            else:
                participation_record = participation_record[
                    participation_record["work-package"] == work_package_id
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
        participation_df, cost_df, budget_df, project_name, start=start, end=end
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
) -> tuple[list[pd.Series], list]:
    """
    Calculate cumulative project costs over a shared time grid for multiple projects.

    The process involves:
        1. Calculating proportional staff costs for each project over its budget period.
        2. Computing cumulative sums of costs over time.
        3. Building a unified time grid across all projects. Time alignment ensures that
           plots of multiple projects will be synchronized.
        4. Forward-filling missing values between known data points.
        5. Inserting drop points at the end of each project to ensure immediate cost fall-off in plots.

    Args:
        participation_df (pd.DataFrame): DataFrame containing staff participation information,
            such as which staff members participated in which projects and during what periods.
        cost_df (pd.DataFrame): DataFrame containing cost information for each staff member.
        budget_df (pd.DataFrame): DataFrame containing project budget information, including
            project name, start date, and end date.

    Returns:
        tuple:
            - list[pd.Series]: A list of cumulative cost Series for each project,
              aligned to the same time grid and adjusted for plotting.
            - list: A sorted list of all timestamps in the time grid, including
              project end timestamps where drops occur.
    """

    time_grid = set()
    labels = []
    series_list = []
    last_drop_timestamps = []
    for idx, record in budget_df.iterrows():
        proportional_cost = calculate_staffs_proportional_cost(
            participation_df,
            cost_df,
            budget_df,
            record["project"],
            start=record["start"],
            end=record["end"],
        )
        values, timestamps = calculate_staff_cumsum_cost(proportional_cost)

        values.pop()
        last_drop_timestamps.append(timestamps.pop())

        time_grid = time_grid.union(set(timestamps))
        series_list.append(pd.Series(values, timestamps))
        labels.append(record["project"])

    time_grid = sorted(time_grid)

    for idx, s in enumerate(series_list):
        # each series is reindexed to make sure that all series have same index.
        s_idx = s.index
        s = s.reindex(time_grid)

        # empty records between original values have to be filled using linear regression method
        for from_, to_ in zip(s_idx, s_idx[1:]):
            s.loc[from_:to_] = s.loc[from_:to_].interpolate(method="linear")

        s = s.fillna(0).sort_index()

        # zero values are added at the end of project to make sure that plot will fall immediately.
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

    return series_list, sorted(time_grid + last_drop_timestamps), labels


def plot_projects_total_cumsum_cost(
    participation_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    ax: plt.Axes = None,
    **kwargs,
):
    series, date_grid, _ = calculate_projects_cumsum_cost(
        participation_df, cost_df, budget_df
    )
    if not ax:
        _, ax = plt.subplots()

    ax.fill(date_grid, sum(series), **kwargs)
    return ax, series, date_grid


def plot_projects_stacked_cumsum_cost(
    participation_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    priority_type: Literal["budget", "start", "duration"],
    priority_ascending: bool,
    ax: plt.Axes = None,
    **kwargs,
):
    series, date_grid, labels = calculate_projects_cumsum_cost(
        participation_df, cost_df, budget_df
    )
    if not ax:
        _, ax = plt.subplots()

    priority = budget_df.sort_values(by=priority_type, ascending=priority_ascending)[
        "project"
    ].tolist()

    # swap rows to fit order
    for p_idx, p in enumerate(priority):
        series[labels.index(p)], series[p_idx] = series[p_idx], series[labels.index(p)]
        labels[labels.index(p)], labels[p_idx] = labels[p_idx], labels[labels.index(p)]

    ax.stackplot(date_grid, labels=priority, *series, **kwargs)
    return ax, series, date_grid
