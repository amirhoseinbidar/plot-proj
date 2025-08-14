from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.collections import PolyCollection
import matplotlib.dates as mdates
import pandas as pd


def get_block_intervals(poly: PolyCollection, check_overlap=True):
    path = poly.get_paths()[0]  # usually only one path per layer
    verts = path.vertices
    n = len(verts) // 2
    if len(verts) % 2 != 0:
        verts = np.delete(verts, n, axis=0)

    # Split into top and bottom boundaries
    bottom = verts[:n]
    top = verts[n:][::-1]  # reverse top so it aligns with bottom

    intervals = []
    start = end = -1
    for idx, (t, b) in enumerate(zip(top[:, 1], bottom[:, 1])):
        if idx < len(bottom) - 1:
            next_t, next_b = top[idx + 1, 1], bottom[idx + 1, 1]
        else:
            next_t, next_b = -1, -1
        end += 1

        # conditions for restarting interval are as bellow.
        # 1- data will be changed in next iteration.
        # 2- current bar or next bar is not 0.
        # 3- there is not any overlap between current bar and next bar.

        # XXX: Adding a tolerance factor for overlap could be good idea.
        # like setting a condition to restart interval only if overlap
        # area is less than 50%.
        condition = ((next_t != t) or (next_b != b)) and (
            ((next_t - next_b) > 0) or ((t - b) > 0)
        )
        if check_overlap:
            condition = condition and (not (next_b < t and b < next_t))

        if condition:
            if start != -1:
                intervals.append((start + 1, end + 1))
            start = end = idx
    return intervals, top, bottom


def get_block_centers(poly: PolyCollection):
    intervals, top, bottom = get_block_intervals(poly)
    centers = []
    for from_, to_ in intervals:
        avg_x = np.average(bottom[from_:to_, 0])
        avg_y = np.average(
            bottom[from_:to_, 1] + ((top[from_:to_, 1] - bottom[from_:to_, 1]) / 2)
        )
        centers.append((avg_x, avg_y))

    return centers


def plot_budget(
    df: pd.DataFrame,
    priority_type: Literal["budget", "start", "duration"],
    priority_ascending: bool,
    ax: plt.Axes = None,
    plot_today: bool = True,
):
    now = datetime.now()
    edges = sorted(set(df["start"]).union(set(df["end"])))
    values = {name: [0] * len(edges) for name in df["name"]}

    for i, edge in enumerate(edges):
        active_records = df[(df["start"] <= edge) & (df["end"] > edge)]
        for _, record in active_records.iterrows():
            values[record["name"]][i] = record["budget"]

    x = np.array(edges, dtype="datetime64")
    priority = df.sort_values(by=priority_type, ascending=priority_ascending)[
        "name"
    ].tolist()
    y_priority = np.vstack([values[name] for name in priority])

    if not ax:
        _, ax = plt.subplots(figsize=(8, 4))

    # plot bars
    polys = ax.stackplot(x, y_priority, labels=priority, baseline="zero", step="post")

    # plot today
    if plot_today:
        ax.axvline(
            x=now,
            color="black",
            label=f"today ({now.strftime('%Y-%m-%d')})",
            linestyle="dashed",
        )

    # set bars' label on them
    for poly in polys:
        centers = get_block_centers(poly)
        for x, y in centers:
            ax.text(
                x,
                y,
                poly.get_label(),
                horizontalalignment="center",
                verticalalignment="center",
            )

    ax.legend()
    return ax, polys


def plot_budget_cumsum(df: pd.DataFrame, ax: plt.Axes = None, plot_today: bool = True):
    start_budget_map = df.groupby("start")["budget"].sum().to_dict()
    end_budget_map = df.groupby("end")["budget"].sum().to_dict()
    edges = sorted(set(df["start"]).union(set(df["end"])))

    values = []
    sum = 0
    for edge in edges:
        if edge in start_budget_map:
            sum += start_budget_map[edge]
        if edge in end_budget_map:
            sum -= end_budget_map[edge]
        values.append(sum)
    values.pop()
    if not ax:
        _, ax = plt.subplots(figsize=(8, 4))
    now = datetime.now()

    ax.stairs(values, edges, label="budget chart", color="tab:blue")
    if plot_today:
        ax.axvline(
            x=now,
            color="black",
            label=f"today ({now.strftime('%Y-%m-%d')})",
            linestyle="dashed",
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=60)
    ax.legend()
    return ax
