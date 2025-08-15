from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.patches as mpatches

from plot_functions.budget import get_block_intervals, plot_budget
from plot_functions.cost import (
    calculate_staff_cumsum_cost,
    calculate_staffs_proportional_cost,
)
import warnings


def plot_consumed_budget(
    participation_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    priority_type: Literal["budget", "start", "duration"],
    priority_ascending: bool,
    ax: plt.Axes = None,
):
    ax, polys = plot_budget(budget_df, priority_type, priority_ascending, ax, True)

    for poly in polys:
        intervals, top, bottom = get_block_intervals(poly, check_overlap=False)
        offset = 0
        for from_, to_ in intervals:
            times = bottom[from_:to_, 0]
            initial_cost = bottom[from_:to_, 1][0]

            start, end = (
                mdates.num2date(times.min()),
                mdates.num2date(times.max()),
            )

            proportional_cost = calculate_staffs_proportional_cost(
                participation_df, cost_df, budget_df, poly.get_label(), start, end
            )
            values, timestamps = calculate_staff_cumsum_cost(
                proportional_cost, initial_cost + offset
            )

            values.pop()
            timestamps.pop()
            offset = values[-1] - initial_cost

            ax.plot(timestamps, values, color="black")

            values.append(initial_cost)
            timestamps.append(end)
            values.append(initial_cost)
            timestamps.append(start)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.fill(timestamps, values, color="black")

    labels = plt.gca().get_legend_handles_labels()
    ax.legend(
        handles=[mpatches.Patch(color="black")] + labels[0],
        labels=["Proportional cost"] + labels[1],
    )
    return ax
