from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.lines import Line2D

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

    lbls = plt.gca().get_legend_handles_labels()
    ax.legend(
        handles=[Line2D([0], [0], color="black")] + lbls[0],
        labels=["Proportional cost"] + lbls[1],
    )
    return ax
