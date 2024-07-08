import functools
import numpy as np
import pandas as pd


@pd.api.extensions.register_series_accessor("cdt")
class CDTAccessor:
    """
    Custom pandas series accessor for Conditional Probability Distribution Tables (CDTs).
    """

    def __init__(self, series: pd.Series):
        """
        Initialize the CDTAccessor.

        Parameters:
        - series (pd.Series): The pandas series to be accessed.
        """
        self.series = series
        self.sampler = None


    def sum_out(self, *variables):
        """
        Sum out the specified variables from the CDT.

        Parameters:
        - variables (tuple): The variables to be summed out.

        Returns:
        - pd.Series: The resulting series after summing out the variables.
        """
        # Get the names of the nodes in the CDT.
        nodes = list(self.series.index.names)
        # Remove the specified variables from the nodes.
        for var in variables:
            nodes.remove(var)
        # Group the series by the remaining nodes and sum them.    
        return self.series.groupby(nodes).sum()


def pointwise_mul_two(left: pd.Series, right: pd.Series):
    """
    Performs pointwise multiplication between two pandas Series objects.

    Args:
        left (pd.Series): The first pandas Series object.
        right (pd.Series): The second pandas Series object.

    Returns:
        pd.Series: The result of pointwise multiplication between the two Series objects.
    """
    # If the intersection of the names of the indices of the two series is empty, perform the Cartesian product.
    if not set(left.index.names) & set(right.index.names):
        df = pd.DataFrame(
            np.outer(left, right), index=left.index, columns=right.index
        )
        #This changes the wide table to a long table.
        return df.stack(list(range(df.columns.nlevels)))

    # If the intersection of the names of the indices of the two series is not empty, perform the inner join.
    (
        index,
        l_idx,
        r_idx,
    ) = left.index.join(right.index, how="inner", return_indexers=True)

    # If the indices of the two series are not aligned, align them.
    if l_idx is None:
        l_idx = np.arange(len(left))
    if r_idx is None:
        r_idx = np.arange(len(right))

    # Perform pointwise multiplication between the two series.
    return pd.Series(left.iloc[l_idx].values * right.iloc[r_idx].values, index=index)


def pointwise_mul(cpts):
    """
    Performs pointwise multiplication on a list of cpts (conditional probability tables).

    Args:
        cpts (list): A list of cpts (conditional probability tables).

    Returns:
        The result of pointwise multiplication on the cpts.

    """
    return functools.reduce(pointwise_mul_two, cpts)