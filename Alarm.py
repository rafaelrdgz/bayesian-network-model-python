import pandas as pd

from BayesianNetwork import BayesianNetwork


def alarm() -> BayesianNetwork:

    bn = BayesianNetwork(
        ("Burglary", "Alarm"),
        ("Earthquake", "Alarm"),
        ("Alarm", "John calls"),
        ("Alarm", "Mary calls")
    )

    T = True
    F = False

    # P(Burglary)
    bn.P["Burglary"] = pd.Series({F: 0.999, T: 0.001})

    # P(Earthquake)
    bn.P["Earthquake"] = pd.Series({F: 0.998, T: 0.002})

    # P(Alarm | Burglary, Earthquake)
    bn.P["Alarm"] = pd.Series(
        {
            (T, T, T): 0.95,
            (T, T, F): 0.05,
            (T, F, T): 0.94,
            (T, F, F): 0.06,
            (F, T, T): 0.29,
            (F, T, F): 0.71,
            (F, F, T): 0.001,
            (F, F, F): 0.999,
        }
    )

    # P(John calls | Alarm)
    bn.P["John calls"] = pd.Series(
        {(T, T): 0.9, (T, F): 0.1, (F, T): 0.05, (F, F): 0.95}
    )

    # P(Mary calls | Alarm)
    bn.P["Mary calls"] = pd.Series(
        {(T, T): 0.7, (T, F): 0.3, (F, T): 0.01, (F, F): 0.99}
    )

    bn.prepare()

    return bn