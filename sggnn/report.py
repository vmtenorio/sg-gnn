"""Small helpers shared by the analysis/ scripts."""
import logging

import numpy as np


class Summary(list):
    """Lines of a summary.txt, also echoed to the log as they are added."""

    def emit(self, line=''):
        self.append(line)
        logging.info(line)


def loglog_slope(x, y):
    """Fitted exponent of y ~ x^b, plus R^2, over strictly positive finite entries."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float('nan'), float('nan')
    b, a = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    pred = a + b * np.log(x[m])
    ss = 1 - np.sum((np.log(y[m]) - pred) ** 2) / np.sum((np.log(y[m]) - np.log(y[m]).mean()) ** 2)
    return b, ss


def latex_thousands(x, dec=0):
    """Thousands-separated number with LaTeX's '{,}' as the group separator."""
    return f"{x:,.{dec}f}".replace(',', '{,}')
