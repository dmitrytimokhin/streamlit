"""
Optimal binning algorithm 2D.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2021

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable

from ...formatting import dataframe_to_string
from ..binning_statistics import _check_build_parameters
from ..binning_statistics import _check_is_built
from ..binning_statistics import BinningTable
from ..metrics import bayesian_probability
from ..metrics import binning_quality_score
from ..metrics import chi2_cramer_v
from ..metrics import frequentist_pvalue
from ..metrics import hhi
from ..metrics import gini
from ..metrics import hellinger
from ..metrics import jeffrey
from ..metrics import jensen_shannon
from ..metrics import triangular


def _bin_fmt(bin, show_digits):
    if np.isinf(bin[0]):
        return "({0:.{2}f}, {1:.{2}f})".format(bin[0], bin[1], show_digits)
    else:
        return "[{0:.{2}f}, {1:.{2}f})".format(bin[0], bin[1], show_digits)


def bin_xy_str_format(bins_x, bins_y, show_digits):
    show_digits = 2 if show_digits is None else show_digits

    bins_xy = []
    for bx, by in zip(bins_x, bins_y):
        _bx = _bin_fmt(bx, show_digits)
        _by = _bin_fmt(by, show_digits)
        bins_xy.append(r"{} $\cup$ {}".format(_bx, _by))

    return bins_xy


def bin_str_format(bins, show_digits):
    show_digits = 2 if show_digits is None else show_digits

    bin_str = []
    for bin in bins:
        bin_str.append(_bin_fmt(bin, show_digits))

    return bin_str


class BinningTable2D(BinningTable):
    """Binning table to summarize optimal binning of two numerical variables
    with respect to a binary target.

    Parameters
    ----------
    name_x : str, optional (default="")
        The name of variable x.

    name_y : str, optional (default="")
        The name of variable y.

    dtype_x : str, optional (default="numerical")
        The data type of variable x. Supported data type is "numerical" for
        continuous and ordinal variables.

    dtype_y : str, optional (default="numerical")
        The data type of variable y. Supported data type is "numerical" for
        continuous and ordinal variables.

    splits_x : numpy.ndarray
        List of split points for variable x.

    splits_y : numpy.ndarray
        List of split points for variable y.

    m : int
        Number of rows of the 2D array.

    n : int
        Number of columns of the 2D array.

    n_nonevent : numpy.ndarray
        Number of non-events.

    n_event : numpy.ndarray
        Number of events.

    D : numpy.ndarray
        Event rate 2D array.

    P : numpy-ndarray
        Records 2D array.

    Warning
    -------
    This class is not intended to be instantiated by the user. It is
    preferable to use the class returned by the property ``binning_table``
    available in all optimal binning classes.
    """
    def __init__(self, name_x, name_y, dtype_x, dtype_y, splits_x, splits_y,
                 m, n, n_nonevent, n_event, D, P):

        self.name_x = name_x
        self.name_y = name_y
        self.dtype_x = dtype_x
        self.dtype_y = dtype_y
        self.splits_x = splits_x
        self.splits_y = splits_y
        self.m = m
        self.n = n
        self.n_nonevent = n_nonevent
        self.n_event = n_event
        self.D = D
        self.P = P

        self._is_built = False
        self._is_analyzed = False

    def build(self, show_digits=2, show_bin_xy=False, add_totals=True):
        """Build the binning table.

        Parameters
        ----------
        show_digits : int, optional (default=2)
            The number of significant digits of the bin column.

        show_bin_xy: bool (default=False)
            Whether to show a single bin column with x and y.

        add_totals : bool (default=True)
            Whether to add a last row with totals.

        Returns
        -------
        binning_table : pandas.DataFrame
        """
        _check_build_parameters(show_digits, add_totals)

        if not isinstance(show_bin_xy, bool):
            raise TypeError("show_bin_xy must be a boolean; got {}."
                            .format(show_bin_xy))

        n_nonevent = self.n_nonevent
        n_event = self.n_event

        n_records = n_event + n_nonevent
        t_n_nonevent = n_nonevent.sum()
        t_n_event = n_event.sum()
        t_n_records = t_n_nonevent + t_n_event
        t_event_rate = t_n_event / t_n_records

        p_records = n_records / t_n_records
        p_event = n_event / t_n_event
        p_nonevent = n_nonevent / t_n_nonevent

        mask = (n_event > 0) & (n_nonevent > 0)
        event_rate = np.zeros(len(n_records))
        woe = np.zeros(len(n_records))
        iv = np.zeros(len(n_records))
        js = np.zeros(len(n_records))

        # Compute weight of evidence and event rate
        event_rate[mask] = n_event[mask] / n_records[mask]
        constant = np.log(t_n_event / t_n_nonevent)
        woe[mask] = np.log(1 / event_rate[mask] - 1) + constant
        W = np.log(1 / self.D - 1) + constant

        # Compute Gini
        self._gini = gini(self.n_event, self.n_nonevent)

        # Compute divergence measures
        p_ev = p_event[mask]
        p_nev = p_nonevent[mask]

        iv[mask] = jeffrey(p_ev, p_nev, return_sum=False)
        js[mask] = jensen_shannon(p_ev, p_nev, return_sum=False)
        t_iv = iv.sum()
        t_js = js.sum()

        self._iv = t_iv
        self._js = t_js
        self._hellinger = hellinger(p_ev, p_nev, return_sum=True)
        self._triangular = triangular(p_ev, p_nev, return_sum=True)

        # Keep data for plotting
        self._n_records = n_records
        self._event_rate = event_rate
        self._woe = woe
        self._W = W

        # Compute KS
        self._ks = np.abs(p_event.cumsum() - p_nonevent.cumsum()).max()

        # Compute HHI
        self._hhi = hhi(p_records)
        self._hhi_norm = hhi(p_records, normalized=True)

        # Compute paths. This is required for both plot and analysis
        # paths x: horizontal
        self._paths_x = []
        for i in range(self.m):
            path = tuple(dict.fromkeys(self.P[i, :]))
            if path not in self._paths_x:
                self._paths_x.append(path)

        # paths y: vertical
        self._paths_y = []
        for j in range(self.n):
            path = tuple(dict.fromkeys(self.P[:, j]))
            if path not in self._paths_y:
                self._paths_y.append(path)

        if show_bin_xy:
            bin_xy_str = bin_xy_str_format(self.splits_x, self.splits_y,
                                           show_digits)

            bin_xy_str.extend(["Special", "Missing"])

            df = pd.DataFrame({
                "Bin": bin_xy_str,
                "Count": n_records,
                "Count (%)": p_records,
                "Non-event": n_nonevent,
                "Event": n_event,
                "Event rate": event_rate,
                "WoE": woe,
                "IV": iv,
                "JS": js
                })
        else:
            bin_x_str = bin_str_format(self.splits_x, show_digits)
            bin_y_str = bin_str_format(self.splits_y, show_digits)

            bin_x_str.extend(["Special", "Missing"])
            bin_y_str.extend(["Special", "Missing"])

            df = pd.DataFrame({
                "Bin x": bin_x_str,
                "Bin y": bin_y_str,
                "Count": n_records,
                "Count (%)": p_records,
                "Non-event": n_nonevent,
                "Event": n_event,
                "Event rate": event_rate,
                "WoE": woe,
                "IV": iv,
                "JS": js
                })

        if add_totals:
            if show_bin_xy:
                totals = ["", t_n_records, 1, t_n_nonevent, t_n_event,
                          t_event_rate, "", t_iv, t_js]
            else:
                totals = ["", "", t_n_records, 1, t_n_nonevent, t_n_event,
                          t_event_rate, "", t_iv, t_js]

            df.loc["Totals"] = totals

        self._is_built = True

        return df

    def plot(self, metric="woe", savefig=None):
        """Plot the binning table.

        Visualize the non-event and event count, and the Weight of Evidence or
        the event rate for each bin.

        Parameters
        ----------
        metric : str, optional (default="woe")
            Supported metrics are "woe" to show the Weight of Evidence (WoE)
            measure and "event_rate" to show the event rate.

        savefig : str or None (default=None)
            Path to save the plot figure.
        """
        _check_is_built(self)

        if metric not in ("event_rate", "woe"):
            raise ValueError('Invalid value for metric. Allowed string '
                             'values are "event_rate" and "woe".')

        if metric == "woe":
            metric_values = self._woe
            metric_matrix = self._W
            metric_label = "WoE"
        elif metric == "event_rate":
            metric_values = self._event_rate
            metric_matrix = self.D
            metric_label = "Event rate"

        fig, ax = plt.subplots(figsize=(7, 7))

        divider = make_axes_locatable(ax)
        axtop = divider.append_axes("top", size=2.5, pad=0.1, sharex=ax)
        axright = divider.append_axes("right", size=2.5, pad=0.1, sharey=ax)
        # Hide x labels and tick labels for top plots and y ticks for
        # right plots.

        # Position [0, 0]
        for path in self._paths_x:
            er = sum([
                [metric_values[p]] * np.count_nonzero(
                    self.P == p, axis=1).max() for p in path], [])

            er = er + [er[-1]]
            axtop.step(np.arange(self.n + 1) - 0.5, er,
                       label=path, where="post")

        for i in range(self.n):
            axtop.axvline(i + 0.5, color="grey", linestyle="--", alpha=0.5)

        axtop.get_xaxis().set_visible(False)
        axtop.set_ylabel(metric_label, fontsize=12)

        # Position [1, 0]
        pos = ax.matshow(metric_matrix, cmap=plt.cm.bwr)
        for j in range(self.n):
            for i in range(self.m):
                c = int(self.P[i, j])
                ax.text(j, i, str(c), va='center', ha='center')

        fig.colorbar(pos, ax=ax, orientation="horizontal",
                     fraction=0.025, pad=0.125)

        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()
        ax.set_ylabel("Bin ID - y ({})".format(self.name_x), fontsize=12)
        ax.set_xlabel("Bin ID - x ({})".format(self.name_y), fontsize=12)

        # Position [1, 1]
        for path in self._paths_y:
            er = sum([
                [metric_values[p]] * (np.count_nonzero(
                    self.P == p, axis=0).max()) for p in path], [])

            er = er + [er[-1]]
            axright.step(er, np.arange(self.m + 1) - 0.5, label=path,
                         where="pre")

        for j in range(self.m):
            axright.axhline(j - 0.5, color="grey", linestyle="--", alpha=0.5)

        axright.get_yaxis().set_visible(False)
        axright.set_xlabel(metric_label, fontsize=12)

        # adjust margins
        axright.margins(y=0)
        axtop.margins(x=0)
        plt.tight_layout()

        axtop.legend(bbox_to_anchor=(1, 1))
        axright.legend(bbox_to_anchor=(1, 1))

        if savefig is None:
            plt.show()
        else:
            if not isinstance(savefig, str):
                raise TypeError("savefig must be a string path; got {}."
                                .format(savefig))
            plt.savefig(savefig)
            plt.close()

    def analysis(self, pvalue_test="chi2", n_samples=100, print_output=True):
        """Binning table analysis.

        Statistical analysis of the binning table, computing the statistics
        Gini index, Information Value (IV), Jensen-Shannon divergence, and
        the quality score. Additionally, several statistical significance tests
        between consecutive bins of the contingency table are performed: a
        frequentist test using the Chi-square test or the Fisher's exact test,
        and a Bayesian A/B test using the beta distribution as a conjugate
        prior of the Bernoulli distribution.

        Parameters
        ----------
        pvalue_test : str, optional (default="chi2")
            The statistical test. Supported test are "chi2" to choose the
            Chi-square test and "fisher" to choose the Fisher exact test.

        n_samples : int, optional (default=100)
            The number of samples to run the Bayesian A/B testing between
            consecutive bins to compute the probability of the event rate of
            bin A being greater than the event rate of bin B.

        print_output : bool (default=True)
            Whether to print analysis information.

        Notes
        -----
        The Chi-square test uses `scipy.stats.chi2_contingency
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.
        chi2_contingency.html>`_, and the Fisher exact test uses
        `scipy.stats.fisher_exact <https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.stats.fisher_exact.html>`_.
        """
        pairs = set()

        for path in self._paths_x:
            tpairs = tuple(zip(path[:-1], path[1:]))
            for tp in tpairs:
                pairs.add(tp)

        for path in self._paths_y:
            tpairs = tuple(zip(path[:-1], path[1:]))
            for tp in tpairs:
                pairs.add(tp)

        pairs = sorted(pairs)

        # Significance tests
        n_bins = len(self._n_records)
        n_metric = n_bins - 2

        n_nev = self.n_nonevent[:n_metric]
        n_ev = self.n_event[:n_metric]

        if len(n_nev) >= 2:
            chi2, cramer_v = chi2_cramer_v(n_nev, n_ev)
        else:
            cramer_v = 0

        t_statistics = []
        p_values = []
        p_a_b = []
        p_b_a = []
        for pair in pairs:
            obs = np.array([n_nev[list(pair)], n_ev[list(pair)]])
            t_statistic, p_value = frequentist_pvalue(obs, pvalue_test)
            pab, pba = bayesian_probability(obs, n_samples)

            p_a_b.append(pab)
            p_b_a.append(pba)

            t_statistics.append(t_statistic)
            p_values.append(p_value)

        df_tests = pd.DataFrame({
                "Bin A": np.array([p[0] for p in pairs]),
                "Bin B": np.array([p[1] for p in pairs]),
                "t-statistic": t_statistics,
                "p-value": p_values,
                "P[A > B]": p_a_b,
                "P[B > A]": p_b_a
            })

        if pvalue_test == "fisher":
            df_tests.rename(columns={"t-statistic": "odd ratio"}, inplace=True)

        tab = 4
        if len(df_tests):
            df_tests_string = dataframe_to_string(df_tests, tab)
        else:
            df_tests_string = " " * tab + "None"

        # Quality score
        self._quality_score = binning_quality_score(self._iv, p_values,
                                                    self._hhi_norm)

        report = (
            "------------------------------------------------\n"
            "OptimalBinning: Binary Binning Table 2D Analysis\n"
            "------------------------------------------------\n"
            "\n"
            "  General metrics"
            "\n\n"
            "    Gini index          {:>15.8f}\n"
            "    IV (Jeffrey)        {:>15.8f}\n"
            "    JS (Jensen-Shannon) {:>15.8f}\n"
            "    Hellinger           {:>15.8f}\n"
            "    Triangular          {:>15.8f}\n"
            "    KS                  {:>15.8f}\n"
            "    HHI                 {:>15.8f}\n"
            "    HHI (normalized)    {:>15.8f}\n"
            "    Cramer's V          {:>15.8f}\n"
            "    Quality score       {:>15.8f}\n"
            "\n"
            "  Significance tests\n\n{}\n"
            ).format(self._gini, self._iv, self._js, self._hellinger,
                     self._triangular, self._ks, self._hhi, self._hhi_norm,
                     cramer_v, self._quality_score, df_tests_string)

        if print_output:
            print(report)

        self._is_analyzed = True
