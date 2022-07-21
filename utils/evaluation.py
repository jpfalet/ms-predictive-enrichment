from sklearn.metrics import auc
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from scipy.stats import spearmanr
from treatment_effect import get_responder_mask
import numpy as np


def compute_ad(cate,
               treatment,
               time_to_event,
               censorship,
               quantiles,
               i_c=0,
               i_t=1,
               clamp_time=2.):
    """
    Computes the AD(c) curve values for each threshold c
    Adapted from Zhao et al. J Am Stat Assoc. 2013 January 1; 108(502): 527–539. doi:10.1080/01621459.2013.770705

    :param cate: 1-D np.ndarray of CATE estimates for each sample, length num samples
    :param treatment: 1-D np.ndarray containing the binary [0,1] treatment assignment for each sample, length num samples
    :param time_to_event: 1-D np.ndarray containing the time-to-event for each sample, length num samples
    :param censorship: 1-D np.ndarray containing a binary [0,1] indicator of censorship for each sample, length num samples
    :param quantiles: tuple containing the quantiles thresholds (c) at which to compute the AD(c) values
    :param i_c: label for the control treatment in the "treatment" parameter
    :param i_t: label for the active treatment in the "treatment" parameter
    :param clamp_time: the time (in years) at which to estimate the RMST
    :return: AD(c) curve values for each threshold c in quantiles (list)
    """

    responders = get_responder_mask(cate, quantiles)

    ad = []
    for i_thresh in range(len(quantiles)):

        # Compute RMST as the main group statistic (instead of HR)
        metric = []
        for i_arm in i_c, i_t:
            responders_arm = np.logical_and(responders[i_thresh], treatment == i_arm)

            # Extract masks for placebo and treatment patients with rs > rs_thresh
            kmf = KaplanMeierFitter().fit(time_to_event[responders_arm], censorship[responders_arm])
            metric.append(restricted_mean_survival_time(kmf, t=clamp_time))

        # Compute average treatment difference
        ad.append(np.mean(metric[1]) - np.mean(metric[0]))

    return ad


def compute_adwabc(ad, quantiles):
    """
    Calculates the AD_wabc using the trapezoid method for AUC
    The AD_wabc is the AD_abc described by Zhao et al., with additional weighting for monotonicity using Spearman's r.
    Refer to Zhao et al. J Am Stat Assoc. 2013 January 1; 108(502): 527–539. doi:10.1080/01621459.2013.770705

    :param ad: AD(c) curve values for each threshold c in quantiles (list)
    :param quantiles: quantile thresholds (c) from which the AD(c) values were computed
    :return: AD_wabc
    """
    if quantiles[0] != 0.:
        raise AssertionError("First quantile threshold should be 0. to calculate the AD_abc. ")

    # Subtract the y-intercept
    ad_0 = ad[0]
    ad_ref = np.array(ad) - ad_0

    # Empirical AUC using trapezoid method to obtain AD_abc
    ad_abc = auc(quantiles, ad_ref)

    abs_r = np.abs(spearmanr(quantiles, ad_ref)[0])
    ad_wabc = abs_r * ad_abc

    return ad_wabc
