import numpy as np


def compute_cate(preds,
                 i_c=0,
                 i_t=1):
    """
    Estimates CATE given predictions on an active treatment and a control
    :param preds: MLP.forward() output. List of length num treatments containing predictions for each sample of shape (num samples, D_out)
    :param i_c: label for the control treatment in the "treatment" parameter
    :param i_t: label for the active treatment in the "treatment" parameter
    :return: 1-D tensor containing CATE estimates for each sample, length num_samples
    """

    cate = preds[i_t] - preds[i_c]

    return cate


def get_responder_mask(cate, quantiles):
    """
    Computes the responder mask at each quantile threshold
    :param cate: 1-D np.ndarray of CATE estimates for each sample, length num samples
    :param quantiles: quantiles thresholds (c) at which to compute the AD(c) values
    :return: list of boolean masks (np.ndarray) for each quantile threshold (True if a sample is a responder, False otherwise), length num quantiles
    """
    thresh = np.quantile(cate, q=quantiles)

    mask = []
    for i_thresh in range(len(thresh)):
        # Extract masks for patients with rs > rs_thresh
        mask.append((cate >= thresh[i_thresh]).ravel())

    return mask
