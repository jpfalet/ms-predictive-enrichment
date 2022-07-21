import numpy as np
from torch import clamp


def stratify_by_treatment(treatment):
    """
    Outputs sample weights for stratified sampling by treatment
    :param treatment: 1-D np.ndarray containing the binary [0,1] treatment assignment for each sample, length num samples
    :return:
    """
    unique_class, class_sample_count = np.unique(treatment, return_counts=True)
    weight = class_sample_count / sum(class_sample_count)
    sample_weights = np.array([weight[unique_class == t] for t in treatment]).ravel()

    return sample_weights


def max_norm(model, max_val=3.):
    """
    Applies max-norm constraint as described by Srivastava et al. Journal of Machine Learning Research 15 (2014) 1929-1958.
    https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf

    :param model: Pytorch model instance
    :param max_val: float, value at which to clamp the maximum of the parameter norm
    :return: Parameters with max-norm constraint applied
    """
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm(2, dim=0, keepdim=True)
            desired = clamp(norm, 0, max_val)
            param *= (desired / norm)
