import math
from sklearn.metrics import mean_squared_error

def unfold(dict):
    elem = []
    for key in dict:
        for t in dict[key]:
            elem.append(t[0])
    return elem

def reverse_dict(dict):
    rev_dict = {}
    for key in dict:
        for idx, prob1, prob2 in dict[key]:
            if idx in rev_dict:
                rev_dict[idx].append((key, prob1, prob2))
            else:
                rev_dict[idx] = [(key, prob1, prob2)]
    return rev_dict

def dup_filtering(dict):
    new_dict = {}
    for key in dict:
        best_tuple = (None, -1, -1)
        for tuple in dict[key]:
            if tuple[1] > best_tuple[1]:
                best_tuple = tuple
            if tuple[1] == best_tuple[1]:
                if tuple[2] > best_tuple[2]:
                    best_tuple = tuple
        if key in new_dict:
            new_dict[key].append(best_tuple)
        else:
            new_dict[key] = [best_tuple]
    return new_dict

def avg(values, weights=None):
    if weights is None:
        weights = [1 for _ in range(len(values))]
    norm = 0
    val = 0
    for value, weight in zip(values, weights):
        val += value * weight
        norm += weight
    return val / norm

def rmse_dict(true_dict, pred_dict):
    ret = {}
    for key in true_dict:
        ret[key] = mean_squared_error(true_dict[key], pred_dict[key], squared=False)
    return ret