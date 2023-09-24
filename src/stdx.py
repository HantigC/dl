from collections import defaultdict


def transpose(xss):
    return list(zip(*xss))


def transpos_dict(dict_list):
    list_dict = defaultdict(list)
    for d in dict_list:
        for k, v in d.items():
            list_dict[k].append(v)
    return list_dict
