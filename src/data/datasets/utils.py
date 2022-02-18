def dl2ld(dl):
    """
    Convert a dictionary of lists to a list of dictonaries
    Example:
        dl = {'a': [0, 1], 'b': [2, 3]}
        ld = [{'a': 0, 'b': 2}, {'a': 1, 'b': 3}]
    """
    return [dict(zip(dl, t)) for t in zip(*dl.values())]


def ld2dl(ld):
    """
    Convert a list of dictionaries to a dictionary of lists
    Example:
        ld = [{'a': 0, 'b': 2}, {'a': 1, 'b': 3}]
        dl = {'a': [0, 1], 'b': [2, 3]}
    """
    return {k: [dic[k] for dic in ld] for k in ld[0]}
