__cache__ = {}


def cached(func):
    def __cached_function(*args, **kwargs):
        key = (func.__name__, args[0], hash(tuple(kwargs.items())))
        if key not in __cache__:
            res = func(*args, **kwargs)
            __cache__[key] = res
        else:
            res = __cache__[key]

        return res

    return __cached_function
