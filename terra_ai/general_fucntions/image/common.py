def make_preprocess(preprocess_list: list):
    def fun(x):
        for prep in preprocess_list:
            x = prep(x)
        if len(x.shape) == 3:
            x = x.reshape((1, *x.shape))
        return x
    return fun
