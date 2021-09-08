def make_preprocess(preprocess_list: list):
    def fun(x):
        for prep in preprocess_list:
            x = prep(x)
        x /= 255
        return x
    return fun
