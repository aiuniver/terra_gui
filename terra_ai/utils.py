import re


def decamelize(camel_case_string: str):
    string = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_case_string)
    return re.sub('([a-z0-9])([A-Z0-9])', r'\1_\2', string).lower()


def camelize(snake_case_string: str):
    if snake_case_string == 'auc':
        return 'AUC'
    if snake_case_string == 'kullback_leibler_divergence':
        return 'KLDivergence'
    if snake_case_string == 'dice_coef':
        return 'DiceCoefficient'
    if snake_case_string == 'logcosh':
        return 'LogCoshError'
    return re.sub('_.', lambda x: x.group()[1].upper(), snake_case_string.title())
