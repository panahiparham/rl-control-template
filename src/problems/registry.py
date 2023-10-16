from importlib import import_module

def getProblem(name):
    if name in ['LongRandChain', 'LongBiasedChain', 'NoisyLongBiasedChain', 'NoisyLongRandChain', 'NoisyShortBiasedChain', 'NoisyShortRandChain', 'ShortBiasedChain', 'ShortRandChain']:
        mod = import_module('problems.RandomWalk')
    else:
        mod = import_module(f'problems.{name}')

    return getattr(mod, name)
