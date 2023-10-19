from typing import Type
from algorithms.BaseAgent import BaseAgent

from algorithms.nn.TD import TD
from algorithms.nn.TDRC import TDRC


def getAgent(name) -> Type[BaseAgent]:
    if name == 'TD':
        return TD
    
    if name == 'TDRC':
        return TDRC

    raise Exception('Unknown algorithm')
