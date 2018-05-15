import numpy as np
from Distributions import InvStdCumNormal


def RN(LowDiscNumbers):
    '''
    Return Random Normal RVs.
    '''
    LowDiscU = LowDiscNumbers.Generate()
    IndptXtilda = np.fromiter(map(lambda u: InvStdCumNormal(u),LowDiscU),dtype=np.float)
    return IndptXtilda
