import numpy as np
from dateutil.relativedelta import relativedelta

def BootstrapForwardRates(spotRates, matchingDatesOrTenors, PresentDate, isTenors=True):
    if not len(spotRates) == len(matchingDatesOrTenors):
        raise ValueError("The number of dates must match the number of spot rates")
    fwdRates = dict()
    if isTenors:
        TenorsFromDatesYrs = matchingDatesOrTenors
    else:
        TenorsFromDatesYrs = np.fromiter(
            map(
                lambda end,start: relativedelta(end, start).years / 1.0 + relativedelta(end, start).months / 12.0 + relativedelta(end, start).days / 365.0 ,
                matchingDatesOrTenors, 
                np.full(fill_value=PresentDate,shape=len(matchingDatesOrTenors))
                ), 
            dtype=float)
    fwdRates[TenorsFromDatesYrs[0]] = spotRates[0]
    for i in range(1,len(spotRates)):
        fwdRates[TenorsFromDatesYrs[i]] = (spotRates[i]*TenorsFromDatesYrs[i] - spotRates[i-1]*TenorsFromDatesYrs[i-1]) / (TenorsFromDatesYrs[i] - TenorsFromDatesYrs[i-1])
    return fwdRates
