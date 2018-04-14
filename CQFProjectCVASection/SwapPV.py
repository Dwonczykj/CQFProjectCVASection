
import math
import numpy as np

def GetParSwapRate(DF,Maturity,Tenors, Tenor):
    floating = 1 - DF(Maturity)
    fixedPaymnts = np.fromiter(map(lambda x: Tenor*DF(x),Tenors),dtype=np.float)
    fixed = sum(fixedPaymnts)
    return floating / fixed

def PriceIRS(FwdsMatrix,r_fix,dt,DF,Tenors,Tenor,Maturity,Notional=1, CurrentTime=0.0):
    #fwds have shape(M/dt,len(tenors))
    #fwds[0,:] is the current observerd Fwd Rates
    if CurrentTime > 5.0 or CurrentTime < 0.0:
        return 0
    rng = np.arange(CurrentTime,5.0,Tenor,dtype=np.float)
    pvpayments = np.zeros(shape=(len(rng)),dtype=np.float)
    pymentNum = 0
    for t in rng:
        pymntTenor = t + Tenor
        r_ft = GetSpot(FwdsMatrix[int(t/dt),:],Tenors,Tenor)
        #!This code could be optimised to only discount float - fixed but we want to be able to see the legs individually.
        pv_flt_pymnt = DF(pymntTenor - CurrentTime) * Notional * r_ft * Tenor
        pv_fix_pymnt = DF(pymntTenor - CurrentTime) * Notional * r_fix * Tenor
        pvpayments[pymentNum] = pv_flt_pymnt - pv_fix_pymnt
        pymentNum += 1
    if CurrentTime == 0:
        print("hey")
    return pvpayments

def GetSpot(FwdRates,Tenors,t):
    pow = 0
    if t <= Tenors[0]:
        return FwdRates[0]

    for i in range(0,len(Tenors)-1):
        if t > Tenors[i+1]:
            pow += FwdRates[i]*(Tenors[i+1] -Tenors[i])
        elif t > Tenors[i]:
            pow += FwdRates[i]*(t - Tenors[i])
    if t > Tenors[len(Tenors)-1]:
        pow += FwdRates[len(Tenors)-1]*(t - Tenors[len(Tenors)-1])
    return pow / t

def IRSExposures(FwdsMatrix,r_fix,dt,DF,PaymentDateTenors,PaymentTenor,ExposureTenor,Maturity,Notional=1): #todochange Tenor to PaymentTenor=0.5 and have an exposuretenor = 1/12 for monthly exposure checkups
    #fwds have shape(M/dt,len(tenors))
    #fwds[0,:] is the current observerd Fwd Rates
    rng = np.arange(0.0,5.0 + ExposureTenor,ExposureTenor,dtype=np.float)

    Exposures = np.zeros(shape=(len(rng)),dtype=np.float)
    pymentNum = 0
    for t in rng:
        pvs = PriceIRS(FwdsMatrix,r_fix,dt,DF,PaymentDateTenors,PaymentTenor,Maturity,Notional,CurrentTime=t)
        Exposures[pymentNum] = max(sum(pvs),0)

        pymentNum += 1


    return Exposures