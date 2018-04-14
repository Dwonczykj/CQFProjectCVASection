import math
import numpy as np

def ComputeCVA(ExposureArr,PD_dic,DFFn,Tenors,Tenor,Maturity,R):
    if not len(ExposureArr) == len(Tenors):
        raise IndexError("Tenors not same length as MTMs")
    MidExposure = np.zeros(shape=(len(ExposureArr)-1),dtype=np.float)
    MidTenors = np.zeros(shape=(len(Tenors)-1),dtype=np.float)
    CVA_Adj = np.zeros(shape=(len(Tenors)-1),dtype=np.float)
    for i in range(len(MidExposure)):
        MidExposure[i] = (ExposureArr[i+1] + ExposureArr[i]) * 0.5
        MidTenors[i] = (Tenors[i+1] + Tenors[i]) * 0.5
        #todo: Check why PD default is not using the default over the period, herer it is using the default until the mid, this is wrong
        #todo: and should be just for that payment date period gap
        CVA_Adj[i] = MidExposure[i] * DFFn(MidTenors[i]) * (1-R) * Tenors[i+1]

    return CVA_Adj
