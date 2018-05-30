
#Step1 Calculate forwards libor rates and add an interpolater from main project and supply with required tenors to get inbetween rates...
#Step2 Calculate Discount Factors from OIS Curves as in main project and interpolate like above.
#Step3 Calculate prob of defaults from Credit Spreads, take Implimentation from main project.
#todo: NEed to get historic rates data

#!Step4 2nd Bullet point: USE HJM / LMM and calibrate model to 2 years of recent data to Evolve the Swap and hence the MTM of the swap.
#!This will provide us with a separate simulation for each of the forward rates (instead of bootstrapping from OIS) which we can then use in the MTM calc.

#Step5 Produce the Expected Exposure profile.
#Step6 Produce the Expected Exposure distribution for each pymnt date and specific dates in brief; (hist + line) + maybe tests for distns.

#todo: Plot all the different forward curves post mc ie, bootstrap simple, spot, hjm mc, hjm pca .... could also plot distribution of mc process for hjm


import numpy as np
from numpy.random import normal
import math
import pandas as pd
import os
import time
from matplotlib.mlab import PCA as PCAM
from sobol_seq import i4_sobol
from scipy.stats import uniform

from Logger import convertToLaTeX, printf
from HazardRates import GetDFs, InterpolateProbsLinearFromSurvivalProbs, GetHazardsFromP, BootstrapImpliedProbalities, InterpolateProbsLinearFromSurvivalProbs, DiscountFactorFn, ImpProbFn
from BootstrapForwardRates import BootstrapForwardRates
from Returns import AbsoluteDifferences, Cov, PCA, VolFromPCA
from plotting import return_histogram, return_scatter, plot_codependence_scatters, showAllPlots, return_lineChart, SuitableRegressionFit, plot_histogram_array, save_all_figs
from Integration import WeightedNumericalIntFromZero
from LowDiscrepancyNumberGenerators import SobolNumbers
from RandomNumber import RN
from SwapPV import PriceIRS, IRSExposures, GetParSwapRate
from RunningMoments import RunningVarianceOfRunningAverage
from CVA import ComputeCVA

cwd = os.getcwd()

#testCQF_PCA_path = cwd + '/CQF_PCA.xlsm'
#testCQF_MC_path = cwd + '/CQF_MC.xlsm'
#CQF_Differences_sheet = pd.read_excel(testCQF_PCA_path,'Differences',0)
#ObservedFwdPD = pd.read_excel(testCQF_MC_path,'ObservedFwd',0)

#debugTenors = np.array(CQF_Differences_sheet.columns[1:],dtype=np.float)
#ProxyTenors = debugTenors
#ProxyTenors[0] = 0.0
#CQF_testdata = pd.read_excel(testCQF_PCA_path,'Data',0)
#CQF_Differences = CQF_Differences_sheet.values[:,1:]
#CovLiborFwd = Cov(CQF_Differences) * (252) / (math.pow(100,2))
#debugPCA = PCA(CovLiborFwd,3)
##check drifts and volfns
#VolFns = VolFromPCA(debugPCA[0],debugPCA[1])
#FittedVolVec = dict()
#for i in range(0,len(VolFns)):
#    #FittedVolVec[i] = FittedValuesLinear(ProxyTenors,VolFns[i],True,"V_%d"%i,len(VolFns[0]))
#    FittedVolVec[i], regresPow, xLin = SuitableRegressionFit(ProxyTenors,VolFns[i],"V_%d"%i,len(VolFns[1]),3 if i > 0 else 0)
#V = np.zeros(shape=(len(VolFns),len(VolFns[0])))
#for i in range(0,len(VolFns)):
#    for l in range(0,len(ProxyTenors)):
#        V[i,l] = FittedVolVec[i](ProxyTenors[l])
#Drift = np.zeros(shape=(len(VolFns[0]),1))
#for i in range(0,len(VolFns[0])):
#    for j in range(0,len(VolFns)):
#        Drift[i,:] += WeightedNumericalIntFromZero(ProxyTenors[i],FittedVolVec[j])


#return_lineChart(ProxyTenors,[Drift],"Fwd Rate Drift",xlabel="Tenor",ylabel="Fwd Rate Drift from PCA")
#for i in range(0,len(V)):
#    return_lineChart(ProxyTenors,[VolFns[i],V[i]],"Fitted Volatility Function %d"%i,xlabel="Tenor",ylabel="Volatility",legend=["Volatility Function", "Fitted Volatility Function"])






#ObservedFwd = np.array(ObservedFwdPD['ObservedFwdRate'])
#dt = 0.01
#IRSMaturity = 5.0

#NumbGen = SobolNumbers()
#NumbGen.initialise(len(VolFns))

#V = V.transpose()

#SimulatedFwdRates = np.zeros(shape=(int(IRSMaturity / dt),len(ObservedFwd))) #!start with observed forward rates, and evolve each rate in timesteps of dt until the maturity of the contract.
#SimulatedFwdRates[0,:] = ObservedFwd#!simulate all of the forward rates for all 500 time steps
##p1 = np.zeros(shape=(int(IRSMaturity / dt),len(ObservedFwd)))
##p15 = np.zeros(shape=(int(IRSMaturity / dt),len(ObservedFwd)))
#p_diffs = np.zeros(shape=(int(IRSMaturity / dt),len(ObservedFwd)))
#p21=np.zeros(shape=(int(IRSMaturity / dt),len(ObservedFwd)))
##p22=np.zeros(shape=(int(IRSMaturity / dt),3))
##p3=np.zeros(shape=(int(IRSMaturity / dt),len(ObservedFwd)))
##SimulatedFwdRates1 = np.zeros(shape=(int(IRSMaturity / dt),len(ObservedFwd)))
#rng = range(0,len(ObservedFwd)-1)
#rng1 = range(1,len(ObservedFwd))
#for j in range(1,int(IRSMaturity / dt)):
#    #!Evolve the obsvtn dt into the future using Musiela
#    #todo: Divide the below eqn up into parts and see which one is consistently +ve / if the middle component is not introducing enough random noise... (compare to spreadsheet)
#    #p1[j,rng] = SimulatedFwdRates[j-1,rng]
#    #p15[j,rng] = Drift[rng,0]*dt
#    #p = np.array([sum(x) for x in zip(Drift[rng,0]*dt ,(np.matmul(V[rng,:],RN(NumbGen)) * math.sqrt(dt)) )]+[0])
#    #p21[j,rng]= (p[rng1] - p[rng]) * 2 * dt #(np.matmul(V[rng,:],normal(size=(3))) * math.sqrt(dt))
#    #p15[j,rng]= SimulatedFwdRates[j-1,rng]  
#    #p3[j,rng] = np.fromiter(map(lambda xs: sum(xs), p_diffs.transpose()[rng,0:j+1]),dtype=np.float)
#    #SimulatedFwdRates1[j,rng] = np.fromiter(map(lambda xs: sum(xs), p_diffs.transpose()[rng,0:j]),dtype=np.float) + p_diffs[j,rng]
#    RN(NumbGen)
#    p_diffs[j,rng] = np.array([sum(x) for x in zip(Drift[rng,0]*dt ,(np.matmul(V[rng,:],RN(NumbGen)) * math.sqrt(dt)), (SimulatedFwdRates[j-1,rng1] - SimulatedFwdRates[j-1,rng]) * 2 * dt )])
#    SimulatedFwdRates[j,rng] = np.array([sum(x) for x in zip(SimulatedFwdRates[j-1,rng], p_diffs[j-1,rng])])
#    i = len(ObservedFwd)-1
#    SimulatedFwdRates[j,i] = SimulatedFwdRates[j-1,i] + Drift[i,0]*dt + (np.matmul(V[i,:],RN(NumbGen))* math.sqrt(dt)) + (((SimulatedFwdRates[j-1,i] - SimulatedFwdRates[j-1,i-1])/(ProxyTenors[i] - ProxyTenors[i-1])) * dt)
##!Plot the most significant simulated fwd rates.
#SimFwdRateLins = np.zeros(shape=(3,len(SimulatedFwdRates[:,0])),dtype=np.float)
#linnum = 0
#histLedge = []
#pTen1 = np.zeros(shape=(3,len(SimulatedFwdRates[:,0])),dtype=np.float)
#for k in [3,16,44]:#k is most important indices from pca analysis
#    SimFwdRateLins[linnum] = SimulatedFwdRates[:,k]
#    pTen1[linnum] = p21[:,k]
#    histLedge.append("%f Tenor History"%(k/2))
#    linnum += 1
#linnum = 0
#SimFwdXTenors = np.zeros(shape=(4,len(SimulatedFwdRates[0,:])),dtype=np.float)
#pTen = np.zeros(shape=(4,len(p21[0,:])),dtype=np.float)
#for k in [1,125,375,499]:
#    SimFwdXTenors[linnum] = SimulatedFwdRates[k,:]
#    pTen[linnum] = p21[k,:]
#    linnum += 1

#return_lineChart(np.arange(0,int(IRSMaturity / dt),1,dtype=np.int),SimFwdRateLins,"Simulated Historical Forward Rates",xlabel="Day",ylabel="Simulated Forward Rate",legend=histLedge)
#return_lineChart(np.arange(0,int(IRSMaturity / dt),1,dtype=np.int),pTen1,"Simulated DF Comp of Historical Forward Rates",xlabel="Day",ylabel="DF Comp",legend=histLedge)
#return_lineChart(ProxyTenors,SimFwdXTenors,"Simulated Forward Rates Curves",xlabel="Tenor",ylabel="Simulated Forward Rate",legend=["Today","1.25 yrs","3.75 yrs","5 yrs"])
#return_lineChart(ProxyTenors,pTen,"Simulated DF Comp of Rates Curves",xlabel="Tenor",ylabel="DF Comp",legend=["Today","1.25 yrs","3.75 yrs","5 yrs"])
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



Datafp = cwd + '/FinalProjData.xlsx'

LiborSpotRates = pd.read_excel(Datafp,'LiborSpotRates',0)
CqfForwardSim = pd.read_excel(Datafp,'CQFFwdRates',0)
DiscountCurves = pd.read_excel(Datafp,'DiscountCurves',0)
TenorCreditSpreads = pd.read_excel(Datafp,'TenorCreditSpreads',0)
PresentDate = TenorCreditSpreads['PresentDate'][0]

#SpotRates = np.asarray(LiborSpotRates['Spots'],dtype=np.float)
#SpotDates = LiborSpotRates['Dates']




t0 = time.time()
step = (1/12)
RequiredDiscountTenors = np.arange(step,5.0+step,step,dtype=np.float)
RequiredDiscountTenors = [ round(elem, 6) for elem in RequiredDiscountTenors ]
DiscountFactors = dict()
DiscountFactorCurve = dict()
for i in range(0,4):
    IndKey = DiscountCurves.columns[2*i + 1]
    IndKeyDate = IndKey + "_Date"
    DiscountFactors[IndKey] = GetDFs(
        RequiredDiscountTenors, 
        PresentDate, 
        pd.to_datetime(DiscountCurves[IndKeyDate])[DiscountCurves[IndKeyDate].notnull()], 
        DiscountCurves[IndKey][DiscountCurves[IndKey].notnull()]
    )

    DiscountFactorCurve[IndKey] = DiscountFactorFn(DiscountFactors[IndKey])

t1 = time.time()
print("Took %.10f seconds to Get DFs" % (t1 - t0))
DataTenorDic = dict()
ImpProbDic = dict()
InterpolatedImpProbSurvDic = dict()
BootstrappedSurvProbs = dict()
BootstrappedIntraPeriodDefaultProbs = dict()
ImpHazdRts = dict()
InterpolatedImpHazdRts = dict()
InvPWCDF = dict()
InterpolatedImpliedIntraPeriodDefaultProbDic = dict()
for i in range(0,5*5,5):
    IndKey = TenorCreditSpreads['Ticker'][i]
    DataTenorDic[IndKey] = list(TenorCreditSpreads['DataSR'][i:(i+5)] / 1000)
    ImpProbDic[IndKey] = BootstrapImpliedProbalities(0.4,DataTenorDic[IndKey],TenorCreditSpreads.index)
    Tenors = ImpProbDic[IndKey].index
    BootstrappedSurvProbs[IndKey] = ImpProbDic[IndKey]['ImpliedPrSurv']
    BootstrappedIntraPeriodDefaultProbs[IndKey] = ImpProbDic[IndKey]['ImpliedPrDefltIntraPeriod']
    #ImpHazdRts[IndKey] = GetHazardsFromP(BootstrappedSurvProbs[IndKey],Tenors)
    #InvPWCDF[IndKey] = ApproxPWCDFDicFromHazardRates(ImpHazdRts[IndKey],0.01)
    InterpolatedImpProbSurvDic[IndKey] = InterpolateProbsLinearFromSurvivalProbs(BootstrappedSurvProbs[IndKey],RequiredDiscountTenors)
    InterpolatedImpliedIntraPeriodDefaultProbDic[IndKey] = InterpolateProbsLinearFromSurvivalProbs(BootstrappedSurvProbs[IndKey],RequiredDiscountTenors,False) #the jump reflects the change in gradient in different period.
    #for t in InterpolatedImpProbSurvDic[IndKey].keys():
    #    InterpolatedImpliedDefaultProbDic[IndKey][t] = InterpolatedImpProbSurvDic[IndKey][t] if t == InterpolatedImpProbSurvDic[IndKey][t] else InterpolatedImpProbSurvDic[IndKey][t]
    
    #IntBootstrappedSurvProbs = pd.Series(list(InterpolatedImpProbSurvDic[IndKey].values()),index=list(InterpolatedImpProbSurvDic[IndKey].keys()))
    #InterpolatedImpHazdRts[IndKey] = GetHazardsFromP(IntBootstrappedSurvProbs,RequiredDiscountTenors)
    #InterpolatedImpProbFn[IndKey] = ImpProbFn(InterpolatedImpHazdRts[IndKey])
    
t2 = time.time()
print("Took %.10f seconds to Grab Tenor Data and Init Inverse Empirical CDF functions." % (t2 - t1))


#!HEATH JARROW MORTON SECTION




#delta_fwd_rates = AbsoluteReturns(pd.Series(fwd_rates))
#! Obtain Historical Fwd Rates so that we see which FWd Rates have most significance Historically, (PCA) 
#! #and then only simulate these forward rates as opposed to all of them. We use MC sim as each fwd rate is another dimension to the problem.
LiborSpot_Tenors = LiborSpotRates.columns[1:]
noOfRows= len(LiborSpotRates[LiborSpot_Tenors[0]])
LiborFwd_DailyDiffRatesMat = np.zeros(shape=(noOfRows-1,len(LiborSpot_Tenors)))
HistoricalFwdRatesTable = np.zeros(shape=(noOfRows,len(LiborSpot_Tenors)))
for i in range(0,noOfRows):
    HistoricalFwdRatesTable[i,:] = list(BootstrapForwardRates(LiborSpotRates.values[i][1:],LiborSpot_Tenors,PresentDate,True).values())
#todo Plot forward vs spot
#!fwdvsspotlins = [FwdRates[0,:],LiborSpotRates.values[0][1:]]
#!return_lineChart(LiborSpot_Tenors,fwdvsspotlins,"Forward vs Spot T - 3yrs",xlabel="Tenor",ylabel="Rate",legend=["Fwd","Spot"])
#!fwdvsspotlins = [FwdRates[250,:],LiborSpotRates.values[250][1:]]
#!return_lineChart(LiborSpot_Tenors,fwdvsspotlins,"Forward vs Spot T - 2yrs",xlabel="Tenor",ylabel="Rate",legend=["Fwd","Spot"])
#!fwdvsspotlins = [FwdRates[500,:],LiborSpotRates.values[500][1:]]
#!return_lineChart(LiborSpot_Tenors,fwdvsspotlins,"Forward vs Spot T - 1yrs",xlabel="Tenor",ylabel="Rate",legend=["Fwd","Spot"])
#!fwdvsspotlins = [FwdRates[750,:],LiborSpotRates.values[750][1:]]
#!return_lineChart(LiborSpot_Tenors,fwdvsspotlins,"Forward vs Spot T - 0yrs",xlabel="Tenor",ylabel="Rate",legend=["Fwd","Spot"])
#showAllPlots() #Calculation of Principal FwdRates (Components, PCA)
for i in range(0,len(LiborSpot_Tenors)):
    LiborFwd_DailyDiffRatesMat[:,i] = AbsoluteDifferences(pd.Series(HistoricalFwdRatesTable[:,i]),1)

CovLiborFwd = Cov(LiborFwd_DailyDiffRatesMat) * (252)
pdCov = convertToLaTeX(pd.DataFrame(CovLiborFwd, dtype=np.float))
noOFFacs = 3
debugPCA = PCA(CovLiborFwd)
printablePCAEvalues = convertToLaTeX(pd.DataFrame(data=np.array([debugPCA[0]],dtype=np.float), index = ["Eigen values"], dtype=np.float))
printablePCAEvectors = convertToLaTeX(pd.DataFrame(data=debugPCA[1], dtype=np.float))
PCAAnal = PCA(CovLiborFwd,noOFFacs)
#PCAAlt = PCAM(HistoricalFwdRatesTable)
return_lineChart(LiborSpot_Tenors,PCAAnal[1],"Principal Component Analysis",xlabel="Tenor",ylabel="Eigen Vector")
#return_lineChart(LiborSpot_Tenors,PCAAlt.Y[0:noOFFacs],"Principal Component Analysis, Using MatplotLib",xlabel="Tenor",ylabel="Eigen Vector") 

#!Calculate the Volatility Functions of the Prinicapally Significant Fwd Rates. (SHort / Long rates)
#PCAAnal[0] = PCAAlt.s[0:noOFFacs]
#PCAAnal[1] = PCAAlt.Y[0:noOFFacs]
VolFns = VolFromPCA(PCAAnal[0],PCAAnal[1])
ProxiedTenors = np.array(LiborSpot_Tenors,dtype=np.float)
ProxiedTenors[0] = 0.0

FwdRateLins = np.zeros(shape=(noOFFacs,len(HistoricalFwdRatesTable[:,0])),dtype=np.float)
linnum = 0
for k in PCAAnal[3][0:noOFFacs]:
    FwdRateLins[linnum] = HistoricalFwdRatesTable[:,k]
    linnum += 1
linnum = 0
FwdXTenors = np.zeros(shape=(4,len(HistoricalFwdRatesTable[0,:])),dtype=np.float)
for k in [0,250,500,750]:
    FwdXTenors[linnum] = HistoricalFwdRatesTable[k,:]
    linnum += 1

return_lineChart(np.arange(1,noOfRows+1,1),FwdRateLins,"Historical Forward Rates",xlabel="Day",ylabel="Forward Rate")
return_lineChart(LiborSpot_Tenors,FwdXTenors,"Forward Rates Curves",xlabel="Tenor",ylabel="Forward Rate")

t3 = time.time()
print("Took %.10f seconds to conduct PCA" % (t3 - t2))

#todo COnvert VolVectors to fittedVolVectors using cubic spline.
#todo CHeck beta of Cubic Spline using Cum R^2 which is given in PCA ANalysis results.
FittedVolVec = dict()
#def fV0(x):
#    return sum(VolFns[0])/len(VolFns[0])
#FittedVolVec[0] = fV0
for i in range(0,len(VolFns)):
    FittedVolVec[i], regresPow, xLin = SuitableRegressionFit(ProxiedTenors,VolFns[i],"V_%d"%i,len(VolFns[1]),3)
V = np.zeros(shape=(len(VolFns),len(VolFns[0])))
for i in range(0,len(VolFns)):
    for l in range(0,len(ProxiedTenors)):
        V[i,l] = FittedVolVec[i](ProxiedTenors[l]) #the reason that its the same is that cubic still goes through all the points, its just not linear between.
Drift = np.zeros(shape=(len(VolFns[0]),1))
for i in range(0,len(VolFns[0])):
    for j in range(0,len(VolFns)):
        Drift[i,:] += WeightedNumericalIntFromZero(ProxiedTenors[i],FittedVolVec[j])
        #Drift[i,:] += IntegrateNum_Trapezoid(ProxiedTenors[0:i+1],FittedVolVec[j][0:i+1])

return_lineChart(ProxiedTenors,[Drift],"Fwd Rate Drift",xlabel="Tenor",ylabel="Fwd Rate Drift from PCA")
for i in range(0,len(V)):
    return_lineChart(ProxiedTenors,[VolFns[i],V[i]],"Fitted Volatility Function %d"%i,xlabel="Tenor",ylabel="Volatility",legend=["Volatility Function", "Fitted Volatility Function"])

#todo Write a function for the update algo that takes tau as a parameter
ObservedFwd = HistoricalFwdRatesTable[noOfRows-1,:]
dt = 0.01
IRSMaturity = 5.0

NumbGen = SobolNumbers()
NumbGen.initialise(len(VolFns))
#for i in range(0,5000):
#    NumbGen.Generate()

class ImportSobolNumbGen:
    def __init__(self, dim):
        quasi, s = i4_sobol(len(VolFns),self.seed)
        self.seed = s
        
    seed = 0

    def Generate(self):
        quasi, s = i4_sobol(len(VolFns),self.seed)
        self.seed = s
        return quasi
#AltNumbGen = ImportSobolNumbGen(len(VolFns))

#todo We should be using pseudo randomness for Musiela as we WANT UNCORRELATED rvs, the low-disc nos are by definition deliberately correlated in order to more quickly cover the distribution.
class ImportPseudoNumbGen: 
    def Generate(self):
        return uniform.rvs(size=(len(VolFns)))
AltNumbGen = ImportPseudoNumbGen()

#def Musiela(tau):

V = V.transpose()
pdV = convertToLaTeX(pd.DataFrame(V,dtype=np.float),name="Fitted_Volatility_Functions")
#!Use observed drift and vols calculated from VolFn to simulate fwd rates and simulate the Swap.
parFixedRate = GetParSwapRate(DiscountFactorCurve['6mLibor'],Maturity=5,Tenors=ProxiedTenors,Tenor=0.5)
def SimulateFwdRatesAndPriceIRS(NumbGen,returnCharts=False):
    SimulatedFwdRates = np.zeros(shape=(int(IRSMaturity / dt),len(ObservedFwd))) #!start with observed forward rates, and evolve each rate in timesteps of dt until the maturity of the contract.
    SimulatedFwdRates[0,:] = ObservedFwd#!simulate all of the forward rates for all 500 time steps
    p_diffs = np.zeros(shape=(int(IRSMaturity / dt),len(ObservedFwd)))
    p1 = np.zeros(shape=(int(IRSMaturity / dt),len(VolFns)))
    p2 = np.zeros(shape=(int(IRSMaturity / dt),len(VolFns)))
    p3 = np.zeros(shape=(2*int(IRSMaturity / dt),len(VolFns)))
    #p15 = np.zeros(shape=(int(IRSMaturity / dt),len(ObservedFwd)))
    #p21=np.zeros(shape=(int(IRSMaturity / dt),len(ObservedFwd),3))
    #p22=np.zeros(shape=(int(IRSMaturity / dt),3))
    #p3=np.zeros(shape=(int(IRSMaturity / dt),len(ObservedFwd)))
    #SimulatedFwdRates1 = np.zeros(shape=(int(IRSMaturity / dt),len(ObservedFwd)))
    rng = range(0,len(ObservedFwd)-1)
    rng1 = range(1,len(ObservedFwd))
    for j in range(1,int(IRSMaturity / dt)):
        #!Evolve the obsvtn dt into the future using Musiela
        #p1[j] = RN(NumbGen)
        #p2[j] = RNs
        #p3[2*j -1] = p1[j]
        #p3[2*j] = p2[j]
        RNs = normal(size=(3)) #only call this once per iteration.
        p_diffs[j,rng] = np.array([sum(x) for x in zip(Drift[rng,0]*dt ,(np.matmul(V[rng,:],RNs) * math.sqrt(dt)), ((SimulatedFwdRates[j-1,rng1] - SimulatedFwdRates[j-1,rng]) / (ProxiedTenors[rng1] - ProxiedTenors[rng])) * dt )])
        SimulatedFwdRates[j,rng] = np.array([sum(x) for x in zip(SimulatedFwdRates[j-1,rng], p_diffs[j-1,rng])])
        i = len(ObservedFwd)-1
        SimulatedFwdRates[j,i] = SimulatedFwdRates[j-1,i] + Drift[i,0]*dt + (np.matmul(V[i,:],RNs)* math.sqrt(dt)) + (((SimulatedFwdRates[j-1,i] - SimulatedFwdRates[j-1,i-1])/(ProxiedTenors[i] - ProxiedTenors[i-1])) * dt)
        #p15[j,rng] = Drift[rng,0]*dt
        #p21[j,rng]= V[rng,:]
        #p15[j,rng]= SimulatedFwdRates[j-1,rng]  
        #p3[j,rng] = np.fromiter(map(lambda xs: sum(xs), p_diffs.transpose()[rng,0:j+1]),dtype=np.float)
        #SimulatedFwdRates1[j,rng] = np.fromiter(map(lambda xs: sum(xs), p_diffs.transpose()[rng,0:j]),dtype=np.float) + p_diffs[j,rng]
        #p_diffs[j,rng] = Drift[rng,0]*dt +  (np.matmul(V[rng,:],RN(NumbGen)) * math.sqrt(dt)) + ((SimulatedFwdRates[j-1,rng1] - SimulatedFwdRates[j-1,rng])/(ProxiedTenors[rng1] - ProxiedTenors[rng])) * dt 
        #SimulatedFwdRates[j,rng] = SimulatedFwdRates[j-1,rng] + p_diffs[j-1,rng]
        #i = len(ObservedFwd)-1
        #SimulatedFwdRates[j,i] = SimulatedFwdRates[j-1,i] + Drift[i,0]*dt + (np.matmul(V[i,:],RN(NumbGen))* math.sqrt(dt)) + (((SimulatedFwdRates[j-1,i] - SimulatedFwdRates[j-1,i-1])/(ProxiedTenors[i] - ProxiedTenors[i-1])) * dt) 

    #return_scatter(p1[:,0],p1[:,1],"Sobol Numbers: D0 vs D1 (P1)",xlabel="D0",ylabel="D1",numberPlot=1,noOfPlotsW=3,noOfPlotsH=2)
    #return_scatter(p1[:,0],p1[:,2],"Sobol Numbers: D0 vs D2 (P1)",xlabel="D0",ylabel="D2",numberPlot=2,noOfPlotsW=3,noOfPlotsH=2)
    #return_scatter(p1[:,1],p1[:,2],"Sobol Numbers: D1 vs D2 (P1)",xlabel="D1",ylabel="D2",numberPlot=3,noOfPlotsW=3,noOfPlotsH=2)
    #return_scatter(p2[:,0],p2[:,1],"Sobol Numbers: D0 vs D1 (P2)",xlabel="D0",ylabel="D1",numberPlot=4,noOfPlotsW=3,noOfPlotsH=2)
    #return_scatter(p2[:,0],p2[:,2],"Sobol Numbers: D0 vs D2 (P2)",xlabel="D0",ylabel="D2",numberPlot=5,noOfPlotsW=3,noOfPlotsH=2)
    #return_scatter(p2[:,1],p2[:,2],"Sobol Numbers: D1 vs D2 (P2)",xlabel="D1",ylabel="D2",numberPlot=6,noOfPlotsW=3,noOfPlotsH=2)
    #return_scatter(p3[:,0],p3[:,1],"Sobol Numbers: D0 vs D1 (P3)",xlabel="D0",ylabel="D1")
    #return_scatter(p3[:,0],p3[:,2],"Sobol Numbers: D0 vs D2 (P3)",xlabel="D0",ylabel="D2")
    #return_scatter(p3[:,1],p3[:,2],"Sobol Numbers: D1 vs D2 (P3)",xlabel="D1",ylabel="D2")
    #plot_codependence_scatters(dict(enumerate(p3.transpose())),"D%","D%")

    #!Plot the most significant simulated fwd rates.
    SimFwdRateLins = np.zeros(shape=(noOFFacs,len(SimulatedFwdRates[:,0])),dtype=np.float)
    linnum = 0
    histLedge = []
    for k in PCAAnal[3][0:noOFFacs]:#k is most important indices from pca analysis
        SimFwdRateLins[linnum] = SimulatedFwdRates[:,k]
        histLedge.append("%d Tenor History"%k)
        linnum += 1
    linnum = 0
    SimFwdXTenors = np.zeros(shape=(4,len(SimulatedFwdRates[0,:])),dtype=np.float)
    for k in [0,125,375,499]:
        SimFwdXTenors[linnum] = SimulatedFwdRates[k,:]
        linnum += 1
    if returnCharts:
        return_lineChart(np.arange(0,int(IRSMaturity / dt),1,dtype=np.int),SimFwdRateLins,"Simulated Historical Forward Rates",xlabel="Day",ylabel="Simulated Forward Rate",legend=histLedge)
        return_lineChart(ProxiedTenors,SimFwdXTenors,"Simulated Forward Rates Curves",xlabel="Tenor",ylabel="Simulated Forward Rate",legend=["Today","1.25 yrs","3.75 yrs","5 yrs"])
    #MC simulation of SimualteFwdRAtes. Then calculate IRS payments for each and take average.
    fixedRate = parFixedRate
    #UNcommment line below for present values of all payments.
    #PVs = PriceIRS(SimulatedFwdRates,fixedRate,dt,DF=DiscountFactorCurve['Sonia'],Tenors=ProxiedTenors,Tenor=0.5,Maturity=5)
    # Using proxied tenors here as these are the tenors that were used to obtain the fwdrates.
    Notional = 1000000
    Exposures = IRSExposures(SimulatedFwdRates,fixedRate,dt,DF=DiscountFactorCurve['Sonia'],PaymentDateTenors=ProxiedTenors,PaymentTenor=0.5,ExposureTenor=0.5,Notional=Notional,Maturity=5) #the massive exposure is the result of r_float - r_fixed
    return Exposures
M = 500 #todo: Change this to the min of a number of iterations and a converging variance.
M_Min = 50
Tolerance = 0.000001
dummy = SimulateFwdRatesAndPriceIRS(AltNumbGen,True)
IRSExposureRunningAv = np.zeros(shape=(M,len(dummy)))
IRSExposureRunningAv[0,:] = dummy
TenoredExposures = np.zeros(shape=(M,len(dummy)))
PaymentDates = np.arange(0,5.5,0.5)
ExposureDates = np.arange(0, 5 + step, step)
for i in range(1,M):
    TenoredExposures[i,:] = SimulateFwdRatesAndPriceIRS(NumbGen,False)
    IRSExposureRunningAv[i,:] = ((IRSExposureRunningAv[i-1,:] * i) + TenoredExposures[i,:]) / (i+1)
    if i % M_Min == 0 and i > M_Min-1:
        runningVar = RunningVarianceOfRunningAverage(np.array(IRSExposureRunningAv[0:i,:]).transpose(),M_Min)
        supRunningVars = np.fromiter(map(lambda x: max(x),runningVar),dtype=np.float)
        sup = max(supRunningVars)
        if(sup < Tolerance):
            break
#!Incorrect Calculation of this..., exposureDic needs to take into account average of tenor pv at each month of pvs.
#!When redoing this, calculate the Exposure densities at the same time.
ExposureDic = dict()
for i in range(len(dummy)):
    ExposureDic["%f"%PaymentDates[i]] = np.fromiter(map(lambda x: max(x,0),TenoredExposures[:,i]),dtype=np.float) 

exposureOutPercentiles = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]
exposureStats = plot_histogram_array(ExposureDic,"Exposure",75,exposureOutPercentiles)
exposure_stats_keys = list(exposureStats.keys())

#todo: Instead, each exposure (like in spreadsheet) should be the MTM of the swap, check that it is the MTM and not just a swap payment in priceIRS function. then From the distrubitions perform VAR style analysis to get median fuiture exposure and 97% future exposure. Might involve reiterating the rates at the 97%? or might just be the corner ts of forward rates.
#!could actually be an issue as we seem to only price IRS once for each iteration of the sim, should be once at each timestep.
#!return_lineChart(np.arange(0,M,1,dtype=np.int),IRSRunningAv.transpose(),"IRS MTM Running Average",xlabel="Monte Carlo Iteration",ylabel="IRS MTM",legend=PaymentDates)
#!showAllPlots()
#the interesting PCA suggests negative influences too which could be because of the poor interest rates. 
#todo: Look at plots of forward rates for some different tenors.
#todo: Use interpolated monthly periods when we have monthly exposures, currently only calculate the exposure on payment dates. Monthly exposure will change due to changes in forward rate projections.
CVAAdjustment = ComputeCVA(IRSExposureRunningAv[-1],PD_dic=BootstrappedIntraPeriodDefaultProbs['DB CDS EUR'],DFFn=DiscountFactorCurve['Sonia'],Tenors=PaymentDates,Tenor=0.5,Maturity=5,R=0.4)
PaymentDateCVATable = convertToLaTeX(pd.DataFrame(data=[PaymentDates,CVAAdjustment], index = ["Payment Dates", "CVA"], dtype=np.float),"CVA_Table")
#CVAAdjustmentMonthly = ComputeCVA(IRSExposureRunningAv[-1,:],PD_dic=BootstrappedIntraPeriodDefaultProbs['DB CDS EUR'],DFFn=DiscountFactorCurve['Sonia'],Tenors=ExposureDates,Tenor=step,Maturity=5,R=0.4)
#MonthlyCVATable = convertToLaTeX(pd.DataFrame(data=np.array([ExposureDates, CVAAdjustmentMonthly],dtype=np.float), index = ["Monthly Dates", "CVA"], dtype=np.float))
TotalCVA1 = sum(CVAAdjustment)
printf("The cumulative sum of the CVA is: {0}".format(TotalCVA1))
#TotalCVA2 = sum(CVAAdjustmentMonthly) #todo: Summing the Exposure at each date is much higher by about 6 times.
#todo: Investigate the affect of altering K.
Ex_means = [exposureStats[k][0] for k in exposure_stats_keys]
Ex_Sds = [exposureStats[k][1] for k in exposure_stats_keys]
#Ex_percentiles = np.zeros(shape=(1+len(exposureOutPercentiles),len(PaymentDates)))
#Ex_percentiles[0]=PaymentDates
Ex_percentiles = np.array([np.array(exposureStats[k][2]) for k in exposure_stats_keys]).transpose()
ExposureStatsTable = convertToLaTeX(pd.DataFrame(data=[Ex_means,Ex_Sds], index = ["Mean", "Std. Dev."], columns=PaymentDates, dtype=np.float),"Exposure_stats_Table",topLeftCellText="Payment dates")
ExposurePercentileTable = convertToLaTeX(pd.DataFrame(data=Ex_percentiles, columns=PaymentDates, index=["{0}".format(p) for p in exposureOutPercentiles], dtype=np.float),"Exposure_Percentiles_Table",topLeftCellText="Payment dates")
return_lineChart(PaymentDates,[Ex_means,Ex_Sds],"Exposure distribution moments at each payment date",xlabel="Payment date", ylabel="Statistic",xticks=PaymentDates,legend=["Mean", "S.D."])
return_lineChart(PaymentDates,Ex_percentiles,"Exposure distribution percentiles",xlabel="Payment date", ylabel="Statistic", xticks=PaymentDates, legend=["{0}".format(ep) for ep in exposureOutPercentiles],rotateXLabels=45)
print("Enter any key to finish.")

save_all_figs()
userIn = input()
#if userIn == 'p':
#    showAllPlots()
Debug = True


