
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
import math
import pandas as pd
import os
import time
from matplotlib.mlab import PCA as PCAM

from Logger import convertToLaTeX, printf
from HazardRates import GetDFs, InterpolateProbsLinearFromSurvivalProbs, GetHazardsFromP, BootstrapImpliedProbalities, InterpolateProbsLinearFromSurvivalProbs, DiscountFactorFn, ImpProbFn
from BootstrapForwardRates import BootstrapForwardRates
from Returns import AbsoluteDifferences, Cov, PCA, VolFromPCA
from plotting import return_histogram, showAllPlots, return_lineChart, FittedValuesLinear, plot_histogram_array
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
#    FittedVolVec[i] = FittedValuesLinear(ProxyTenors,VolFns[i],True,"V_%d"%i,len(VolFns[0]))
#V = np.zeros(shape=(len(VolFns),len(VolFns[0])))
#for i in range(0,len(VolFns)):
#    for l in range(0,len(ProxyTenors)):
#        V[i,l] = FittedVolVec[i](ProxyTenors[l])
#Drift = np.zeros(shape=(len(VolFns[0]),1))
#for i in range(0,len(VolFns[0])):
#    for j in range(0,len(VolFns)):
#        Drift[i,:] += WeightedNumericalIntFromZero(ProxyTenors[i],FittedVolVec[j])

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
def fV0(x):
    return sum(VolFns[0])/len(VolFns[0])
FittedVolVec[0] = fV0
for i in range(1,len(VolFns)):
    FittedVolVec[i] = FittedValuesLinear(ProxiedTenors,VolFns[i],True,"V_%d"%i,len(VolFns[1]))
V = np.zeros(shape=(len(VolFns),len(VolFns[0])))
for i in range(0,len(VolFns)):
    for l in range(0,len(ProxiedTenors)):
        V[i,l] = FittedVolVec[i](ProxiedTenors[l])
Drift = np.zeros(shape=(len(VolFns[0]),1))
for i in range(0,len(VolFns[0])):
    for j in range(0,len(VolFns)):
        Drift[i,:] += WeightedNumericalIntFromZero(ProxiedTenors[i],FittedVolVec[j])
        #Drift[i,:] += IntegrateNum_Trapezoid(ProxiedTenors[0:i+1],FittedVolVec[j][0:i+1])

return_lineChart(ProxiedTenors,[Drift],"Fwd Rate Drift",xlabel="Tenor",ylabel="Fwd Rate Drift from PCA")
for i in range(0,len(V)):
    return_lineChart(ProxiedTenors,[V[i],VolFns[i]],"Fitted Volatility Function %d"%i,xlabel="Tenor",ylabel="Volatility")

#todo Write a function for the update algo that takes tau as a parameter
ObservedFwd = HistoricalFwdRatesTable[noOfRows-1,:]
dt = 0.01
IRSMaturity = 5.0

NumbGen = SobolNumbers()
NumbGen.initialise(len(VolFns))
for i in range(0,5000):
    NumbGen.Generate()

#def Musiela(tau):

V = V.transpose()
pdV = convertToLaTeX(pd.DataFrame(V,dtype=np.float))
#!Use observed drift and vols calculated from VolFn to simulate fwd rates and simulate the Swap.
parFixedRate = GetParSwapRate(DiscountFactorCurve['6mLibor'],Maturity=5,Tenors=ProxiedTenors,Tenor=0.5)
def SimulateFwdRatesAndPriceIRS(NumbGen,returnCharts=False):
    SimulatedFwdRates = np.zeros(shape=(int(IRSMaturity / dt),len(ObservedFwd))) #start with observed forward rates, and evolve each rate in timesteps of dt until the maturity of the contract.
    SimulatedFwdRates[0,:] = ObservedFwd#simulate all of the forward rates for all 500 simulations
    for j in range(1,int(IRSMaturity / dt)):
        #!Evolve the obsvtn dt into the future using Musiela
        rng = range(0,len(ObservedFwd)-1)
        rng1 = range(1,len(ObservedFwd)) 
        SimulatedFwdRates[j,rng] = SimulatedFwdRates[j-1,rng] + Drift[rng,0]*dt + np.matmul(V[rng,:],(RN(V.shape[1],1,NumbGen) * math.sqrt(dt) )) + ((SimulatedFwdRates[j-1,rng1] - SimulatedFwdRates[j-1,rng])/(ProxiedTenors[rng1] - ProxiedTenors[rng])) * dt 
        i = len(ObservedFwd)-1
        SimulatedFwdRates[j,i] = SimulatedFwdRates[j-1,i] + Drift[i,0]*dt + np.matmul(V[i,:],(RN(V.shape[1],1,NumbGen)* math.sqrt(dt))) + ((SimulatedFwdRates[j-1,i] - SimulatedFwdRates[j-1,i-1])/(ProxiedTenors[i] - ProxiedTenors[i-1])) * dt 
    #!Plot the most significant simulated fwd rates.
    SimFwdRateLins = np.zeros(shape=(noOFFacs,len(SimulatedFwdRates[:,0])),dtype=np.float)
    linnum = 0
    for k in PCAAnal[3][0:noOFFacs]:
        SimFwdRateLins[linnum] = SimulatedFwdRates[:,k]
        linnum += 1
    linnum = 0
    SimFwdXTenors = np.zeros(shape=(4,len(SimulatedFwdRates[0,:])),dtype=np.float)
    for k in [0,125,375,499]:
        SimFwdXTenors[linnum] = SimulatedFwdRates[k,:]
        linnum += 1
    if returnCharts:
        return_lineChart(np.arange(0,int(IRSMaturity / dt),1,dtype=np.int),SimFwdRateLins,"Simulated Historical Forward Rates",xlabel="Day",ylabel="Simulated Forward Rate")
        return_lineChart(ProxiedTenors,SimFwdXTenors,"Simulated Forward Rates Curves",xlabel="Tenor",ylabel="Simulated Forward Rate")
    #MC simulation of SimualteFwdRAtes. Then calculate IRS payments for each and take average.
    fixedRate = parFixedRate
    #UNcommment line below for present values of all payments.
    #PVs = PriceIRS(SimulatedFwdRates,fixedRate,dt,DF=DiscountFactorCurve['Sonia'],Tenors=ProxiedTenors,Tenor=0.5,Maturity=5)
    # Using proxied tenors here as these are the tenors that were used to obtain the fwdrates.
    Notional = 1000000
    Exposures = IRSExposures(SimulatedFwdRates,fixedRate,dt,DF=DiscountFactorCurve['Sonia'],PaymentDateTenors=ProxiedTenors,PaymentTenor=0.5,ExposureTenor=0.5,Notional=Notional,Maturity=5)
    return Exposures
M = 500 #todo: Change this to the min of a number of iterations and a converging variance.
M_Min = 50
Tolerance = 0.000001
dummy = SimulateFwdRatesAndPriceIRS(NumbGen,False)
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

plot_histogram_array(ExposureDic,"Exposure")
#todo: Instead, each exposure (like in spreadsheet) should be the MTM of the swap, check that it is the MTM and not just a swap payment in priceIRS function. then From the distrubitions perform VAR style analysis to get median fuiture exposure and 97% future exposure. Might involve reiterating the rates at the 97%? or might just be the corner ts of forward rates.
#!could actually be an issue as we seem to only price IRS once for each iteration of the sim, should be once at each timestep.
#!return_lineChart(np.arange(0,M,1,dtype=np.int),IRSRunningAv.transpose(),"IRS MTM Running Average",xlabel="Monte Carlo Iteration",ylabel="IRS MTM",legend=PaymentDates)
#!showAllPlots()
#the interesting PCA suggests negative influences too which could be because of the poor interest rates. 
#todo: Look at plots of forward rates for some different tenors.
#todo: Use interpolated monthly periods when we have monthly exposures, currently only calculate the exposure on payment dates. Monthly exposure will change due to changes in forward rate projections.
CVAAdjustment = ComputeCVA(IRSExposureRunningAv[-1,0:61:6],PD_dic=BootstrappedIntraPeriodDefaultProbs['DB CDS EUR'],DFFn=DiscountFactorCurve['Sonia'],Tenors=PaymentDates,Tenor=0.5,Maturity=5,R=0.4)
PaymentDateCVATable = convertToLaTeX(pd.DataFrame(data=np.array([PaymentDates,CVAAdjustment],dtype=np.float), index = ["Payment Dates", "CVA"], dtype=np.float))
#CVAAdjustmentMonthly = ComputeCVA(IRSExposureRunningAv[-1,:],PD_dic=BootstrappedIntraPeriodDefaultProbs['DB CDS EUR'],DFFn=DiscountFactorCurve['Sonia'],Tenors=ExposureDates,Tenor=step,Maturity=5,R=0.4)
#MonthlyCVATable = convertToLaTeX(pd.DataFrame(data=np.array([ExposureDates, CVAAdjustmentMonthly],dtype=np.float), index = ["Monthly Dates", "CVA"], dtype=np.float))
TotalCVA1 = sum(CVAAdjustment)
#TotalCVA2 = sum(CVAAdjustmentMonthly) #todo: Summing the Exposure at each date is much higher by about 6 times.
#todo: Investigate the affect of altering K.

Debug = True


