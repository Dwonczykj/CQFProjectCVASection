import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import math
from CumulativeAverager import CumAverage
from scipy.interpolate import interp1d
from scipy import interpolate

def plot_DefaultProbs(x,y,name,legendArray):
    fig = plt.figure(figsize=(7.5, 4.5))
    fig.canvas.set_window_title(name)
    fig.canvas.figure.set_label(name)
    noOfLines = len(y)
    i = 0
    lines = []
    while i < noOfLines:
        plt.plot(x, y[i], color=colorPicker(i), linewidth=2)
        lines.append(mlines.Line2D([],[],color=colorPicker(i),label=legendArray[i]))
        i += 1
    
    
    plt.xlabel('Equity Volatility')
    plt.ylabel('PD')
    plt.legend(bbox_to_anchor=(0.65, 1), loc=2, borderaxespad=0.,handles=lines)
    plt.tight_layout()
    #plt.show()

def showAllPlots():
    plt.show()

def plot_codependence_scatters(dataDic,xlabel):
    keys = list(dataDic.keys())
    ln = len(keys)
    nPlt = math.factorial(ln-1)
    n = nPlt
    #keep on dividing by 2 with no remainder until it is not possible:
    i = 0
    if not( n % 2 == 0) and n > 2:
        n -= 1
    while n % 2 == 0 and n > 1:
        i += 1
        n >>= 1
    
    numCols = int(nPlt / i)
    if nPlt % i > 0:
        numCols += 1
    numRows = i
    j = 0
    for j1 in range(0,ln):
        for j2 in range(j1+1,ln):
            key1 = keys[j1]
            key2 = keys[j2]
            return_scatter(dataDic[key1],dataDic[key1],"%s vs %s" % (key1,key2),j+1,numCols,numRows,xlabel)
            j += 1

def return_scatter(xdata,ydata,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = ""):
    ''' Plots a scatter plot showing any co-dependency between 2 variables. '''
    if numberPlot==1:
        fig = plt.figure(figsize=(10, 6))
        fig.canvas.set_window_title(name)
        fig.canvas.figure.set_label(name)
    plt.subplot(noOfPlotsW,noOfPlotsH,numberPlot)
    x = np.linspace(min(xdata), max(xdata), 100)
    plt.xlabel(xlabel)
    plt.ylabel('frequency/probability')
    plt.title(name[:70] + '\n' + name[70:])
    plt.grid(True)
    plt.scatter(x=xdata,y=ydata)

def Plot_Converging_Averages(ArrOfArrays,baseName):
    ln = len(ArrOfArrays[0])
    #keep on dividing by 2 with no remainder until it is not possible:
    i = 0
    if not( ln % 2 == 0) and ln > 2:
        ln -= 1
    while ln % 2 == 0 and ln > 1:
        i += 1
        ln >>= 1
    ln = len(ArrOfArrays[0])
    numCols = int(ln / i)
    if ln % i > 0:
        numCols += 1
    numRows = i
    for j in range(0,ln):
        Plot_Converging_Average(CumAverage(np.asarray([item[j] for item in ArrOfArrays],dtype=np.float)),"%s %dth to default" % (baseName,j+1),j+1,numCols,numRows)

def Plot_Converging_Average(data,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1):
    ''' Plots a line plot showing the convergence against iterations of the average. '''
    if numberPlot==1:
        fig = plt.figure(figsize=(10, 6))
        fig.canvas.set_window_title(name)
        fig.canvas.figure.set_label(name)
    plt.subplot(noOfPlotsW,noOfPlotsH,numberPlot)
    x = np.linspace(0, len(data), 100)
    plt.xlabel("Number of iterations")
    plt.ylabel('Average')
    plt.title(name[:70] + '\n' + name[70:])
    plt.grid(True)
    y = dN(x, np.mean(data), np.std(data))
    plt.plot(x, y, linewidth=2)


def return_Densitys(datatable,name,legendArray,noOfLines):
    ''' Plots Normal PDFs on an array of returns. '''
    fig = plt.figure(figsize=(7.5, 4.5))
    fig.canvas.set_window_title(name)
    fig.canvas.figure.set_label(name)
    
    #plt.hist(np.array(data), bins=50, normed=True)
    i = 0
    if type(datatable) is pd.DataFrame:
        dt0 = np.asarray(datatable[0].dropna())
    else:
        dt0 = datatable[0]
    x = np.linspace(min(dt0), max(dt0), 100)
    lines = []
    while i < noOfLines:
        if type(datatable) is pd.DataFrame:
            dti = np.asarray(datatable[i].dropna())
        else:
            dti = datatable[i]
        y = dN(x, np.mean(dti), np.std(dti))
        plt.plot(x, y,color=colorPicker(i,True), linewidth=2)
        lines.append(mlines.Line2D([],[],color=colorPicker(i,True),label=legendArray[i]))
        i += 1
    plt.xlabel('Discounted Payoff')
    plt.ylabel('frequency/probability')
    plt.legend(bbox_to_anchor=(0.65, 1), loc=2, borderaxespad=0.,handles=lines)
    ' \n '.join()
    plt.title(name[:70] + '\n' + name[70:])
    
    plt.grid(True)
    plt.tight_layout()#rect=[0, 0, 0.7, 1]

def dN(x, mu, sigma):
    ''' Probability density function of a normal random variable x.
    Parameters
    ==========
    mu : float
        expected value
    sigma : float
        standard deviation
    Returns
    =======
    pdf : float
        value of probability density function
    '''
    z = np.zeros(shape=x.shape) if sigma == 0 else (x - mu) / sigma
    pdf = np.zeros(shape=x.shape) if sigma == 0 else np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi * sigma ** 2)
    return pdf

def dExp(x, rate):
    ''' Probability density function of an Exponential random variable x.
    Parameters
    ==========
    rate : float
        rate of exponential distn
    =======
    pdf : float
        value of probability density function
    '''
    expL = lambda x: 0 if x < 0 else rate * np.exp(-1 * rate * x)
    return np.fromiter(map(expL,x),dtype=np.float)

def dExpForPiecewiselambda(rates, tenors):
    ''' Probability density function of an Exponential random variable x using piecewise constant lambda to display varying rates.
    Parameters
    ==========
    rates : array(float)
        piecewise rates of exponential distn
    =======
    pdf : float
        value of probability density function
    '''
    x = np.linspace(0, max(tenors), 100)
    def rate(s):
        k = 1
        while k <= max(tenors):
            if s <= tenors[k]:
                return rates[k-1]
            k += 1
        return "how is x > than our biggest Tenor"
    expL = lambda x: 0 if x < 0 else rate(x) * np.exp(-1 * rate(x) * x)
    y = np.fromiter(map(expL,x),dtype=np.float)

    fig = plt.figure(figsize=(7.5, 4.5))
    name = "Exponential Distribution for Piecewise constant lambda"
    fig.canvas.set_window_title(name)
    fig.canvas.figure.set_label(name)
    plt.plot(x, y,color=colorPicker(5), linewidth=2)
    lines = [mlines.Line2D([],[],color=colorPicker(5),label="Exponential Distn")]
    plt.xlabel('t')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(0.65, 1), loc=2, borderaxespad=0.,handles=lines)
    plt.tight_layout()
    return y

# histogram of annualized daily log returns
def return_histogram(data,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = ""):
    ''' Plots a histogram of the returns. '''
    if numberPlot==1:
        fig = plt.figure(figsize=(10, 6))
        fig.canvas.set_window_title(name)
        fig.canvas.figure.set_label(name)
    plt.subplot(noOfPlotsW,noOfPlotsH,numberPlot)
    x = np.linspace(min(data), max(data), 100)
    plt.hist(np.array(data), bins=50, normed=True)
    y = dN(x, np.mean(data), np.std(data))
    w = np.zeros(shape=x.shape) if np.mean(data) == 0 else dExp(x, 1 / np.mean(data))
    plt.plot(x, y, linewidth=2)
    plt.plot(x, w, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel('frequency/probability')
    plt.title(name[:70] + '\n' + name[70:])
    plt.grid(True)

def plot_histogram_array(dataDic,xlabel):
    keys = list(dataDic.keys())
    ln = len(keys)
    #keep on dividing by 2 with no remainder until it is not possible:
    i = 0
    if not( ln % 2 == 0) and ln > 2:
        ln -= 1
    while ln % 2 == 0 and ln > 1:
        i += 1
        ln >>= 1
    ln = len(keys)
    numCols = int(ln / i)
    if ln % i > 0:
        numCols += 1
    numRows = i
    for j in range(0,ln):
        key = keys[j]
        return_histogram(dataDic[key],key,j+1,numCols,numRows,xlabel)

def FittedValuesLinear(x,y,ReturnCubicFitted=True,name="",numberOfPointsToReturn=50):
    fig = plt.figure(figsize=(10, 6))
    fig.canvas.set_window_title(name)
    fig.canvas.figure.set_label(name)
    x = np.array(x,dtype=np.float)
    tck = interpolate.splrep(x, y, s=0)
    f_cubic = interp1d(x, y, kind='cubic')   
    f_lin = interp1d(x, y)
    xLin = np.linspace(min(x),max(x),endpoint=True,num=numberOfPointsToReturn)
    yLin = interpolate.splev(xLin,tck,der=0)
    plt.plot(x, y, 'o', xLin, f_lin(xLin), '-', xLin, yLin, '--')
    plt.legend(['data', 'linear', 'cubic'], loc='best')
    plt.title(name[:70] + '\n' + name[70:])
    plt.grid(True)
    if ReturnCubicFitted:
        return f_cubic
    else:
        return f_lin(xLin)

def return_lineChart(x,arrLines,name,numberPlot=1,noOfPlotsW=1, noOfPlotsH=1,xlabel = "",ylabel="",legend=[]):
    ''' Plots all lines on the same chart against x.'''
    if numberPlot==1:
        fig = plt.figure(figsize=(10, 6))
        fig.canvas.set_window_title(name)
        fig.canvas.figure.set_label(name)
    plt.subplot(noOfPlotsW,noOfPlotsH,numberPlot)
    xLin = np.linspace(min(x), max(x), 100)
    #y = dN(x, np.mean(x), np.std(x))
    for y in arrLines:
        plt.plot(x, y, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name[:70] + '\n' + name[70:])
    if len(legend) > 0:
        plt.legend(legend,loc='best')
    plt.grid(True)

def colorPicker(i): #https://matplotlib.org/users/colors.html
    colors = ["aqua","green","magenta","navy","red","salmon","sienna","yellow","olive","orange","chartreuse","coral","crimson","cyan","black","brown","darkgreen","fuchsia","gold","grey","khaki","lavender","pink","purple"]
    return colors[(i % len(colors))]