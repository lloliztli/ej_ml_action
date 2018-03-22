#__________________________________1___________________________________________________
from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remueve la media
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)          
    eigValInd = eigValInd[:-(topNfeat+1):-1]  
    redEigVects = eigVects[:,eigValInd]       #Ordenar los valores propios de mayor a menor
    lowDDataMat = meanRemoved * redEigVects   #Transformar datos en nuevas dimensiones
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

#_______________________________________2______________________________________________
def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #Calcular la media de los valores que no son NaN
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #Sustituir cualquier valor NaN por la media
    return datMat
