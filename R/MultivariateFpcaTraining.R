

## Construct a large spline basis matrix
## #Cols: spline degree of freedom X number of components
generate_bMatLarge = function(obsT, elemID, spObj){
    
    uniqueElem = levels(elemID)
    numElem = length(uniqueElem)
    nObs = length(obsT)
    
    # spObj = new(orthoSpline)
    # spObj$set_parameters(4, t_min, t_max, nknots)
    bMat = t(spObj$evalSpline(obsT))
    splineDF = ncol(bMat) # degree of freedom of the spline
    
    bMatLarge = matrix(0, nObs, splineDF * numElem)
    colRange = 1:splineDF
    for(bandI in uniqueElem){
        sel = (elemID == bandI)
        bMatLarge[sel, colRange] = bMat[sel,]
        colRange = colRange + splineDF
    }
    return(bMatLarge)
}




#Input SHat on the manifold and splineObj
#Return eigenvalues and eigen fuction
MFPCA_convert = function(SHat, splineObj, numPCA, numElem){
    splineDF = splineObj$getDoF()
    tmin = splineObj$getTMin()
    tmax = splineObj$getTMax()
    
    eiDec = eigen(SHat[[2]])
    eVector = SHat[[1]] %*% eiDec$vectors
    eValues = eiDec$values
    
    tSeq = seq(tmin, tmax, length.out = 200)
    bMatSeq = t(splineObj$evalSpline(tSeq))
    bMatSeqDeriv1 = t(splineObj$evalSplineDeriv(tSeq, 1))
    bMatSeqDeriv2 = t(splineObj$evalSplineDeriv(tSeq, 2))
    
    eigenFunctions = list()
    eigenFunctionsDeriv1 = list()
    eigenFunctionsDeriv2 = list()
    
    for(k in 1:numPCA){
        selR = 1:splineDF
        compFunctions = list()
        compFunctionsDeriv1 = list()
        compFunctionsDeriv2 = list()
        
        for(e in 1:numElem){
            eFunSeq = bMatSeq %*% eVector[selR, k]
            eFun = approxfun(tSeq, eFunSeq)
            compFunctions = c(compFunctions, list(eFun))
            
            eFunSeqDeriv1 = bMatSeqDeriv1 %*% eVector[selR, k]
            eFunDeriv1 = approxfun(tSeq, eFunSeqDeriv1)
            compFunctionsDeriv1 = c(compFunctionsDeriv1, list(eFunDeriv1))
            
            eFunSeqDeriv2 = bMatSeqDeriv2 %*% eVector[selR, k]
            eFunDeriv2 = approxfun(tSeq, eFunSeqDeriv2)
            compFunctionsDeriv2 = c(compFunctionsDeriv2, list(eFunDeriv2))
            
            selR = selR + splineDF
        }
        eigenFunctions = c(eigenFunctions, list(compFunctions))
        eigenFunctionsDeriv1 = c(eigenFunctionsDeriv1, list(compFunctionsDeriv1))
        eigenFunctionsDeriv2 = c(eigenFunctionsDeriv2, list(compFunctionsDeriv2))
        
    }
    model = list(eigenValues = eValues, 
                 eigenFunctions = eigenFunctions,
                 eigenFunctionsDeriv1 = eigenFunctionsDeriv1,
                 eigenFunctionsDeriv2 = eigenFunctionsDeriv2)
    return(model)
}



# 
# 
# MFPCA_CrossValidation = function(obsCol, sigmaHat, splineObj, 
#                                  optRank, mu1, mu2,
#                                  controlList1, controlList2){
#     l1 = length(mu1)
#     l2 = length(mu2)
#     nTotal = nrow(obsCol)
#     cvGroup = sample(1:10, nTotal, replace = TRUE)
#     
#     testLoss = matrix(0, nrow = l1, ncol = l2)
#     # resLoss = list()
#     for(g in 1:10){
#         sel = (g == cvGroup)
#         trainData = obsCol[!sel, ]
#         testData = obsCol[sel, ]
#         
#         trainObj = createMFPCAObj(obsCol[, "obsID"], obsCol[, "elemID"], 
#                                   obsCol[, "obsT"], obsCol[, "obsY"], 
#                                   obsSigma = sigmaHat, splineObj)
#         #controlList = list(alpha = 0.5, tol = 1e-4, iterMax = 100)
#         SInit = MFPCA_Initial(trainObj, optRank, controlList1 )
#         SFinalCV = SInit
#         sigmaCV = 0
#         
#         for(m1 in 1:l1){
#             sigmaCV = 0
#             SFinalCV = SInit
#             for(m2 in 1:l2){
#                 cat(g, m1, m2, "\n")
#                 flush.console()
#                 
#                 result = MFPCA_Estimate(trainData,  splineObj,
#                                         optRank, mu1[m1], mu2[m2],
#                                         controlList1, controlList2,
#                                         SInit, sigmaCV)
#                 sigmaCV = result$sigmaSq
#                 SFinalCV = result$SFinal
#                 sigmaCV = 0
#                 testObj = createMFPCAObj(testData[, "obsID"], testData[, "elemID"], 
#                                          testData[, "obsT"], testData[, "obsY"], 
#                                          obsSigma = sigmaCV, splineObj)
#                 testObj$set_tuningParameter(0, 0)
#                 SFinalCV = SFinalCV[[1]] %*% SFinalCV[[2]] %*% t(SFinalCV[[1]])
#                 testLoss[m1, m2] = testLoss[m1, m2] + 
#                     testObj$objF_EucCV(SFinalCV)
#             } # for m2
#         } # for m1
#         #resLoss = c(resLoss, list(testLoss))            
#     } # for g
#     return(testLoss)
#     
# }
