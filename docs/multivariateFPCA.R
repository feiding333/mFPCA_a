### These are a collection of functions for the original squared loss version.

# Second step estimation on the manifold
MFPCA_Second = function(optObj, optRank, controlList, SInit){
    SEstimate = SInit
    iter = 1
    flag = TRUE
    sigmaSq = optObj$get_sigmaSq()
    while(iter < 2 & flag){
        SEstimate = MFPCA_SecondCore(optObj, optRank, controlList, SEstimate)
        SHat = SEstimate[[1]] %*% SEstimate[[2]] %*% t(SEstimate[[1]])
        optObj$updateSigmaSq(SHat)
        sigmaSqNew = optObj$get_sigmaSq()
        if(sigmaSqNew < 0) sigmaSqNew = 0
        diffSigma = abs(sigmaSqNew - sigmaSq) / max(sigmaSq,1)
        if(diffSigma < 0.01) flag = FALSE
        sigmaSq = sigmaSqNew
        iter = iter + 1
        #cat(iter, diffSigma, "\n")
    }
    return(SEstimate)
}


## observation ID, n = 1, .. , N
## observation element ID, each obs has Q vector elements 
## observation obsT(time), obsY (value), obsSigma (uncertainty)
## everything sorted by obsID

#' Create MFPCA objective function
#' @param dataF the data frame of the observation data. This should have four columns.
#' obsID vector of observation ID
#' obsElem vector of observation vector element ID (Factor, optional)
#' obsT vector of observation time points
#' obsY vector of observation value
#' @param obsSigma double, measurement error SD
#' @param splineObj spline object 
#' @return an object of the class MFPCLoss, 
#' the objective and gradient function can be evaluated.
#' 
#' The obsElem is optional. If not specified, the result is univariate FPCA.
createMFPCAObj = function(dataF, splineObj, obsSigma = 0){
    # Only one factor, this is the 
    if(is.null(dataF$obsElem))
        obsElem = factor(rep(1,length(obsY)))
    #obsID = as.numeric(as.factor(obsID))
    sortI = sort(dataF$obsID, index.return = TRUE)$ix
    obsID = dataF$obsID[sortI]
    
    obsElem = dataF$obsElem[sortI]
    obsT = dataF$obsT[sortI]
    obsY = dataF$obsY[sortI]
    # obsSigma = obsSigma[sortI]
    
    ## The number of points for each obs ID
    obsIDCount = as.numeric(table(obsID)) 
    
    bMatLarge = generate_bMatLarge(obsT, obsElem, splineObj)
    
    # smoothness penalty
    Omega = splineObj$get_Omega()
    
    # the number of multivariate components
    elemNum = length(unique(obsElem))
    Gamma = Omega
    if(elemNum > 1){
        for(e in 2:elemNum)
            Gamma = bdiag(Gamma, Omega)
    }
    Gamma = as.matrix(Gamma)
    
    optObj = new(MFPCLoss, obsY, bMatLarge, obsSigma, obsIDCount)
    optObj$set_penaltyMatrix(Gamma)
    return(optObj)
}



MFPCA_Estimate = function(obsCol, splineObj, 
                          optRank, mu1, mu2,
                          controlList1, controlList2,
                          SInit = NULL, sigmaHat = 0){
    tmin = splineObj$getTMin()
    tmax = splineObj$getTMax()
    
    trainObj = createMFPCAObj(obsCol[, "obsID"], obsCol[, "elemID"], 
                              obsCol[, "obsT"], obsCol[, "obsY"], 
                              obsSigma = sigmaHat, splineObj)
    numElem = length(levels(obsCol[,"elemID"]))
    #controlList = list(alpha = 0.5, tol = 1e-4, iterMax = 100)
    trainObj$set_tuningParameter(mu1, mu2)
    if(is.null(SInit))
        SInit = MFPCA_Initial(trainObj, optRank, controlList1 )
    SFinal = MFPCA_Second(trainObj, optRank, controlList2, SInit)
    sigmaSq = trainObj$get_sigmaSq()
    
    # convert matrices SFinal to R functions
    model = MFPCA_convert(SFinal, splineObj, optRank, numElem)
    model = c(model, 
              list(tmin = tmin, tmax = tmax,
                   SFinal = SFinal, sigmaSq = sigmaSq,
                   numPCA = optRank, numElem = numElem,
                   elemLevels = levels(obsCol[,"elemID"])))
    
    return(model)
}
