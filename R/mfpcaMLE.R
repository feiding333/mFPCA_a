## observation ID, n = 1, .. , N
## observation element ID, each obs has Q vector elements 
## observation obsT(time), obsY (value), obsSigma (uncertainty)
## everything sorted by obsID

#' Create MFPCA objective function
#' @param dataF the data frame of the observation data. This should have four columns.
#' obsID vector of observation ID
#' elemID vector of observation vector element ID (Factor, optional)
#' obsT vector of observation time points
#' obsY vector of observation value
#' @param obsSigma double, measurement error SD
#' @param splineObj spline object 
#' @return an object of the class MFPCLoss, 
#' the objective and gradient function can be evaluated.
#' 
#' The elemID is optional. If not specified, the result is univariate FPCA.
createMFPCAObjMLE = function(dataF, splineObj, obsSigma = 1){

    #obsID = as.numeric(as.factor(obsID))
    sortI = sort(dataF$obsID, index.return = TRUE)$ix
    obsID = dataF$obsID[sortI]
    
    elemID = dataF$elemID[sortI]
    obsT = dataF$obsT[sortI]
    obsY = dataF$obsY[sortI]
    # obsSigma = obsSigma[sortI]
    
    ## The number of points for each obs ID
    obsIDCount = as.numeric(table(obsID)) 
    
    bMatLarge = generate_bMatLarge(obsT, elemID, splineObj)
    
    # smoothness penalty
    Omega = splineObj$get_Omega()
    
    # the number of multivariate components
    elemNum = length(unique(elemID))
    Gamma = Omega
    if(elemNum > 1){
        for(e in 2:elemNum)
            Gamma = bdiag(Gamma, Omega)
    }
    Gamma = as.matrix(Gamma)
    
    optObj = new(mfpcaMLEWeighted, obsY, bMatLarge, obsSigma, obsIDCount)
    optObj$set_penaltyMatrix(Gamma)
    return(optObj)
}





#' @param obsData the data frame for the observation data.
#' @param splineObj the spline object for training.
#' @param optRank Select the rank for model fitting.
#' @param mu2 the tuning parameter for smoothness penalty.
#' @param controlList1 the control parameter for the first round of optimization.
#' @param controlList2 the control parameter for the second round of optimization.
#' @param SInit Initial Value
#' @param sigmaSq the initial value for sigma squared
MFPCA_EstimateMLE = function(obsData, splineObj, 
                             optRank, mu2,
                             controlList1 = NULL,
                             controlList2 = NULL,
                             SInit = NULL, sigmaSq = 1){
    tmin = splineObj$getTMin()
    tmax = splineObj$getTMax()
    
    # Check the existence of the column elemID.
    # If not there, this is the univariate FPCA
    if(is.null(obsData$elemID)){
        obsData$elemID = factor(rep(1,length(obsData$obsY)))
    }else{
        obsData$elemID = factor(obsData$elemID)
    }
    
    trainObj = createMFPCAObjMLE(obsData, splineObj, sigmaSq)

    #controlList = list(alpha = 0.5, tol = 1e-4, iterMax = 100)
    
    # The rank penalty is not used.
    mu1 = 0
    trainObj$set_tuningParameter(mu2)
    
    if(is.null(SInit))
        SInit = MFPCA_Initial(trainObj, optRank, controlList1 )
    SFinal = MFPCA_SecondMLE(trainObj, optRank, controlList2, SInit)
    sigmaSq =  1 ##trainObj$get_sigmaSq() XX
    
    # convert matrices SFinal to R functions
    numElem = length(levels(obsData$elemID))
    model = MFPCA_convert(SFinal, splineObj, optRank, numElem)
    model = c(model, 
              list(tmin = tmin, tmax = tmax,
                   SFinal = SFinal, sigmaSq = sigmaSq,
                   numPCA = optRank, numElem = numElem,
                   elemLevels = levels(obsData$elemID)))
    
    return(model)
}



# Second step estimation on the manifold
MFPCA_SecondMLE = function(optObj, optRank, controlList, SInit){
    SEstimate = SInit
    iter = 1
    flag = TRUE
    #sigmaSq = optObj$get_sigmaSq()
    #while(iter < 2 & flag){
        SEstimate = MFPCA_SecondCore(optObj, optRank, controlList, SEstimate)
        #SHat = SEstimate[[1]] %*% SEstimate[[2]] %*% t(SEstimate[[1]])
        #params = c(0.1, 0.618, 0.1, 1)
        #optObj$updateSigmaSq(SEstimate, params)
        #sigmaSqNew = optObj$get_sigmaSq()
        #if(sigmaSqNew < 0) sigmaSqNew = 0
        #diffSigma = abs(sigmaSqNew - sigmaSq) / max(sigmaSq,1)
        #if(diffSigma < 0.01) flag = FALSE
        #sigmaSq = sigmaSqNew
        #iter = iter + 1
        #cat(iter, diffSigma, "\n")
    #}
    return(SEstimate)
}



# Initial value estimation in the Euclidean space
MFPCA_Initial = function(optObj, optRank, controlList){
    #list(alpha = 1e3, sigma = 0.5, tol = 1e-4, iterMax = 30)
    # total DoF, with component number included.
    splineDF = optObj$get_totalDF()
    
    problem = new(manifoldOpt)
    problem$set_euclidean(splineDF, splineDF)
    # The euclidean loss version
    problem$setObjective(optObj$objF_Euc)
    problem$setGradient(optObj$gradF_Euc)
    problem$update_control(controlList)
    problem$solve()
    SInit = problem$get_optimizer()
    SInit_eigen = eigen(SInit)
    UInit = SInit_eigen$vectors[,1:optRank]
    WInit = SInit_eigen$values[1:optRank]
    mm = min(WInit[WInit>0])
    WInit[WInit <= 0] = mm
    XInit = list(UInit, 
                 diag(WInit))
    return(XInit)
}




MFPCA_SecondCore = function(optObj, optRank,  controlList, XInit){
    #list(alpha = 1e-2, sigma = 1e-2, tol = 1e-10,
    #        iterMax = 400, iterSubMax = 30)
    
    splineDF = optObj$get_totalDF()
    problem = new(manifoldOpt)
    problem$select_algorithm("cg")
    problem$set_eigenReg(splineDF, optRank)
    
    # The manifold loss version
    problem$setObjective(optObj$objF)
    problem$setGradient(optObj$gradF)
    problem$initial_point(XInit)
    problem$update_control(controlList)
    problem$solve()
    SFinal = problem$get_optimizer()
    
    return(SFinal)
} 




