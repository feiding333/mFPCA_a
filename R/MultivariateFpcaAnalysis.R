# Model object; 

plotFpcaModel = function(pcaModel, selK = NULL){
    tmin = pcaModel$tmin
    tmax = pcaModel$tmax
    numPCA = pcaModel$numPCA
    numElem = pcaModel$numElem
    
    nSeq = 200
    tSeq = seq(tmin, tmax, length.out = nSeq)
    plotData = data.frame();#matrix(0, nSeq * numPCA * numElem, 4)  
    selR = 1:nSeq
    for(k in 1:numPCA){
        for(e in 1:numElem){
            em = pcaModel$elemLevels[e]
            fSeq = pcaModel$eigenFunctions[[k]][[e]](tSeq)
            tmp = data.frame(obsT = tSeq, obsY = fSeq, 
                             pcaID =  k, elemID =  em)
            plotData = rbind(plotData, tmp)
            #selR = selR + nSeq
        }
    }
    colnames(plotData) = c("obsT", "obsY", "pcaID", "elemID")
    plotData$elemID = factor(plotData$elemID, levels = pcaModel$elemLevels)
    

    if(!is.null(selK)){
        plotData = subset(plotData, plotData$pcaID == selK)

        p = ggplot(plotData, aes(obsT, obsY, 
                                 group = elemID, color = elemID)) +
            geom_line()
        p = p + facet_wrap(~elemID)
        
    }else{
        p = ggplot(plotData, aes(obsT, obsY, 
                                 group = elemID, color = elemID)) +
            geom_line()
        p = p + facet_wrap(~pcaID)
        
    }
    return(p)
}


