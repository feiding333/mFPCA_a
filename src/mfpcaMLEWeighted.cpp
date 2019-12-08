#include "mfpcaMLEWeighted.hpp"

//' Initialize the MFPCA loss function. 
//' The only requirement is the sample groups are ordered.
//' 
//' @param y_ All the stacked observation points. The mean curve has been subtracted.
//' @param bMatLarge The stacked basis matrix
//' @param sigmaSq_ The standard variance of the measurement error
//' @param sCount The number of obs points for each obs group
mfpcaMLEWeighted::mfpcaMLEWeighted(vec y_, mat bMatLarge,
                   double sigmaSq_, vec sCount_){
    
    sCount = sCount_;
    totalS = sCount.size();
    
    int startI = -1, endI = -1;
    double m_n = 0;
    vec yn;
    mat Bn, Zn, BnTBn;
    
    weights = ones<vec>(totalS);
    totalDF = bMatLarge.n_cols;
    BtZBSum = mat(totalDF, totalDF, fill::zeros);
    sigmaSqLog = std::log(sigmaSq_);
    nTotal = 0;
    for(int n=0; n < totalS; n++){
        m_n = sCount(n);
        nTotal += m_n;
        startI = endI + 1;
        endI = startI + m_n - 1;
        
        yn = y_(span(startI, endI));
        Bn = bMatLarge.rows(startI, endI);
        
        BmatList.push_back(Bn);
        yVecList.push_back(yn);
        
        
        // For initial computation with least square loss
        // and euclidean geometry.
        // Only compute with initial value of sigmaSq
        BnTBn = Bn.t() * Bn / m_n;
        BtBList.push_back(BnTBn);
        // In the document Zn = yn * yn.t() - sigmaSq I
        Zn =  yn * yn.t();
        Zn.diag() -= sigmaSq_;
        BtZBSum += Bn.t() * Zn * Bn / (m_n * m_n);
    }
    
}// MFPCALoss



// Weights only need to merge as part of m_n
void mfpcaMLEWeighted::setNewZ(double zNew_){

    vec diff = zCovariate - zNew_;
    weights = exp(abs(diff) * h);
    
    double m_n = 0, sigmaSq;
    vec yn;
    mat Bn, Zn, BnTBn;
    

    sigmaSq = exp(sigmaSqLog);
    BtZBSum = mat(totalDF, totalDF, fill::zeros);
    
    for(int n=0; n < totalS; n++){
        m_n = sCount(n) / sqrt(sigmaSq);
        yn = yVecList.at(n);
        Bn = BmatList.at(n);
        
        
        // For initial computation with least square loss
        // and euclidean geometry.
        // Only compute with initial value of sigmaSq
        BnTBn = Bn.t() * Bn / m_n;
        BtBList.at(n) = BnTBn;
        
        // In the document Zn = yn * yn.t() - sigmaSq I
        Zn =  yn * yn.t();
        Zn.diag() -= sigmaSq;
        BtZBSum += Bn.t() * Zn * Bn / (m_n * m_n);
    }
    
}

// Input list: UWUt = U X W
// Output: value of objective function
double mfpcaMLEWeighted::objF(List UWUt){
    arma::mat U, W;
    U = as<arma::mat>(UWUt(0));
    W = as<arma::mat>(UWUt(1));
    double loss = 0;
    mat  Pi;
    
    // try{
    // Accumulate loglikelihood
    for(int i = 0; i < totalS; i++){
        Pi = computePi(U, W, i);
        loss += computeLogliki(Pi, i) * weights(i);
    }
    // }catch(const std::exception& e){
    //     //vec eigval = eig_sym(W);
    //     Rcpp::Rcerr <<  " Obj "  << endl;
    //     //throw(e);
    // }
    
    // penalty
    //loss += mu1 * trace(W);
    loss += mu2 * dot(U, Gamma*U);
    return loss;
}

// Input list: UWUt = U X W
// Output list: gradient w.r.t U and W respectively.
List mfpcaMLEWeighted::gradF(List UWUt){
    arma::mat U, W, comp1, comp2;
    U = as<arma::mat>(UWUt(0));
    W = as<arma::mat>(UWUt(1));
    
    arma::mat coreGrad, grad1, grad2;
    coreGrad = zeros(totalDF, totalDF);
    
    mat  Pi;
    //try{
        for(int i = 0; i < totalS; i++){
            Pi = computePi(U, W, i);
            // Accumulate the negative summand in K
            coreGrad += computeKSummand(Pi, i) * weights(i);
        }
    // }catch(const std::exception& e){
    //     vec eigval = eig_sym(W);
    //     Rcpp::Rcerr << eigval.max()  << " Grad " << eigval.min() << endl;
    // }
    
    coreGrad *= -1;
    //coreGrad /= totalS;
    grad2 = coreGrad * U;
    grad1 = 2 * grad2 * W;
    grad2 = U.t() * grad2;
    
    grad1 += (2*mu2) * Gamma * U;
    //grad2.diag() += mu1;
    
    List gradL = List::create(grad1, grad2);
    return gradL;
}


        
// Objective for Euclidean S withouth penalty.
double mfpcaMLEWeighted::objF_Euc(mat S){
    // double loss = 0;
    // summation of negative log-likelihood
    // mat  Pi;
    // for(int i = 0; i < totalS; i++){
    //     Pi = computePi(S, i);
    //     loss += computeLogliki(Pi, i);
    // }
    // return loss;
    
    // for the initialization, we can not ensure the positive
    // definteness of the matrix, so we use least square loss
    // Only compute with the initial value of sigmaSq
    
    arma::mat BBSBB;
    double loss = 0;
    // quadratic loss
    for(int i = 0; i < totalS; i++){
        BBSBB =  BtBList.at(i) * S * BtBList.at(i);
        loss += dot(BBSBB, S);
    }
    loss -= 2.0 * dot(BtZBSum, S);
    loss /= (2*totalS);
    return loss;
    
}


// Gradient for Euclidean S withouth penalty.
mat mfpcaMLEWeighted::gradF_Euc(mat S){
    // mat  gradS, Pi;
    // gradS = zeros(totalDF, totalDF);
    // for(int i = 0; i < totalS; i++){
    //     Pi = computePi(S, i);
    //     // Compute the negative of the summand in K
    //     gradS += computeKSummand(Pi, i);
    // }
    // return -gradS;
    
    arma::mat gradS;
    gradS = - BtZBSum;
    
    for(int i = 0; i < totalS; i++)
        gradS +=  BtBList.at(i) * S * BtBList.at(i);
    
    gradS /= totalS;
    return gradS;
}


