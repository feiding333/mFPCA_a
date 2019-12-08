#include "multivariateFPCA.hpp"

//' Initialize the MFPCA loss function. 
//' The only requirement is the sample groups are ordered.
//' 
//' @param y_ All the stacked observation points. The mean curve has been subtracted.
//' @param bMatLarge The stacked basis matrix
//' @param sigma_ The standard deviation of the measurement error
//' @param sCount The number of obs points for each obs group

MFPCALoss::MFPCALoss(vec y_, mat bMatLarge,
                     double sigma_, vec sCount_){
    
    sCount = sCount_;
    double mAve = mean(sCount);
    totalS = sCount.size();
    
    int startI = -1, endI = -1; 
    double m_n = 0, yns;
    vec yn;
    mat Zn, Bn, BnTBn;
    
    totalDF = bMatLarge.n_cols;
    BtZBSum = mat(totalDF, totalDF, fill::zeros);
    BtB_mn2_Sum = mat(totalDF, totalDF, fill::zeros);
    
    sigmaSq = sigma_;
    nTotal = 0;
    cv1 = 0; 
    cv2 = 0;
    for(int n=0; n < totalS; n++){
        m_n = sCount(n);
        nTotal += m_n;
        startI = endI + 1;
        endI = startI + m_n - 1;
        
        yn = y_(span(startI, endI));
        Bn = bMatLarge.rows(startI, endI);
        
        m_n = m_n / mAve;
        yns = sum(yn % yn);
        yntyn.push_back(yns);
        cv1 += yns / (m_n * m_n);
        cv2 += 1.0 / m_n;
        
        Zn =  yn * yn.t();
        Zn.diag() -= sigmaSq*sigmaSq;

        BnTBn = Bn.t() * Bn / m_n;
        BtB.push_back(BnTBn);
        
        BtZBSum += Bn.t() * Zn * Bn / (m_n * m_n);
        BtB_mn2_Sum += BnTBn / m_n;
        
    }
    
    cv1 /= totalS;
    cv2 /= 2 * totalS;
}// MFPCALoss


// Input list: UWUt = U X W
// Output: value of objective function
double MFPCALoss::objF(List UWUt){
    arma::mat U, W, comp1, comp2, UtBtBU;
    U = as<arma::mat>(UWUt(0));
    W = as<arma::mat>(UWUt(1));
    double loss = 0;
    
    // quadratic loss
    comp1 = U.t() * BtZBSum * U;
    for(int i = 0; i < totalS; i++){
        UtBtBU = U.t() * BtB.at(i) * U;
        comp2 = UtBtBU * W * UtBtBU;
        loss += dot(comp2, W);
    }
    loss -= 2.0 * dot(comp1, W);
    loss /= (2.0 * static_cast<double>(totalS));
    
    // penalty
    loss += mu1 * trace(W);
    loss += mu2 * trace(U.t()*Gamma*U);
    return loss;
}

// Input list: UWUt = U X W
// Output list: gradient w.r.t U and W respectively.
List MFPCALoss::gradF(List UWUt){
    arma::mat U, W, comp1, comp2, BtBU;
    U = as<arma::mat>(UWUt(0));
    W = as<arma::mat>(UWUt(1));
    
    arma::mat coreGrad, grad1, grad2;
    coreGrad =  BtZBSum;
    coreGrad.zeros();
    for(int i = 0; i < totalS; i++){
        BtBU =  BtB.at(i) * U;
        comp2 = BtBU * W * BtBU.t();
        coreGrad += comp2;
    }
    coreGrad =  BtZBSum - coreGrad;
    coreGrad *= (-1.0)/static_cast<double>(totalS);
    grad2 = U.t() * coreGrad * U;
    grad2.diag() += mu1;
    grad1 = coreGrad * U * W;

    grad1 += (2*mu2) * Gamma * U;

    List gradL = List::create(grad1, grad2);
    return gradL;
}


// Objective for Euclidean S withouth penalty.
double MFPCALoss::objF_Euc(mat S){
    arma::mat BBSBB;
    double loss = 0;
    // quadratic loss
    for(int i = 0; i < totalS; i++){
        BBSBB =  BtB.at(i) * S * BtB.at(i);
        loss += dot(BBSBB, S);
    }
    loss -= 2.0 * dot(BtZBSum, S);
    loss /= (2*totalS);
    return loss;
}

// Objective for Euclidean S withouth penalty.
double MFPCALoss::objF_EucCV(mat S){
    arma::mat BBSBB;
    double loss = 0;
    // quadratic loss
    for(int i = 0; i < totalS; i++){
        BBSBB =  BtB.at(i) * S * BtB.at(i);
        loss += dot(BBSBB, S);
    }
    loss -= 2.0 * dot(BtZBSum, S);
    loss /= (2*totalS);
    loss -= cv1 * sigmaSq;
    loss += cv2 * sigmaSq * sigmaSq;
    return loss;
}


// Gradient for Euclidean S withouth penalty.
mat MFPCALoss::gradF_Euc(mat S){
    arma::mat gradS;
    gradS = - BtZBSum;

    for(int i = 0; i < totalS; i++)
        gradS +=  BtB.at(i) * S * BtB.at(i);
    
    gradS /= totalS;
    return gradS;
}




//' update the estimation for the measurement error sigma squared
//' Update BtZBSum for the obj/grad functions
void MFPCALoss::updateSigmaSq(mat S){
    double sigmaSqNew = 0;
    double tmp;
    size_t n;
    for(n = 0; n < totalS; n++){
        tmp = yntyn.at(n) - dot(BtB.at(n), S) * sCount(n);
        sigmaSqNew += tmp;
    }
    sigmaSqNew /= nTotal;
    if(sigmaSqNew < 0) sigmaSqNew = 0;
    BtZBSum += (sigmaSqNew - sigmaSq) * BtB_mn2_Sum;
    sigmaSq = sigmaSqNew;
}

