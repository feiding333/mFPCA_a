#include "mFPCA-EM.hpp"

void EMFPCA::setData(List YiVec_, List BiMat_){
    nSample = BiMat_.length();
    int iterI;
    mat tmp;
    nTotal = 0;
    for(iterI = 0; iterI < nSample; iterI++){
        BiMat.push_back(as<arma::mat>(BiMat_(iterI)));
        YiVec.push_back(as<arma::vec>(YiVec_(iterI)));
        tmp = BiMat.at(iterI).t() * YiVec.at(iterI);
        BiTYi.push_back(tmp);
        tmp = BiMat.at(iterI).t() * BiMat.at(iterI);
        BiTBi.push_back(tmp);
        nTotal += YiVec.at(iterI).n_rows;
    }
    
    splineDF = BiMat.at(0).n_cols;
    BiTBiSum = mat(splineDF, splineDF, fill::zeros);
    BiTYiSum = mat(splineDF, 1, fill::zeros);
    for(iterI = 0; iterI < nSample; iterI++){
        BiTBiSum += BiTBi.at(iterI);
        BiTYiSum += BiTYi.at(iterI);
    }
    
}


void EMFPCA::setPenalty(mat Omega_, double lambda1_,
                        double lambda2_){
    Omega = Omega_;
    lambda1 = lambda1_;
    lambda2 = lambda2_;
}




void EMFPCA::emCompute(double epsilon, int iterMax) {
    bool flag;
    int iterI = 0;
    mat Theta_Null = Theta;
    double diffMax;
    flag = true;
    
    algorithmInit();
    
    while (flag && iterI < iterMax) {
        // Update
        EstepUpdate();
        SigmaUpdate();
        if (computeMu) UpdateMu();
        UpdateTheta();
        
        // Check Convergence
        Theta_Null -= Theta;
        diffMax = norm(Theta_Null, "frob"); // ??
        if(diffMax < epsilon * norm(Theta, "frob")) // ??
            flag = false;
        iterI++;
        Theta_Null = Theta;
    }
}

void EMFPCA::algorithmInit(){
    if(!computeMu)
        theta_mu = zeros<vec>(splineDF);
}




// E-step, compute the conditional mean, 
// variance and second moment of the scores
void EMFPCA::EstepUpdate() {
    int i;
    mat tmp1, tmp2;
    for (i = 0; i < nSample; i++) {
        if(computeMu)
            tmp1 = Theta.t() * (BiTYi.at(i) - BiTBi.at(i) * theta_mu);
        else 
            tmp1 = Theta.t() * BiTYi.at(i) ;
        tmp2 = Theta.t() * BiTBi.at(i) * Theta;
        tmp2.diag() += sigmaSqHat * DdiagInv;
        alphaIHat.at(i) = solve(tmp2, tmp1);
        alphaICov.at(i) = inv(tmp2) * sigmaSqHat;
        alphaI2M.at(i) = alphaICov.at(i) + alphaIHat.at(i) * alphaIHat.at(i).t();
    }
}


void EMFPCA::SigmaUpdate() {
    vec resid;
    int i;
    mat tmp;
    
    sigmaSqHat = 0.0;
    Ddiag = vec(nComp, fill::zeros);
    for (i = 0; i < nSample; i++) {
        resid = YiVec.at(i) - BiMat.at(i) * (theta_mu + 
            Theta * alphaIHat.at(i));
        sigmaSqHat += sum(resid % resid);
        tmp = Theta.t() * BiTBi.at(i) * Theta;
        sigmaSqHat += dot(tmp,  alphaICov.at(i));
        Ddiag += alphaI2M.at(i).diag();
    }
    sigmaSqHat /= nTotal;
    Ddiag /= nSample;
    DdiagInv = 1.0 / Ddiag;
}


void EMFPCA::UpdateMu() {
    // Step 2-1, Update theta_mu
    int i;
    theta_mu = BiTYiSum;
    for (i = 0; i < nSample; i++) {
        theta_mu -= BiTBi.at(i) * Theta * alphaIHat.at(i);
    }
    mat tleft = BiTBiSum + nSample * lambda1 * Omega;
    theta_mu = solve(tleft, theta_mu);
}


void EMFPCA::UpdateTheta() {
    // Step 2-2, Update Theta
    int j;
    bool flag = true;
    mat Theta_Null = Theta;
    double diffNorm;
    int iter = 0;
    while (flag && iter < 1000) {
        for (j = 0; j < nComp; j++) {
            UpdateThetaColumn(j);
        } // for j
        
        // check convergence
        Theta_Null -= Theta;
        diffNorm = norm(Theta_Null, "frob");
        // compute frobenius norm ??
        if(diffNorm < 0.1 * norm(Theta, "frob")){
            flag = false;
        }
        Theta_Null = Theta;
        iter++;
    }
    
    // Orthonormalize the columns of Theta 
    mat Q, R;
    qr_econ(Q, R, Theta);
    Theta = Q;
}


void EMFPCA::UpdateThetaColumn(int j) {
    int i, k;
    mat BiTBiSumWeighted;
    vec resid_theta;
    double wgts;
    
    BiTBiSumWeighted = mat(splineDF, splineDF, fill::zeros);
    resid_theta = vec(splineDF, fill::zeros);
    for (i = 0; i < nSample; i++) {
        wgts = alphaI2M.at(i)(j, j);
        BiTBiSumWeighted += wgts * BiTBi.at(i);
        if(computeMu)
            resid_theta += (BiTYi.at(i) - BiTBi.at(i) * theta_mu) *
                alphaIHat.at(i)(j);
        else
            resid_theta += BiTYi.at(i) * alphaIHat.at(i)(j);
        
        for (k = 0; k < nComp; k++){
            if (k == j) continue;
            resid_theta -= BiTBi.at(i) * Theta.col(k) * alphaI2M.at(i)(j, k);
        } // for k
    }// for i
    BiTBiSumWeighted += lambda2 * nSample * Omega;
    Theta.col(j) = solve(BiTBiSumWeighted, resid_theta);
}




void EMFPCA::initFPC(mat Theta_, vec Ddiag_, double sigmaSqHat_){
    Theta = Theta_;
    Ddiag = Ddiag_;
    sigmaSqHat = sigmaSqHat_;
    
    nComp = Theta.n_cols;
    DdiagInv = 1 / Ddiag;
    int i;
    
    for(i = 0; i < nSample; i++){
        alphaIHat.push_back(vec(nComp, fill::zeros));
        alphaICov.push_back(mat(nComp, nComp, fill::zeros));
        alphaI2M.push_back(mat(nComp, nComp, fill::zeros));
    }
}

void EMFPCA::initMeanFun(vec theta_mu_){
    if(theta_mu_.n_elem != splineDF)
    computeMu = true;
    theta_mu = theta_mu_;
}
