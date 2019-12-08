#include <RcppArmadillo.h>
#include <vector>
using namespace Rcpp;
using namespace arma;
using std::vector;

class EMFPCA {
public:
    //' The core computing function. Estimate the mean function 
    //' and eigenfunctions by the EM algorithm.
    //' 
    //' @param epsilon convergence accuracy of the algorithm
    //' @param iterMax the max number of iterations
    void emCompute(double epsilon, int iterMax);
    
    //' Set data for MFPCA estimation
    //' 
    //' @param YiVec_ a list of observed values
    //' @param BiMat_ a list of basis matrices
    void setData(List YiVec_, List BiMat_);
    
    //' Set Penalty
    void setPenalty(mat Omega_, double mu1_, double mu2_);
    
    //' Initialize 
    void initFPC(mat Theta_, vec Ddiag_, double sigmaSqHat_);
    void initMeanFun(vec theta_mu_);
    
    
    
    //' Get results
    vec getThetaMu(){ return theta_mu;}
    mat getThetaFPC(){ return Theta;}
    
private:
    // -----------  Raw Data ---------------
    vector<vec> YiVec;
    vector<mat> BiMat;
    
    // ------------- Derived Data ------------
    // nSample number of samples
    // nTotal number of total observation points
    // splineDF the degree of freedom of the spline basis
    // nComp the number of principal components
    int nSample, nTotal, splineDF, nComp;
    vector<vec> BiTYi; // B_i^T*Y_i
    vector<mat> BiTBi; // B_i^T*B_i
    mat BiTBiSum; // summation of  B_i^T*B_i
    vec BiTYiSum; // summation of B_i^T*Y_i
    
    // ---------  Algorithm Parameters ------------
    bool computeMu; // should we compute the mean function?
    double lambda1, lambda2; //tuning parameters for mean and FPC smoothness
    
    // ------------- Core Parameters -------------
    mat Omega; // matrix for smoothness penalty
    mat Theta; // Orthonormal matrix for the FPC functions
    vec theta_mu; // parameter vector for the mean function
    vec Ddiag, DdiagInv; // variance of the scores
    double sigmaSqHat; //sigma^2 of measurement error
    vector<vec> alphaIHat; // the score of the i-th sample
    vector<mat> alphaICov; // the conditional variance
    vector<mat> alphaI2M; // the conditional second moment
    
    // ------------ Updating Functions ------------
    void algorithmInit();
    void EstepUpdate();
    void SigmaUpdate();
    void UpdateMu();
    void UpdateTheta();
    void UpdateThetaColumn(int j);
};
