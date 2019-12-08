#ifndef MFPCAMLE_CLASS
#define MFPCAMLE_CLASS

#include <RcppArmadillo.h>
#include <vector>
#include <cmath>
using namespace Rcpp;
using namespace arma;

class mfpcaMLE{
public:
    
    // Ordered by sample n = 1, ..., N
    // bMatLarge, sparse b-spline matrix, band * df_spline columns
    // sCount_, number of obs points in each sample
    // length(y_): total number of obs points
    // length(sCount_): total number of obs
    mfpcaMLE(vec y_, mat bMatLarge, 
             double sigmaSq_, vec sCount_);
    // we might have further speed improvment with refinement of bMatLarge?
    
    
    void set_penaltyMatrix(mat Gamma_){
        Gamma = Gamma_;
    }
    
    void set_tuningParameter(double mu2_){
        //mu1 = mu1_;
        mu2 = mu2_;
    }
    
    int get_totalDF(){ return totalDF;}
    double get_sigmaSq(){return std::exp(sigmaSqLog);}
    
    // the objective and the first order gradient 
    // in the Euclidean space without penalty
    // These two functions are used for initialization
    double objF_Euc(mat S);
    mat gradF_Euc(mat S);

    // on the manifold
    double objF(List UWUt);
    List gradF(List UWUt);
    
    // Params (0) alpha: conservative rate
    // (1) beta: backtracking shrinkage rate
    // (2) epsilon: accuracy
    // (3) verbose: 0, 1, 2 level of details
    void updateSigmaSq(List UWUt, vec params);
    
    
private:
    //  total sample size;
    // total degree of freedom. S and UWUt are both DF-by-DF matrices.
    size_t totalS, totalDF;
    size_t nTotal; // total obs points
    vec sCount;
    //tuning parameters on  smoothness
    double  mu2; //mu1, on rank and
    double sigmaSqLog; // measurement error, log of sigma squared
    // vec sCount;
    std::vector<mat> BmatList;
    std::vector<mat> BtBList;
    std::vector<vec> yVecList;
    mat BtZBSum, Gamma;
    
    vec updateSigmaSqGradient(List UWUt);
    double updateSigmaSqBackTracking(vec res, List UWUt, 
                                     double alpha, double beta,
                                     int verbose);
    
    // Compute the covariance matrix for the i-th observation
    // from Euclidean matrix
    inline mat computePi(const mat& S, const int &i){
        mat Pi;
        Pi = BmatList.at(i) * S * BmatList.at(i).t();
        Pi.diag() += exp(sigmaSqLog);
        return Pi;
    }
    
    // Compute the covariance matrix for the i-th observation
    // from Manifold matrix
    inline mat computePi(const mat& U, const mat& W, const int &i){
        mat Pi;
        Pi = BmatList.at(i) * U;
        Pi = Pi * W * Pi.t();
        Pi.diag() += exp(sigmaSqLog);
        return Pi;
    }
    
    inline double computeLogliki(const mat& Pi, const int &i){
        double mlogLik;
        mat PiChol;
        vec tmpY;
        PiChol = chol(Pi, "lower");
        mlogLik = 2*sum(log(abs(PiChol.diag()) ));
        tmpY = solve(trimatl(PiChol), yVecList.at(i));
        mlogLik += dot(tmpY, tmpY);
        return mlogLik;
    }
    
    // Compute the negative of the summand in K
    inline mat computeKSummand(const mat& Pi, const int &i){
        mat PiChol, tmpSum, tmpB;
        vec ytmp;
        PiChol = chol(Pi, "lower");
        ytmp = solve(trimatl(PiChol), yVecList.at(i));
        tmpSum = ytmp * ytmp.t();
        tmpSum.diag() -= 1;
        tmpB = solve(trimatl(PiChol), BmatList.at(i));
        return tmpB.t() * tmpSum * tmpB;
    }
};



#endif