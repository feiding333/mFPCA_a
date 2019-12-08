#ifndef MULTIVARIATEFPCA_CLASS
#define MULTIVARIATEFPCA_CLASS

#include <RcppArmadillo.h>
#include <vector>
using namespace Rcpp;
using namespace arma;


class MFPCALoss{
public:
    
    // Ordered by sample n = 1, ..., N
    // bMatLarge, sparse b-spline matrix, band * df_spline columns
    // sCount_, number of obs points in each sample
    // length(y_): total number of obs points
    // length(sCount_): total number of obs
    MFPCALoss(vec y_, mat bMatLarge, 
              double sigma_, vec sCount_);
    // we might have further speed improvment with refinement of bMatLarge?
    
    
    void set_penaltyMatrix(mat Gamma_){
        Gamma = Gamma_;
    }

    void set_tuningParameter(double mu1_, double mu2_){
        mu1 = mu1_;
        mu2 = mu2_;
    }
    
    int get_totalDF(){ return totalDF;}
    double get_sigmaSq(){return sigmaSq;}

    // the objective and the first order gradient 
    // in the Euclidean space without penalty
    // These two functions are used for initialization
    double objF_Euc(mat S);
    mat gradF_Euc(mat S);
    double objF_EucCV(mat S);
    
    // on the manifold
    double objF(List UWUt);
    List gradF(List UWUt);
    
    
    
    void updateSigmaSq(mat S);

private:
    //  total sample size;
    // total degree of freedom. S and UWUt are both DF-by-DF matrices.
    size_t totalS, totalDF;
    size_t nTotal; // total obs points
    vec sCount;
    //tuning parameters on rank and smoothness respectively
    double mu1, mu2;
    double sigmaSq; // measurement error, sigma squared
    double cv1, cv2; // quantities used for cross-validation error
    // vec sCount;
    std::vector<mat> BtB;
    std::vector<double> yntyn;
    mat BtB_mn2_Sum, BtZBSum, Gamma;
};



#endif