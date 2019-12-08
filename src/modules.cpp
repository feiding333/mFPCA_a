#include "mFPCA-EM.hpp"
#include "mfpcaMLE.hpp"
#include "mfpcaMLEWeighted.hpp"


RCPP_MODULE(MFPCA){
    class_<EMFPCA>("EMFPCA")
    .constructor()
    .method("emCompute", &EMFPCA::emCompute)
    .method("setData", &EMFPCA::setData)
    .method("setPenalty", &EMFPCA::setPenalty)
    .method("initFPC", &EMFPCA::initFPC)
    .method("initMeanFun", &EMFPCA::initMeanFun)
    .method("getThetaMu", &EMFPCA::getThetaMu)
    .method("getThetaFPC", &EMFPCA::getThetaFPC)
    ;
    
    // class_<MFPCALoss>("MFPCLoss")
    //     .constructor<vec,  mat, double, vec>()
    //     .method("objF", &MFPCALoss::objF)
    //     .method("gradF", &MFPCALoss::gradF)
    //     .method("objF_Euc", &MFPCALoss::objF_Euc)
    //     .method("objF_EucCV", &MFPCALoss::objF_EucCV)
    //     .method("gradF_Euc", &MFPCALoss::gradF_Euc)
    //     .method("set_penaltyMatrix", &MFPCALoss::set_penaltyMatrix)
    //     .method("set_tuningParameter", &MFPCALoss::set_tuningParameter)
    //     .method("get_totalDF", &MFPCALoss::get_totalDF)
    //     .method("updateSigmaSq", &MFPCALoss::updateSigmaSq)
    //     .method("get_sigmaSq", &MFPCALoss::get_sigmaSq)
    // ;

    class_<mfpcaMLE>("mfpcaMLE")
        .constructor<vec,  mat, double, vec>()
        .method("objF", &mfpcaMLE::objF)
        .method("gradF", &mfpcaMLE::gradF)
        .method("objF_Euc", &mfpcaMLE::objF_Euc)
        .method("gradF_Euc", &mfpcaMLE::gradF_Euc)
        .method("set_penaltyMatrix", &mfpcaMLE::set_penaltyMatrix)
        .method("set_tuningParameter", &mfpcaMLE::set_tuningParameter)
        .method("get_totalDF", &mfpcaMLE::get_totalDF)
        .method("updateSigmaSq", &mfpcaMLE::updateSigmaSq)
        .method("get_sigmaSq", &mfpcaMLE::get_sigmaSq)
    ;
    
    
    class_<mfpcaMLEWeighted>("mfpcaMLEWeighted")
        .constructor<vec,  mat, double, vec>()
        .method("objF", &mfpcaMLEWeighted::objF)
        .method("gradF", &mfpcaMLEWeighted::gradF)
        .method("objF_Euc", &mfpcaMLEWeighted::objF_Euc)
        .method("gradF_Euc", &mfpcaMLEWeighted::gradF_Euc)
        .method("set_penaltyMatrix", &mfpcaMLEWeighted::set_penaltyMatrix)
        .method("set_tuningParameter", &mfpcaMLEWeighted::set_tuningParameter)
        .method("get_totalDF", &mfpcaMLEWeighted::get_totalDF)
        .method("set_covariateZ", &mfpcaMLEWeighted::set_covariateZ)
        .method("setNewZ", &mfpcaMLEWeighted::setNewZ)
    ;
    
}
