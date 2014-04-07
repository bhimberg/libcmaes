
#ifndef CMAPARAMETERS_H
#define CMAPARAMETERS_H

#include "parameters.h"
#include "eo_matrix.h"

namespace libcmaes
{
  class CMAParameters : public Parameters
  {
  public:
    CMAParameters() {};
    CMAParameters(const int &dim, const int &lambda,
		  const int &max_iter=-1);
    ~CMAParameters();
    
    int _mu;
    dVec _weights;
    double _csigma;
    double _c1;
    double _cmu;
    double _cc;
    double _muw;
    double _dsigma;
    
    // computed once at init for speeding up operations.
    double _fact_ps; //TODO.
    double _fact_pc; //TODO.
    double _chi; // norm of N(0,I).
  };
  
}

#endif