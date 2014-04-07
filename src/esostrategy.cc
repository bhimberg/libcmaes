
#include "esostrategy.h"
#include "cmaparameters.h" // in order to pre-instanciated template into library.
#include "cmasolutions.h"
#include "cmastopcriteria.h"
#include <iostream>
#include <glog/logging.h>

namespace libcmaes
{
  template<class TParameters,class TSolutions,class TStopCriteria>
  ESOStrategy<TParameters,TSolutions,TStopCriteria>::ESOStrategy(FitFunc &func,
						   TParameters &parameters)
    :_func(func),_nevals(0),_niter(0),_parameters(parameters)
  {
    _solutions = TSolutions(_parameters._dim,_parameters._lambda);
  }
  
  template<class TParameters,class TSolutions,class TStopCriteria>
  ESOStrategy<TParameters,TSolutions,TStopCriteria>::~ESOStrategy()
  {
  }
  
  template<class TParameters,class TSolutions,class TStopCriteria>
  void ESOStrategy<TParameters,TSolutions,TStopCriteria>::eval(const dMat &candidates)
  {
    // one candidate per row.
    for (int r=0;r<candidates.cols();r++)
      {
	_solutions._candidates.at(r)._x = candidates.col(r);
	_solutions._candidates.at(r)._fvalue = _func(_solutions._candidates.at(r)._x.data(),candidates.rows());
	++_nevals;
	
	//std::cerr << "candidate x: " << _solutions._candidates.at(r)._x.transpose() << std::endl;
      }
  }

  template<class TParameters,class TSolutions,class TStopCriteria>
  Candidate ESOStrategy<TParameters,TSolutions,TStopCriteria>::best_solution() const
  {
    return _solutions.best_candidate();
  }
  
  template class ESOStrategy<CMAParameters,CMASolutions,CMAStopCriteria>;
  
}