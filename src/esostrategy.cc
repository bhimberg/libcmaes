/**
 * CMA-ES, Covariance Matrix Adaptation Evolution Strategy
 * Copyright (c) 2014 Inria
 * Author: Emmanuel Benazera <emmanuel.benazera@lri.fr>
 *
 * This file is part of libcmaes.
 *
 * libcmaes is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libcmaes is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libcmaes.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "libcmaes_config.h"
#include "esostrategy.h"
#include "cmaparameters.h" // in order to pre-instanciate template into library.
#include "cmasolutions.h"
#include "cmastopcriteria.h"
#include <iostream>
#include "llogging.h"

#ifdef HAVE_DEBUG
#include <chrono>
#endif

namespace libcmaes
{
  template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
  }

  template<class TParameters,class TSolutions,class TStopCriteria>
  ESOStrategy<TParameters,TSolutions,TStopCriteria>::ESOStrategy(FitFunc &func,
								 TParameters &parameters)
    :_func(func),_nevals(0),_niter(0),_parameters(parameters)
  {
    if (parameters._maximize)
      {
	_funcaux = _func;
	_func = [&](const double *x, const int N) { return -1.0*_funcaux(x,N); };
      }
    _pfunc = [](const TParameters&,const TSolutions&){return 0;}; // high level progress function does do anything.
    _solutions = TSolutions(_parameters);
    if (parameters._uh)
      {
	std::random_device rd;
	_uhgen = std::mt19937(rd());
	_uhgen.seed(static_cast<uint64_t>(time(nullptr)));
	_uhunif = std::uniform_real_distribution<>(0,1);
      }
  }

  template<class TParameters,class TSolutions,class TStopCriteria>
  ESOStrategy<TParameters,TSolutions,TStopCriteria>::ESOStrategy(FitFunc &func,
								 TParameters &parameters,
								 const TSolutions &solutions)
    :_func(func),_nevals(0),_niter(0),_parameters(parameters)
  {
    _pfunc = [](const TParameters&,const TSolutions&){return 0;}; // high level progress function does do anything.
    start_from_solution(solutions);
    if (parameters._uh)
      {
	std::random_device rd;
	_uhgen = std::mt19937(rd());
	_uhgen.seed(static_cast<uint64_t>(time(nullptr)));
	_uhunif = std::uniform_real_distribution<>(0,1);
      }
  }
  
  template<class TParameters,class TSolutions,class TStopCriteria>
  ESOStrategy<TParameters,TSolutions,TStopCriteria>::~ESOStrategy()
  {
  }
  
  template<class TParameters,class TSolutions,class TStopCriteria>
  void ESOStrategy<TParameters,TSolutions,TStopCriteria>::eval(const dMat &candidates,
							       const dMat &phenocandidates)
  {
#ifdef HAVE_DEBUG
    std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
#endif
    // one candidate per row, send the vectors to be evaluated
    int command = 0;
    for (int r=0;r<candidates.cols();r++)
      {
	_solutions._candidates.at(r).set_x(candidates.col(r));
	_solutions._candidates.at(r).set_id(r);
        // send the evaluation command
        //std::cout << "esostrategy.h 1st Send Rank " << 0 << std::endl;
        for (int g = 0; g < _parameters._gamma; ++g)
        {
            MPI_Send(&command, 1, MPI_INT, r*_parameters._gamma+g+1, 0, MPI_COMM_WORLD);

            // send the data to be evaluated
            if (phenocandidates.size())
            {
                std::cout << "esostrategy.h 2nd Send Rank " << 0 << " evalp" << std::endl;
                MPI_Send(phenocandidates.col(r).data(), candidates.rows(), MPI_DOUBLE, r*_parameters._gamma+g+1, 0, MPI_COMM_WORLD);
            }
            else
            {
                std::cout << "esostrategy.h 3rd Send Rank " << 0 << " eval" <<  std::endl;
                MPI_Send(candidates.col(r).data(), candidates.rows(), MPI_DOUBLE, r*_parameters._gamma+g+1, 0, MPI_COMM_WORLD);
            }
        }
      }
    // we will retrieve our fvals now
    for (int r=0;r<candidates.cols();r++)
      {
	_solutions._candidates.at(r).set_x(candidates.col(r));
	_solutions._candidates.at(r).set_id(r);

        MPI_Status fvalue_status;
        
        double fvalue = 0.0;
        double fvalue_count = 0;
        for (int g = 0; g < _parameters._gamma; ++g)
        {
            double fvalue_temp = std::numeric_limits<double>::infinity();
            //std::cout << "esostrategy.h 1st Recv Rank " << 0 << std::endl;
            MPI_Recv(&fvalue_temp, 1, MPI_DOUBLE, r*_parameters._gamma+g+1, 0, MPI_COMM_WORLD,&fvalue_status);

            if (std::isfinite(fvalue_temp))
            {
                fvalue += fvalue_temp;
                fvalue_count += 1;
            }
        }
        fvalue /= fvalue_count;
        _solutions._candidates.at(r).set_fvalue(fvalue);
      }
    int nfcalls = candidates.cols();
    
    // evaluation step of uncertainty handling scheme.
    if (_parameters._uh)
      {
				perform_uh(candidates,phenocandidates,nfcalls);
      }

    // if an elitist is active, reinject initial solution as needed.
    if (_niter > 0 && (_parameters._elitist || _parameters._initial_elitist || (_initial_elitist && _parameters._initial_elitist_on_restart)))
      {
	// get reference values.
	double ref_fvalue = std::numeric_limits<double>::max();
	Candidate ref_candidate;
	
	if (_parameters._initial_elitist_on_restart || _parameters._initial_elitist)
	  {
	    ref_fvalue = _solutions._initial_candidate.get_fvalue();
	    ref_candidate = _solutions._initial_candidate;
	  }
	else if (_parameters._elitist)
	  {
	    ref_fvalue = _solutions._best_seen_candidate.get_fvalue();
	    ref_candidate = _solutions._best_seen_candidate;
	  }

	// reinject intial solution if half or more points have value above that of the initial point candidate.
	int count = 0;
	for (int r=0;r<candidates.cols();r++)
	  if (_solutions._candidates.at(r).get_fvalue() < ref_fvalue)
	    ++count;
	if (count/2.0 < candidates.cols()/2)
	  {
#ifdef HAVE_DEBUG
	    //std::cout << "reinjecting solution=" << ref_fvalue << std::endl;
#endif
	    _solutions._candidates.at(1) = ref_candidate;
	  }
      }
    
    update_fevals(nfcalls);
    
#ifdef HAVE_DEBUG
    std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
    _solutions._elapsed_eval = std::chrono::duration_cast<std::chrono::milliseconds>(tstop-tstart).count();
#endif
  }

  template<class TParameters,class TSolutions,class TStopCriteria>
  void ESOStrategy<TParameters,TSolutions,TStopCriteria>::inc_iter()
  {
    _niter++;
    _solutions._niter++;
  }

  template<class TParameters,class TSolutions,class TStopCriteria>
  void ESOStrategy<TParameters,TSolutions,TStopCriteria>::update_fevals(const int &evals)
  {
    _nevals += evals;
    _solutions._nevals += evals;
  }

  template<class TParameters,class TSolutions,class TStopCriteria>
  dVec ESOStrategy<TParameters,TSolutions,TStopCriteria>::gradf(const dVec &x)
  {
    if (_gfunc != nullptr)
      return _gfunc(x.data(),_parameters._dim);
    dVec vgradf(_parameters._dim);
    dVec epsilon = 1e-8 * (dVec::Constant(_parameters._dim,1.0) + x.cwiseAbs());

    // send off fx to be evaluated by rank 1 (0 tag signals to use the FitFunc function)
    int command = 0;
    for (int g = 0; g < _parameters._gamma; ++g)
    {
        //std::cout << "esostrategy.h 4th Send Rank " << 0 << std::endl;
        MPI_Send(&command, 1, MPI_INT, 1+g, 0, MPI_COMM_WORLD);

        //std::cout << "esostrategy.h 5th Send Rank " << 0 << std::endl;
        MPI_Send(x.data(), _parameters._dim, MPI_DOUBLE, 1+g, 0, MPI_COMM_WORLD);
    }

    // send off the gradient evaluation portions to remaining nodes
    for (int i = 0; i<_parameters._dim;++i)
    {
	dVec ei1 = x;
	ei1(i,0) += epsilon(i);
	ei1(i,0) = std::min(ei1(i,0),_parameters.get_gp().get_boundstrategy_ref().getUBound(i));
        //std::cout << "esostrategy.h 6th Send Rank " << 0 << " grad" <<  std::endl;
        for (int g = 0; g < _parameters._gamma; ++g)
        {
            MPI_Send(&command, 1, MPI_INT, (i+1)*_parameters._gamma+g+1, 0, MPI_COMM_WORLD);
            std::cout << "esostrategy.h 7th Send Rank " << 0 << " grad " << std::endl;
            MPI_Send(ei1.data(), _parameters._dim, MPI_DOUBLE, (i+1)*_parameters._gamma+g+1, 0, MPI_COMM_WORLD);
        }
    }

    // recv the fitness evaluation from rank 1
    MPI_Status fx_status;
    double fx = 0.0;
    int fx_count = 0;
    for (int g = 0; g < _parameters._gamma; ++g)
    {
        double fx_temp = std::numeric_limits<double>::infinity();
        //std::cout << "esostrategy.h 2nd Recv Rank " << 0 << std::endl;
        MPI_Recv(&fx_temp, 1, MPI_DOUBLE, 1+g, 0, MPI_COMM_WORLD,&fx_status);
        if (std::isfinite(fx_temp) == true)
        {
            fx += fx_temp;
            fx_count += 1;
        }
    }
    fx /= fx_count;

    // recv remaining fitness evaluations
    for (int i=0;i<_parameters._dim;i++)
      {
        MPI_Status gradi_status;
        double gradi = 0.0;
        int gradi_count = 0;
        for (int g = 0; g < _parameters._gamma; ++g)
        {
            double gradi_temp = std::numeric_limits<double>::infinity();
            //std::cout << "esostrategy.h 3rd Recv Rank " << 0 << std::endl;
            MPI_Recv(&gradi, 1, MPI_DOUBLE, (i+1)*_parameters._gamma+g+1, 0, MPI_COMM_WORLD,&gradi_status);
            if (std::isfinite(gradi_temp) == true)
            {
                gradi += gradi_temp;
                gradi_count += 1;
            }
        }
        gradi /= gradi_count;

        // store it
	vgradf(i,0) = (gradi - fx)/epsilon(i);
      }
    update_fevals(_parameters._dim+1); // numerical gradient increases the budget.
    return vgradf;
  }

  template<class TParameters,class TSolutions,class TStopCriteria>
  dVec ESOStrategy<TParameters,TSolutions,TStopCriteria>::gradgp(const dVec &x) const
  {
    dVec epsilon = 1e-8 * (dVec::Constant(_parameters._dim,1.0) + x.cwiseAbs());
    return (_parameters._gp.pheno(dVec(x+epsilon))-_parameters._gp.pheno(dVec(x-epsilon))).cwiseQuotient(2.0*epsilon);
  }
  
  template<class TParameters,class TSolutions,class TStopCriteria>
  double ESOStrategy<TParameters,TSolutions,TStopCriteria>::edm()
  {
    int n = _parameters._dim;
    double edm = n / (10.0*(sqrt(_parameters._lambda / 4.0 + 0.5)-1));
    dVec gradff = gradf(_parameters._gp.pheno(_solutions._xmean));
    dVec gradgpf = gradgp(_solutions._xmean);
    gradff = gradff.cwiseProduct(gradgpf);
    dMat gradmn;
    if (!_parameters._sep)
      gradmn = _solutions._leigenvectors*_solutions._leigenvalues.cwiseSqrt().asDiagonal() * gradff;
    else gradmn = _solutions._sepcov.cwiseSqrt().cwiseProduct(gradff);
    double gradn = _solutions._sigma * gradmn.norm();
    edm *= gradn;
    _solutions._edm = edm;
    return edm;
  }

  template<class TParameters,class TSolutions,class TStopCriteria>
  void ESOStrategy<TParameters,TSolutions,TStopCriteria>::uncertainty_handling()
  {
    std::sort(_solutions._candidates_uh.begin(),
	      _solutions._candidates_uh.end(),
	      [](const RankedCandidate &c1, const RankedCandidate &c2)
	      { 
		bool lower = c1.get_fvalue() < c2.get_fvalue();
		return lower;
	      });
    int pos = 0;
    auto vit = _solutions._candidates_uh.begin();
    while(vit!=_solutions._candidates_uh.end())
      {
	(*vit)._r1 = pos;
	++vit;
	++pos;
      }
    
    // sort second uh set of candidates
    std::sort(_solutions._candidates_uh.begin(),
	      _solutions._candidates_uh.end(),
	      [](const RankedCandidate &c1, const RankedCandidate &c2)
	      { 
		bool lower = c1._fvalue_mut < c2._fvalue_mut;
		return lower;
	      });
    pos = 0;
    vit = _solutions._candidates_uh.begin();
    while(vit!=_solutions._candidates_uh.end())
      {
	(*vit)._r2 = pos;
	++vit;
	++pos;
      }
    
    // compute delta
    vit = _solutions._candidates_uh.begin();
    while(vit!=_solutions._candidates_uh.end())
      {
	if ((*vit)._idx >= _solutions._lambda_reev)
	  {
	    ++vit;
	    continue;
	  }
	int diffr = (*vit)._r2 - (*vit)._r1;
	(*vit)._delta = diffr - sgn(diffr);
	++vit;
      }
    double meandelta = std::accumulate(_solutions._candidates_uh.begin(),
				       _solutions._candidates_uh.end(),
				       0.0,
				       [](double sum, const RankedCandidate &c){ return sum + fabs(c._delta); });
    meandelta /= _solutions._lambda_reev;
    
    // compute uncertainty level
    double s = 0.0;
    for (size_t i=0;i<_solutions._candidates_uh.size();i++)
      {
	RankedCandidate rc = _solutions._candidates_uh.at(i);
	if (rc._idx >= _solutions._lambda_reev)
	  continue;
	s += 2*fabs(rc._delta);
	double d1 = rc._r2 - static_cast<int>(rc._r2 > rc._r1);
	std::vector<double> dv;
	double fact = _parameters._thetauh*0.5;
	for (int j=1;j<2*_parameters._lambda;j++)
	  dv.push_back(fabs(j-d1));
	std::nth_element(dv.begin(),dv.begin()+int(dv.size()*fact),dv.end());
	double comp1 = *(dv.begin()+int(dv.size()*fact));
	s -= comp1;
	
	double d2 = rc._r1 - static_cast<int>(rc._r1 > rc._r2);
	dv.clear();
	for (int j=1;j<2*_parameters._lambda;j++)
	  dv.push_back(fabs(j-d2));
	std::nth_element(dv.begin(),dv.begin()+int(dv.size()*fact),dv.end());
	double comp2 = *(dv.begin()+int(dv.size()*fact));
	s -= comp2;
      }
    s /= static_cast<double>(_solutions._lambda_reev);
    _solutions._suh = s;
    
    // rerank according to r1 + r2
    int lreev = _solutions._lambda_reev;
    std::sort(_solutions._candidates_uh.begin(),
	      _solutions._candidates_uh.end(),
	      [lreev,meandelta](RankedCandidate const &c1, RankedCandidate const &c2)
	      { 
		int s1 = c1._r1 + c1._r2;
		int s2 = c2._r2 + c2._r2;
		if (s1 == s2)
		  {
		    if (c1._delta == c2._delta)
		      return c1.get_fvalue() + c1._fvalue_mut < c2.get_fvalue() + c2._fvalue_mut;
		    else
		      {
			double c1d = c1._idx < lreev ? fabs(c1._delta) : meandelta;
			double c2d = c2._idx < lreev ? fabs(c2._delta) : meandelta;
			return c1d < c2d;
		      }
		  }
		else return c1._r1 + c1._r2 < c2._r1 + c2._r2;
	      });
    std::vector<Candidate> ncandidates;
    vit = _solutions._candidates_uh.begin();
    while(vit!=_solutions._candidates_uh.end())
      {
	ncandidates.push_back(_solutions._candidates.at((*vit)._idx));
	++vit;
      }
    _solutions._candidates = ncandidates;
  }

  template<class TParameters,class TSolutions,class TStopCriteria>
  void ESOStrategy<TParameters,TSolutions,TStopCriteria>::perform_uh(const dMat& candidates, const dMat& phenocandidates, int& nfcalls)
	{
		dMat candidates_uh;
		select_candidates_uh(candidates, phenocandidates, candidates_uh);
		std::vector<RankedCandidate> nvcandidates;
		eval_candidates_uh(candidates,candidates_uh,nvcandidates,nfcalls);
		set_candidates_uh(nvcandidates);
	}

  template<class TParameters,class TSolutions,class TStopCriteria>
  void ESOStrategy<TParameters,TSolutions,TStopCriteria>::select_candidates_uh(const dMat& candidates, const dMat& phenocandidates, dMat& candidates_uh)
	{
	// compute the number of solutions to re-evaluate
	_solutions._lambda_reev = 0.0;
	double r_l = _parameters._rlambda * _parameters._lambda;
	int lr_l = std::floor(r_l);
	double pr_l = r_l - lr_l;
	double p = _uhunif(_uhgen);
	if (p < pr_l)
	  _solutions._lambda_reev = lr_l + 1;
	else _solutions._lambda_reev = lr_l;
	if (_solutions._lambda_reev == 0)
	  _solutions._lambda_reev = 1;
	
	// mutate candidates.
	if (phenocandidates.size())
	  candidates_uh = phenocandidates.block(0,0,phenocandidates.rows(),_solutions._lambda_reev);
	else candidates_uh = candidates.block(0,0,candidates.rows(),_solutions._lambda_reev);
	if (_solutions._sepcov.size())
	  _uhesolver.set_covar(_solutions._sepcov);
	else _uhesolver.set_covar(_solutions._cov);
	candidates_uh += _parameters._epsuh * _solutions._sigma * _uhesolver.samples_ind(_solutions._lambda_reev);
	}

  template<class TParameters,class TSolutions,class TStopCriteria>
  void ESOStrategy<TParameters,TSolutions,TStopCriteria>::eval_candidates_uh(const dMat& candidates, const dMat& candidates_uh, std::vector<RankedCandidate>& nvcandidates, int& nfcalls)
	{
	// send the candidates for evaluation
        int command = 0;
	for (int r=0;r<candidates.cols();r++)
	  {
	    if (r < _solutions._lambda_reev)
	      {
                for (int g = 0; g < _parameters._gamma; ++g)
                {
                  //std::cout << "esostrategy.h 8th Send Rank " << 0 << std::endl;
                  MPI_Send(&command,1, MPI_INT, r*_parameters._gamma+g+1, 0, MPI_COMM_WORLD);
                  //std::cout << "esostrategy.h 9th Send Rank " << 0 << std::endl;
                  MPI_Send(candidates_uh.col(r).data(),candidates_uh.rows(), MPI_DOUBLE, r*_parameters._gamma+g+1, 0, MPI_COMM_WORLD);
                }
	      }
	  }
        // recv the results
	for (int r=0;r<candidates.cols();r++)
	  {
	    if (r < _solutions._lambda_reev)
	      {
                MPI_Status nfvalue_status;
		double nfvalue = 0.0;
                int nfvalue_count = 0;
                for (int g = 0; g < _parameters._gamma; ++g)
                {
                    double nfvalue_temp = std::numeric_limits<double>::infinity();
                    //std::cout << "esostrategy.h 4th Recv Rank " << 0 << std::endl;
                    MPI_Recv(&nfvalue_temp, 1, MPI_DOUBLE, r*_parameters._gamma+g+1, 0, MPI_COMM_WORLD,&nfvalue_status);

                    if (std::isfinite(nfvalue_temp))
                    {
                        nfvalue += nfvalue_temp;
                        nfvalue_count += 1;
                    }
                }
                nfvalue /= nfvalue_count;
		nvcandidates.emplace_back(nfvalue,_solutions._candidates.at(r),r);
		nfcalls++;
	      }
	    else
                nvcandidates.emplace_back(_solutions._candidates.at(r).get_fvalue(),_solutions._candidates.at(r),r);
	  }
	}

  template<class TParameters,class TSolutions,class TStopCriteria>
  void ESOStrategy<TParameters,TSolutions,TStopCriteria>::set_candidates_uh(const std::vector<RankedCandidate>& nvcandidates)
	{
	_solutions._candidates_uh = nvcandidates;
	}
  
  template<class TParameters,class TSolutions,class TStopCriteria>
  void ESOStrategy<TParameters,TSolutions,TStopCriteria>::tpa_update()
  {
    int r1 = -1;
    int r2 = -1;
    for (size_t i=0;i<_solutions._candidates.size();i++)
      {
	if (r1 == -1 && _solutions._candidates.at(i).get_id() == _solutions._tpa_p1)
	{
	    r1 = i;
	  }
	if (r2 == -1 && _solutions._candidates.at(i).get_id() == _solutions._tpa_p2)
	  {
	    r2 = i;
	  }
	if (r1 != -1 && r2 != -1)
	  {
	    break;
	  }
      }
    int rank_diff = r2-r1;
    _solutions._tpa_s = (1.0 - _parameters._tpa_csigma) * _solutions._tpa_s
      + _parameters._tpa_csigma * rank_diff / (_parameters._lambda - 1.0);
  }
  
  template<class TParameters,class TSolutions,class TStopCriteria>
  Candidate ESOStrategy<TParameters,TSolutions,TStopCriteria>::best_solution() const
  {
    return _solutions.best_candidate();
  }
  
  template class ESOStrategy<CMAParameters<GenoPheno<NoBoundStrategy>>,CMASolutions,CMAStopCriteria<GenoPheno<NoBoundStrategy>> >;
  template class ESOStrategy<CMAParameters<GenoPheno<pwqBoundStrategy>>,CMASolutions,CMAStopCriteria<GenoPheno<pwqBoundStrategy>> >;
  template class ESOStrategy<CMAParameters<GenoPheno<NoBoundStrategy,linScalingStrategy>>,CMASolutions,CMAStopCriteria<GenoPheno<NoBoundStrategy,linScalingStrategy>> >;
  template class ESOStrategy<CMAParameters<GenoPheno<pwqBoundStrategy,linScalingStrategy>>,CMASolutions,CMAStopCriteria<GenoPheno<pwqBoundStrategy,linScalingStrategy>> >;
}
