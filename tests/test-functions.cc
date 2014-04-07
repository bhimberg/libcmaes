#include "esoptimizer.h"
#include "cmastrategy.h"
#include <map>
#include <iostream>
#include <math.h>
#include <glog/logging.h>

//#define STRIP_FLAG_HELP 1
#include <gflags/gflags.h>

#include <assert.h>

using namespace libcmaes;

bool compEp(const double &a, const double &b, const double &epsilon)
{
  return fabs(a-b) <= epsilon;
}

// classical test functions for single-objective optimization problems.
FitFunc ackleys = [](const double *x, const int N)
{
  return -20.0*exp(-0.2*sqrt(0.5*(x[0]*x[0]+x[1]*x[1]))) - exp(0.5*(cos(2.0*M_PI*x[0]) + cos(2.0*M_PI*x[1]))) + 20.0 + exp(1.0);
};

FitFunc fsphere = [](const double *x, const int N)
{
  double val = 0.0;
  for (int i=0;i<N;i++)
    val += x[i]*x[i];
  return val;
};

FitFunc cigtab = [](const double *x, const int N)
{
  int i;
  double sum = 1e4*x[0]*x[0] + 1e-4*x[1]*x[1];
  for(i = 2; i < N; ++i)
    sum += x[i]*x[i];
  return sum;
};

FitFunc rosenbrock = [](const double *x, const int N)
{
  double val = 0.0;
  for (int i=0;i<N-1;i++)
    {
      val += 100.0*pow((x[i+1]-x[i]*x[i]),2) + pow((x[i]-1.0),2);
    }
  return val;
};

FitFunc beale = [](const double *x, const int N)
{
  return pow(1.5-x[0]+x[0]*x[1],2) + pow(2.25 - x[0] + x[0]*x[1]*x[1],2) + pow(2.625-x[0]+x[0]*pow(x[1],3),2);
};

FitFunc goldstein_price = [](const double *x, const int N)
{
  return (1.0 + pow(x[0] + x[1] + 1.0,2)*(19.0-14.0*x[0]+3*x[0]*x[0]-14.0*x[1]+6*x[0]*x[1]+3*x[1]*x[1]))*(30.0+pow(2.0*x[0]-3.0*x[1],2)*(18.0-32.0*x[0]+12.0*x[0]*x[0]+48.0*x[1]-36.0*x[0]*x[1]+27.0*x[1]*x[1]));
};

FitFunc booth = [](const double *x, const int N)
{
  return pow(x[0]+2.0*x[1]-7,2) + pow(2*x[0]+x[1]-5,2);
};

FitFunc bukin = [](const double *x, const int N)
{
  return 100.0 * sqrt(fabs(x[1]-0.01*x[0]*x[0])) + 0.01*fabs(x[0]+10.0);
};

FitFunc matyas = [](const double *x, const int N)
{
  return 0.26*(x[0]*x[0]+x[1]*x[1])-0.48*x[0]*x[1];
};

FitFunc levi = [](const double *x, const int N)
{
  return pow(sin(3*M_PI*x[0]),2) + pow(x[0]-1,2) * (1.0+pow(sin(3*M_PI*x[1]),2)) + pow(x[1]-1,2)*(1.0+pow(sin(2.0*M_PI*x[1]),2));
};

FitFunc camel = [](const double *x, const int N)
{
  return 2.0*x[0]*x[0] - 1.05*pow(x[0],4) + pow(x[0],6)/6.0 + x[0]*x[1] + x[1]*x[1];
};

FitFunc easom = [](const double *x, const int N)
{
  return -cos(x[0])*cos(x[1])*exp(-(pow((x[0]-M_PI),2)+pow((x[1]-M_PI),2)));
};

FitFunc crossintray = [](const double *x, const int N)
{
  return -0.0001*pow(fabs(sin(x[0])*sin(x[1])*exp(fabs(100.0-sqrt(x[0]*x[0]+x[1]*x[1])/M_PI)))+1.0,0.1);
};

FitFunc eggholder = [](const double *x, const int N)
{
  return -(x[1]+47)*sin(sqrt(fabs(x[1]+0.5*x[0]+47.0))) - x[0]*sin(sqrt(fabs(x[0]-(x[1] + 47.0))));
};

FitFunc holdertable = [](const double *x, const int N)
{
  return -fabs(sin(x[0])*cos(x[1])*exp(fabs(1.0-sqrt(x[0]*x[0]+x[1]*x[1])/M_PI)));
};

FitFunc mccormick = [](const double *x, const int N)
{
  return sin(x[0]+x[1])+pow(x[0]-x[1],2) - 1.5*x[0] + 2.5*x[1] + 1.0;
};

FitFunc schaffer1 = [](const double *x, const int N)
{
  return 0.5 + (pow(sin(x[0]*x[0]-x[1]*x[1]),2)-0.5) / pow(1.0+0.001*(x[0]*x[0]+x[1]*x[1]),2);
};

FitFunc schaffer2 = [](const double *x, const int N)
{
  return 0.5 + (cos(sin(fabs(x[0]*x[0]-x[1]*x[1])))-0.5) / pow(1.0+0.001*(x[0]*x[0]+x[1]*x[1]),2);
};

FitFunc styblinski_tang = [](const double *x, const int N)
{
  double val = 0.0;
  for (int i=0;i<N;i++)
    val += pow(x[i],4) - 16.0*x[i]*x[i] + 5.0*x[i];
  return 0.5*val;
};

std::map<std::string,FitFunc> mfuncs;
std::map<std::string,Candidate> msols;
std::map<std::string,FitFunc>::const_iterator mit;

void fillupfuncs()
{
  mfuncs["ackleys"]=ackleys;
  msols["ackleys"]=Candidate(0.0,dVec::Constant(2,0));
  mfuncs["fsphere"]=fsphere;
  msols["fsphere"]=Candidate(0.0,dVec::Constant(20,0));
  mfuncs["cigtab"]=cigtab;
  mfuncs["rosenbrock"]=rosenbrock;
  msols["rosenbrock"]=Candidate(0.0,dVec::Constant(20,1));
  mfuncs["beale"]=beale;
  mfuncs["goldstein_price"]=goldstein_price;
  mfuncs["booth"]=booth;
  mfuncs["bukin"]=bukin;
  mfuncs["matyas"]=matyas;
  mfuncs["levi"]=levi;
  mfuncs["camel"]=camel;
  mfuncs["easom"]=easom;
  mfuncs["crossintray"]=crossintray;
  mfuncs["eggholder"]=eggholder;
  mfuncs["holdertable"]=holdertable;
  mfuncs["mccormick"]=mccormick;
  mfuncs["schaffer1"]=schaffer1;
  mfuncs["schaffer2"]=schaffer2;
  mfuncs["styblinski_tang"]=styblinski_tang;
}

// command line options.
DEFINE_string(fname,"fsphere","name of the function to optimize");
DEFINE_int32(dim,2,"problem dimension");
DEFINE_int32(lambda,10,"number of offsprings");
DEFINE_int32(max_iter,1000,"maximum number of iteration (-1 for unlimited)");
DEFINE_bool(list,false,"returns a list of available functions");
DEFINE_bool(all,false,"test on all functions");
DEFINE_double(epsilon,1e-10,"epsilon on function result testing, with --all");

int main(int argc, char *argv[])
{
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr=1;
  google::SetLogDestination(google::INFO,"");
  //FLAGS_log_prefix=false;

  fillupfuncs();

  if (FLAGS_list)
    {
      std::cout << "available functions: ";
      for (auto imap: mfuncs)
	std::cout << imap.first << " ";
      std::cout << std::endl;
      exit(1);
    }
  else if (FLAGS_all)
    {
      mit = mfuncs.begin();
      while(mit!=mfuncs.end())
	{
	  int dim = msols[(*mit).first]._x.rows();
	  CMAParameters cmaparams(dim,FLAGS_lambda,FLAGS_max_iter);
	  cmaparams._quiet = true;
	  ESOptimizer<CMAStrategy<CovarianceUpdate>,CMAParameters> cmaes(mfuncs[(*mit).first],cmaparams);
	  cmaes.optimize();
	  Candidate c = cmaes.best_solution();
	  //TODO: check on solution in x space.
	  if (compEp(c._fvalue,msols[(*mit).first]._fvalue,FLAGS_epsilon))
	    LOG(INFO) << (*mit).first << " -- OK\n";
	  else LOG(INFO) << (*mit).first << " -- FAILED\n";
	  ++mit;
	}
      exit(1);
    }
  
  if ((mit=mfuncs.find(FLAGS_fname))==mfuncs.end())
    {
      LOG(ERROR) << FLAGS_fname << " function does not exist, run with --list to get the list of all functions. Exiting.\n";
      exit(1);
    }
  CMAParameters cmaparams(FLAGS_dim,FLAGS_lambda,FLAGS_max_iter);
  ESOptimizer<CMAStrategy<CovarianceUpdate>,CMAParameters> cmaes(mfuncs[FLAGS_fname],cmaparams);
  cmaes.optimize();
}