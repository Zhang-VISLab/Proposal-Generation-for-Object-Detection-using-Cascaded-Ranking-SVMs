#include "mex.h"
#include <math.h>
#include <omp.h>

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }

static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }


double compute_ov(double *bb, double *bbgt)
{	
	double bi[4];
	bi[0] = (double)(max(bb[0],bbgt[0]));
	bi[1] = (double)(max(bb[1],bbgt[1]));
	bi[2] = (double)(min(bb[2],bbgt[2]));
	bi[3] = (double)(min(bb[3],bbgt[3]));	
	
	double iw = bi[2] - bi[0] + 1;
	double ih = bi[3] - bi[1] + 1;
	double ov = 0;
	if (iw>0 && ih>0)
	{
		double ua = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)+(bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1)-iw*ih;
		ov = iw*ih/ua;
	}	
	
	return ov;
}


void overlap(double *bbPtr, double *bbgtPtr, const int *bbDims, const int *bbgtDims, double thd, double *lxPtr, double *ovPtr)
{
   #pragma omp parallel for
   for (int i=0; i<bbDims[1]; i++)
   {
	  double A[4];
	  for (int k=0; k<4; k++)
	  {
		A[k] = bbPtr[i*bbDims[0]+k];
	  }
	  double max_ov = 0;
	  for (int j=0; j<bbgtDims[1]; j++)
	  {		   
		  double B[4];
		  for (int k=0; k<4; k++)
		  {
			B[k] = bbgtPtr[j*bbgtDims[0]+k];
		  }
		  double ov = compute_ov(A,B);		  
		  if (ov >= max_ov)
		  {
			  max_ov = ov;				  
		  }		 
	  }
	  if (max_ov >= thd)
	  {
	    lxPtr[i] = 1;
	  }
	  else
	  {
	    lxPtr[i] = -1;
	  }
	  ovPtr[i] = max_ov;
  }  
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
	const mxArray *bb = prhs[0];
	const mxArray *bbgt = prhs[1];
	const mxArray *mxthd = prhs[2];
	double *bbPtr = (double *)mxGetPr(bb);
	double *bbgtPtr = (double *)mxGetPr(bbgt);
	const int *bbDims = mxGetDimensions(bb);
	const int *bbgtDims = mxGetDimensions(bbgt);
	double thd = (double)mxGetScalar(mxthd);
	
	mxArray *lx = mxCreateNumericMatrix((int)(bbDims[1]), 1, mxDOUBLE_CLASS, mxREAL);
	double *lxPtr = (double *)mxGetPr(lx);
	mxArray *ov = mxCreateNumericMatrix((int)(bbDims[1]), 1, mxDOUBLE_CLASS, mxREAL);
	double *ovPtr = (double *)mxGetPr(ov);

	overlap(bbPtr,bbgtPtr,bbDims,bbgtDims,thd,lxPtr,ovPtr);
	plhs[0] = lx;
	plhs[1] = ov;
}