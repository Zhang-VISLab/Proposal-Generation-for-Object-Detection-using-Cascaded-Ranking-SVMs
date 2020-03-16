#include "mex.h"
#include <math.h>

static inline double min(double x, double y) { return (x <= y ? x : y); }

static inline int min(int x, int y) { return (x <= y ? x : y); }

mxArray *true_location(const mxArray *mxpos, const mxArray *mxratio, const mxArray *mxsize)
{
	double *pos = (double *)(mxGetPr(mxpos));
	double *ratio = (double *)(mxGetPr(mxratio));
	double *size = (double *)(mxGetPr(mxsize));
	const int *dims = mxGetDimensions(mxpos);

	int out[2];
	out[0] = 4;
    out[1] = dims[1];  
	mxArray *TL = mxCreateNumericArray(2, out, mxDOUBLE_CLASS, mxREAL);
	double *tlPtr = (double *)mxGetPr(TL);

	for (int i=0; i<dims[1]; i++)
	{
		tlPtr[i*4] = (int)((pos[i*4]-1)*ratio[0]+1);
		tlPtr[i*4+1] = (int)((pos[i*4+1]-1)*ratio[1]+1);
		tlPtr[i*4+2] = (int)(min(size[1],(pos[i*4+2]-pos[i*4]+1)*ratio[0]+tlPtr[i*4]-1));
		tlPtr[i*4+3] = (int)(min(size[0],(pos[i*4+3]-pos[i*4+1]+1)*ratio[1]+tlPtr[i*4+1]-1));
	}

	return TL;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	plhs[0] = true_location(prhs[0], prhs[1], prhs[2]);
}