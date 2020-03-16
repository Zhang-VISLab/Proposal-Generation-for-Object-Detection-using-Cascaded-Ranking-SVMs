#include <math.h>
#include "mex.h"

void sparse_matrices(double *x, int *pair, int *ratio, int dim, int pair_len, double *S, int *I, int *J)
{	
	//	sparse matrices
	for (int i=0; i<pair_len; i++)
	{
		if (ratio[2*i] == ratio[2*i+1])
		{
			for (int j=0; j<dim; j++)
			{
				S[i*2*dim+j] = x[dim*(pair[2*i]-1)+j] - x[dim*(pair[2*i+1]-1)+j];
				I[i*2*dim+j] = (ratio[2*i]-1)*dim+j+1;
				J[i*2*dim+j] = i+1;
			}
			for (int j=0; j<dim; j++)
			{
				I[i*2*dim+j+dim] = 1;
				J[i*2*dim+j+dim] = i+1;
			}
		}
		else
		{
			for (int j=0; j<dim; j++)
			{
				S[i*2*dim+j] = x[dim*(pair[2*i]-1)+j];
				I[i*2*dim+j] = (ratio[2*i]-1)*dim+j+1;
				J[i*2*dim+j] = i+1;
			}
			for (int j=0; j<dim; j++)
			{
				S[i*2*dim+j+dim] = -x[dim*(pair[2*i+1]-1)+j];
				I[i*2*dim+j+dim] = (ratio[2*i+1]-1)*dim+j+1;
				J[i*2*dim+j+dim] = i+1;
			}
		}	
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	const mxArray *mxx = prhs[0];
	const mxArray *mxpair = prhs[1];
	const mxArray *mxratio = prhs[2];
	double *x = (double *)mxGetPr(mxx);
	int *pair = (int *)mxGetPr(mxpair);
	int *ratio = (int *)mxGetPr(mxratio);
	int dim = (int)mxGetM(mxx);
	int xlen = (int)mxGetN(mxx);
	int pair_len = (int)mxGetN(mxpair);

	mxArray *S = mxCreateNumericMatrix(2*dim*pair_len,1,mxDOUBLE_CLASS, mxREAL);
	double *SPtr = (double *)mxGetPr(S);
	mxArray *I = mxCreateNumericMatrix(2*dim*pair_len,1,mxINT32_CLASS, mxREAL);
	int *IPtr = (int *)mxGetPr(I);
	mxArray *J = mxCreateNumericMatrix(2*dim*pair_len,1,mxINT32_CLASS, mxREAL);
	int *JPtr = (int *)mxGetPr(J);
	
	sparse_matrices(x, pair, ratio, dim, pair_len, SPtr, IPtr, JPtr);
	
	plhs[0] = S;
	plhs[1] = I;
	plhs[2] = J;
}