#include <math.h>
#include "mex.h"
#include <string.h>

void binary_pegasos(float *x, int *y, float *margin, int *order, float *w, float lambda, int iter_num, int start_point, int dim, int order_len)
{
	float A, rate;
	int oi;

	for (int iter=0; iter<iter_num; iter++)
	{
		for (int i=0; i<order_len; i++)
		{
			/* pick a data  */
			oi = order[i];			

			// calculate margins
			A = 0;
			for (int k=0; k<dim; k++)
				A += x[oi*dim+k]*w[k];
			
			// update
			rate = 1/(float)(iter*order_len+i+start_point+1);
			if (y[oi]*A<margin[oi])
			{				
				for (int j=0; j<dim; j++)				
					w[j] = (1-rate)*w[j] + rate/lambda*y[oi]*x[oi*dim+j];
			}
			else
			{
				for (int j=0; j<dim; j++)
					w[j] = (1-rate)*w[j];
			}
		}
	}	
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	const mxArray *mxX = prhs[0];
	const mxArray *mxY = prhs[1];
	const mxArray *mxMargin = prhs[2];
	const mxArray *mxOrder = prhs[3];
	const mxArray *mxW0 = prhs[4];
	const mxArray *mxLambda = prhs[5];
	const mxArray *mxIter_num = prhs[6];
	const mxArray *mxStart_point0 = prhs[7];

	float *x = (float *)mxGetPr(mxX);
	int *y = (int *)mxGetPr(mxY);
	float *margin = (float *)mxGetPr(mxMargin);
	int *order = (int *)mxGetPr(mxOrder);
	float lambda = (float)mxGetScalar(mxLambda);	
	int iter_num = (int)mxGetScalar(mxIter_num);
	int start_point0 = (int)mxGetScalar(mxStart_point0);

	int dim = (int)mxGetM(mxX);
	int order_len = (int)mxGetM(mxOrder);
	
	mxArray *w = mxCreateNumericMatrix(dim, 1, mxSINGLE_CLASS, mxREAL);
	float *wPtr = (float *)mxGetPr(w);
	float *wPtr0 = (float *)mxGetPr(mxW0);	
	memcpy(wPtr, wPtr0, sizeof(float)*dim);
	mxArray *mxStart_point = mxCreateDoubleScalar(start_point0+iter_num*order_len);
	
	binary_pegasos(x, y, margin, order, wPtr, lambda, iter_num, start_point0, dim, order_len);
	
	plhs[0] = w;	
	plhs[1] = mxStart_point;	
}