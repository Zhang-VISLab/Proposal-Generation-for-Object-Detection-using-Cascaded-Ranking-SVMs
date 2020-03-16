#include <math.h>
#include "mex.h"

#define eps (float)0.0001

static inline float min(float x, float y) { return (x <= y ? x : y); }
static inline float max(float x, float y) { return (x <= y ? y : x); }

void rank_pegasos_L2(float *x, int *pair, float *w, int *cls, int max_cls, float lambda, int dim, int pair_len, int iteration)
{
	float acc, scale = 1.0, eta, a, sqrtLambda=sqrt(lambda);
	int pi, ni, pc, nc;

	for (int i=0; i<pair_len; i++)
	{
		/* pick a pair  */
		pi = pair[2*i];
		ni = pair[2*i+1];
		pc = cls[pi];
		nc = cls[ni];

		/* project on the weight vector */
		acc = 0.0;
		for (int j=0; j<dim; j++)
		{
			acc += x[pi*dim+j]*w[pc*dim+j] - x[ni*dim+j]*w[nc*dim+j];
		}
		acc *= scale ;

		/* learning rate */
		eta = 1.0 / (((float)(i+iteration)+1.0) * lambda) ;

		if (acc < 1.0)
		{
			/* margin violated */
			a = scale * (1.0 - eta * lambda) ;
			acc = 0.0;
			for (int j=0; j<max_cls; j++)
			{
				if (j==pc)
				{
					for (int k=0; k<dim; k++)
					{
						w[j*dim+k] = a*w[j*dim+k] + eta*x[pi*dim+k];
						acc += w[j*dim+k]*w[j*dim+k];
					}
				}
				if (j==nc)
				{
					for (int k=0; k<dim; k++)
					{
						w[j*dim+k] = a*w[j*dim+k] - eta*x[ni*dim+k];
						acc += w[j*dim+k]*w[j*dim+k];
					}
				}
				if (j!=pc && j!=nc)
				{
					for (int k=0; k<dim; k++)
					{
						w[j*dim+k] = a*w[j*dim+k];
						acc += w[j*dim+k]*w[j*dim+k];
					}
				}
			}
			scale = min(1.0 / (sqrtLambda * sqrt(acc + eps)), 1.0) ;
		}
		else
		{
			 /* margin not violated */
			scale *= 1.0 - eta * lambda ;
		}
	}

	/* denormalize representation */
	for (int i=0 ; i<dim*max_cls; i++)
	{
		w[i] *= scale;
	}
}

void rank_pegasos_L1(float *x, int *pair, float *pw, int *cls, int max_cls, float lambda_L2, float lambda_L1, int dim, int pair_len, int iteration)
{
	float acc, scale = 1.0, eta, a, b, sqrtLambda=(sqrt(4.0*lambda_L2+lambda_L1*lambda_L1)-lambda_L1)/(2.0*lambda_L2);
	int pi, ni, pc, nc;
	
	// duplicate w
	float *nw = (float *)mxCalloc(dim*max_cls, sizeof(float));

	for (int i=0; i<pair_len; i++)
	{
		/* pick a pair  */
		pi = pair[2*i];
		ni = pair[2*i+1];
		pc = cls[pi];
		nc = cls[ni];

		/* project on the weight vector */
		acc = 0.0;
		for (int j=0; j<dim; j++)
		{
			acc += x[pi*dim+j]*(pw[pc*dim+j]-nw[pc*dim+j]) - x[ni*dim+j]*(pw[nc*dim+j]-nw[nc*dim+j]);
		}
		acc *= scale ;

		/* learning rate */
		eta = 1.0 / (((float)(i+iteration)+1.0) * lambda_L2) ;
		a = 1.0 / (((float)(i+iteration)+1.0) * lambda_L1);
		b = scale * (1.0 - eta * lambda_L2);		
			
		if (acc < 1.0)
		{
			/* margin violated */		
			acc = 0.0;			
			for (int j=0; j<max_cls; j++)
			{
				if (j==pc)
				{
					for (int k=0; k<dim; k++)
					{
						pw[j*dim+k] = max(0.0, b*pw[j*dim+k] + eta*x[pi*dim+k] - a);
						nw[j*dim+k] = max(0.0, b*nw[j*dim+k] - eta*x[pi*dim+k] - a);
						acc += pw[j*dim+k] + nw[j*dim+k];
					}
				}
				if (j==nc)
				{
					for (int k=0; k<dim; k++)
					{
						pw[j*dim+k] = max(0.0, b*pw[j*dim+k] - eta*x[ni*dim+k] - a);
						nw[j*dim+k] = max(0.0, b*nw[j*dim+k] + eta*x[ni*dim+k] - a);
						acc += pw[j*dim+k] + nw[j*dim+k];
					}
				}
				if (j!=pc && j!=nc)
				{
					for (int k=0; k<dim; k++)
					{
						pw[j*dim+k] = max(0.0, b*pw[j*dim+k] - a);
						nw[j*dim+k] = max(0.0, b*nw[j*dim+k] - a);
						acc += pw[j*dim+k] + nw[j*dim+k];
					}
				}
			}			
		}
		else
		{
			 /* margin not violated */
			acc = 0.0;
			for (int j=0; j<max_cls; j++)
			{
				for (int k=0; k<dim; k++)
				{
					pw[j*dim+k] = max(0.0, b*pw[j*dim+k] - a);
					nw[j*dim+k] = max(0.0, b*nw[j*dim+k] - a);
					acc += pw[j*dim+k] + nw[j*dim+k];
				}
			}
		}
		scale = min(1.0 / (sqrtLambda * sqrt(acc + eps)), 1.0) ;
	}

	/* denormalize representation */
	for (int i=0 ; i<dim*max_cls; i++)
	{
		pw[i] = scale*(pw[i]-nw[i]);
	}

	mxFree(nw);
}

void rank_pegasos_L0(float *x, int *pair, float *pw, int *cls, int max_cls, float lambda_L2, float lambda_L1, int dim, int pair_len, int iteration)
{
	float margin, acc, w_scale = 1.0, z_scale, eta, a, b, sqrtLambda=sqrt(lambda_L2);
	int pi, ni, pc, nc;
	
	// create z
	float *pz = (float *)mxCalloc(dim*max_cls, sizeof(float));
	float *nz = (float *)mxCalloc(dim*max_cls, sizeof(float));
	for (int i=0; i<dim*max_cls; i++)
	{
		pz[i] = 1.0;
		nz[i] = 1.0;
	}
	z_scale = min(1.0 / (sqrtLambda * sqrt(2.0*dim*max_cls + eps)), 1.0) ;
	
	// duplicate w
	float *nw = (float *)mxCalloc(dim*max_cls, sizeof(float));

	for (int i=0; i<pair_len; i++)
	{
		/* pick a pair  */
		pi = pair[2*i];
		ni = pair[2*i+1];
		pc = cls[pi];
		nc = cls[ni];

		/* project on the weight vector */
		margin = 0.0;
		for (int j=0; j<dim; j++)
		{
			margin += x[pi*dim+j]*(pw[pc*dim+j]-nw[pc*dim+j])*(pz[pc*dim+j]-nz[pc*dim+j]) - x[ni*dim+j]*(pw[nc*dim+j]-nw[nc*dim+j])*(pz[nc*dim+j]-nz[nc*dim+j]);
		}
		margin *= w_scale*z_scale ;

		/* learning rate */
		eta = 1.0 / (((float)(i+iteration)+1) * lambda_L2) ;
		a = 1.0 / (((float)(i+iteration)+1) * lambda_L1);
		
		//	update w
		if (margin < 1.0)
		{
			/* margin violated */		
			b = w_scale * (1.0 - eta * lambda_L2);	
			acc = 0.0;			
			for (int j=0; j<max_cls; j++)
			{
				if (j==pc)
				{
					for (int k=0; k<dim; k++)
					{
						pw[j*dim+k] = max(0.0, b*pw[j*dim+k] + (eta*x[pi*dim+k] - a)*pz[j*dim+k]*z_scale);
						nw[j*dim+k] = max(0.0, b*nw[j*dim+k] - (eta*x[pi*dim+k] + a)*nz[j*dim+k]*z_scale);
						acc += pw[j*dim+k] + nw[j*dim+k];
					}
				}
				if (j==nc)
				{
					for (int k=0; k<dim; k++)
					{
						pw[j*dim+k] = max(0.0, b*pw[j*dim+k] - (eta*x[ni*dim+k] + a)*pz[j*dim+k]*z_scale);
						nw[j*dim+k] = max(0.0, b*nw[j*dim+k] + (eta*x[ni*dim+k] - a)*nz[j*dim+k]*z_scale);
						acc += pw[j*dim+k] + nw[j*dim+k];
					}
				}
				if (j!=pc && j!=nc)
				{
					for (int k=0; k<dim; k++)
					{
						pw[j*dim+k] = max(0.0, b*pw[j*dim+k] - a*pz[j*dim+k]*z_scale);
						nw[j*dim+k] = max(0.0, b*nw[j*dim+k] - a*nz[j*dim+k]*z_scale);
						acc += pw[j*dim+k] + nw[j*dim+k];
					}
				}
			}			
		}
		else
		{
			 /* margin not violated */
			for (int j=0; j<max_cls; j++)
			{
				for (int k=0; k<dim; k++)
				{
					pw[j*dim+k] = max(0.0, b*pw[j*dim+k] - a*pz[j*dim+k]*z_scale);
					nw[j*dim+k] = max(0.0, b*nw[j*dim+k] - a*nz[j*dim+k]*z_scale);
					acc += pw[j*dim+k] + nw[j*dim+k];
				}
			}
		}
		w_scale = min(1.0 / (sqrtLambda * sqrt(acc + eps)), 1.0) ;
		
		//	update z
		if (margin < 1.0)
		{
			/* margin violated */		
			b = z_scale * (1.0 - eta * lambda_L2);	
			acc = 0.0;			
			for (int j=0; j<max_cls; j++)
			{
				if (j==pc)
				{
					for (int k=0; k<dim; k++)
					{
						pz[j*dim+k] = max(0.0, b*pz[j*dim+k] + (eta*x[pi*dim+k] - a)*pw[j*dim+k]*w_scale);
						nz[j*dim+k] = max(0.0, b*nz[j*dim+k] - (eta*x[pi*dim+k] + a)*nw[j*dim+k]*w_scale);
						acc += pz[j*dim+k] + nz[j*dim+k];
					}
				}
				if (j==nc)
				{
					for (int k=0; k<dim; k++)
					{
						pz[j*dim+k] = max(0.0, b*pz[j*dim+k] - (eta*x[ni*dim+k] + a)*pw[j*dim+k]*w_scale);
						nz[j*dim+k] = max(0.0, b*nz[j*dim+k] + (eta*x[ni*dim+k] - a)*nw[j*dim+k]*w_scale);
						acc += pz[j*dim+k] + nz[j*dim+k];
					}
				}
				if (j!=pc && j!=nc)
				{
					for (int k=0; k<dim; k++)
					{
						pz[j*dim+k] = max(0.0, b*pz[j*dim+k] - a*pw[j*dim+k]*w_scale);
						nz[j*dim+k] = max(0.0, b*nz[j*dim+k] - a*nw[j*dim+k]*w_scale);
						acc += pz[j*dim+k] + nz[j*dim+k];
					}
				}
			}			
		}
		else
		{
			 /* margin not violated */
			for (int j=0; j<max_cls; j++)
			{
				for (int k=0; k<dim; k++)
				{
					pz[j*dim+k] = max(0.0, b*pz[j*dim+k] - a*pw[j*dim+k]*w_scale);
					nz[j*dim+k] = max(0.0, b*nz[j*dim+k] - a*nw[j*dim+k]*w_scale);
					acc += pz[j*dim+k] + nz[j*dim+k];
				}
			}
		}
		z_scale = min(1.0 / (sqrtLambda * sqrt(acc + eps)), 1.0) ;
	}

	/* denormalize representation */
	for (int i=0 ; i<dim*max_cls; i++)
	{
		pw[i] = w_scale*z_scale*(pw[i]*pz[i]-nw[i]*nz[i]);
	}

	mxFree(nw);
	mxFree(pz);
	mxFree(nz);
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	const mxArray *mxx = prhs[0];
	const mxArray *mxpair = prhs[1];
	const mxArray *mxcls = prhs[2];
	const mxArray *mxmax_cls = prhs[3];
	const mxArray *mxlambda_L2 = prhs[4];
	const mxArray *mxlambda_L1 = prhs[5];
	const mxArray *mxnorm = prhs[6];
	const mxArray *mxw0 = prhs[7];
	const mxArray *mxiter = prhs[8];
	float *x = (float *)mxGetPr(mxx);
	int dim = (int)mxGetM(mxx);
	int *pair = (int *)mxGetPr(mxpair);
	int pair_len = (int)mxGetN(mxpair);
	int *cls = (int *)mxGetPr(mxcls);
	int max_cls = (int)mxGetScalar(mxmax_cls);
	float lambda_L2 = (float)mxGetScalar(mxlambda_L2);
	float lambda_L1 = (float)mxGetScalar(mxlambda_L1);
	int norm = (int)mxGetScalar(mxnorm);
	int iteration = (int)mxGetScalar(mxiter);
	
	mxArray *w = mxCreateNumericMatrix(dim, max_cls, mxSINGLE_CLASS, mxREAL);
	float *wPtr = (float *)mxGetPr(w);
	float *wPtr0 = (float *)mxGetPr(mxw0);	
	for (int i=0; i<dim*max_cls; i++)
	{
		wPtr[i] = wPtr0[i];
	}

	if (norm == 2)
	{
		rank_pegasos_L2(x, pair, wPtr, cls, max_cls, lambda_L2, dim, pair_len, iteration);
	}
	if (norm == 1)
	{
		rank_pegasos_L1(x, pair, wPtr, cls, max_cls, lambda_L2, lambda_L1, dim, pair_len, iteration);
	}
	if (norm == 0)
	{
		rank_pegasos_L0(x, pair, wPtr, cls, max_cls, lambda_L2, lambda_L1, dim, pair_len, iteration);
	}
	
	mxArray *diff = mxCreateDoubleScalar(0);
	double *diffPtr = (double *)mxGetPr(diff);	
	for (int i=0; i<dim*max_cls; i++)
	{
		*diffPtr += (wPtr[i]-wPtr0[i])*(wPtr[i]-wPtr0[i]);
	}
	*diffPtr = sqrt(*diffPtr);
	
	plhs[0] = w;	
	plhs[1] = diff;	
}