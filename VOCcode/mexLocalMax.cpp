#include <math.h>
#include "mex.h"

#define INF 10000
#define EPS 0.00001

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }

static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }


int nonmaxsup(double * im, int imy, int imx, double *ts, double maxpoint, double *sup_size, double *pos, double *val)
{	
	int counter = 0;
	for (int i=0; i<(int)maxpoint; i++)
	{
		//	search for maximum
		double max_val = -INF;
		int row = 0;
		int col = 0;
		for (int j1=0; j1<imx; j1++)
		{
			for (int j0=0; j0<imy; j0++)
			{				
				if (im[j0+j1*imy]>max_val)
				{					
					max_val = im[j0+j1*imy];					 
					row = j0;
					col = j1;
				}				
			}			
		}
		if (max_val==-INF)
		{
			break;
		}

		// record
		counter++;
		val[i] = max_val;
		pos[i*4] = col+1;
		pos[i*4+1] = row+1;
		pos[i*4+2] = col+ts[1];
		pos[i*4+3] = row+ts[0];

		//	suppress
    	for (int j1=max(0,(int)(pos[i*4]-sup_size[1])); j1<min(imx,(int)(pos[i*4]+sup_size[1])); j1++)
		{
			for (int j0=max(0,(int)(pos[i*4+1]-sup_size[0])); j0<min(imy,(int)(pos[i*4+1]+sup_size[0])); j0++)
			{
				im[j0+j1*imy] = -INF;
			}
		}			
	}	
	
	return counter;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{  
	double *im = (double *)mxGetPr(prhs[0]);
	double *ts = (double *)mxGetPr(prhs[1]);
	double maxpoint = (double)mxGetScalar(prhs[2]);
	double *sup_size = (double *)mxGetPr(prhs[3]);
	const int *dims = mxGetDimensions(prhs[0]);
	int imy = mxGetM(prhs[0]);
	int imx = mxGetN(prhs[0]);	

	double *pos1 = (double *)mxCalloc(4*maxpoint, sizeof(double));
	double *val1 = (double *)mxCalloc(maxpoint, sizeof(double));

	int counter = nonmaxsup(im, imy, imx, ts, maxpoint, sup_size, pos1, val1);
	
	mxArray *mxpos = mxCreateNumericMatrix(counter, 4, mxDOUBLE_CLASS, mxREAL);
	double *pos = (double *)mxGetPr(mxpos);
	mxArray *mxval = mxCreateNumericMatrix(counter, 1, mxDOUBLE_CLASS, mxREAL);
	double *val = (double *)mxGetPr(mxval);

	for (int i=0; i<counter; i++)
	{
		val[i] = val1[i];
		for (int j=0; j<4; j++)
		{
			pos[i+j*counter] = pos1[j+i*4];
		}
	}

	plhs[0] = mxpos;
	plhs[1] = mxval;

	mxFree(pos1);
	mxFree(val1);
}