//some numeric tests of xlns16lpvip32.cpp and xlns16monte.cpp
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "xlns16.cpp"
#include "xlns32lpvip.cpp"
#include "xlns16monte.cpp"  //monte,if needed, must be included after lpvip

float rndfp[10000];
xlns16 rndlns16[10000];
float rndfpx[10000];
xlns16 rndlns16x[10000];

void initrndfp(float rndfp[],int limit)
{
        int i;
        for (i=0; i<limit; i++)
        {
           rndfp[i] = 4.0*((float) rand())/RAND_MAX - 1.0;
           rndlns16[i] = fp2xlns16(rndfp[i]);
           rndfpx[i] = 4.0*((float) rand())/RAND_MAX - 1.0;
           rndlns16x[i] = fp2xlns16(rndfpx[i]);
           //printf("%f\n",rndfp[i]);
        }
}
float frndtest1fp(int limit,float rndfp[])
{
	float sum;
	int i;

	sum = 0;
	for (i=1; i<=limit; i++)  //scaled down to 100
	{
		sum += rndfp[i];
	}
	//printf("rndtest1fp sum=%f\n", sum);
        return sum;
}

float xfrndtest1fp(int limit,float rndfp[],float rndfpx[])
{
	float sum;
	int i;

	sum = 0;
	for (i=1; i<=limit; i++)  //scaled down to 100
	{
		sum += rndfp[i]*rndfpx[i];
	}
	//printf("rndtest1fp sum=%f\n", sum);
        return sum;
}

float frndtest1bf16(int limit,float rndfp[])
{
	float sum;
	int i;
        int bfval;

	sum = 0;
	for (i=1; i<=limit; i++)  //scaled down to 100
	{
                bfval = 0xFFFF0000&*((unsigned int *) &rndfp[i]);
		sum +=  *((float *) &bfval);
	}
	//printf("rndtest1fp sum=%f\n", sum);
        return sum;
}

float xfrndtest1bf16(int limit,float rndfp[], float rndfpx[])
{
	float sum;
	int i;
        int bfval, bfvalx;

	sum = 0;
	for (i=1; i<=limit; i++)  //scaled down to 100
	{
                bfval = 0xFFFF0000&*((unsigned int *) &rndfp[i]);
                bfvalx = 0xFFFF0000&*((unsigned int *) &rndfpx[i]);
		sum += ( *((float *) &bfval) ) * ( *((float *) &bfvalx) );
	}
	//printf("rndtest1fp sum=%f\n", sum);
        return sum;
}


int main()
{
      int limit;
      float rfp;
      float rlns,rerr;
      float rlns16monte,rerrmonte;
      float rlns32lpvip,rerr32lpvip;
      float rfp16,rerr16;
      printf("   n   fp32(exact)     fp32/bf16    rerr       xlns16_sum   rerr     xlns16_sum_monte rerr  xlns16_sum_lpvip32 rerr\n");
      for (limit=1000; limit<=10000; limit+=1000)
      {
        initrndfp(rndfp,limit);
	rfp = frndtest1fp(limit, rndfp);
	rlns = xlns162fp(xlns16_sum(rndlns16, limit));
        rerr = fabs((rfp-rlns)/rfp);
	rfp16 = frndtest1bf16(limit, rndfp);
        rerr16 = fabs((rfp-rfp16)/rfp);
	rlns16monte = xlns162fp(xlns16_sum_monte(rndlns16, limit));
        rerrmonte   = fabs((rfp-rlns16monte)/rfp);
	rlns32lpvip = xlns162fp(xlns16_sum_lpvip32(rndlns16, limit));
        rerr32lpvip = fabs((rfp-rlns32lpvip)/rfp);
        printf("%5i %12.6f   %12.6f %8.6f   %12.6f %8.6f  %12.6f %8.6f  %12.6f %8.6f\n",
              limit,rfp,rfp16,rerr16,rlns,rerr,rlns16monte,rerrmonte,rlns32lpvip,rerr32lpvip);
     }
     printf("\n");
      printf("   n   fp32(exact)     fp32/bf16    rerr      xlns16_vec_dot rerr    _vec_dot_monte rerr    _vec_dot_lpvip32 rerr\n");
     for (limit=1000; limit<=10000; limit+=1000)
      {
        initrndfp(rndfp,limit);
	rfp = xfrndtest1fp(limit, rndfp, rndfpx);
	rlns = xlns162fp(xlns16_vec_dot(rndlns16, rndlns16x, limit));
        rerr = fabs((rfp-rlns)/rfp);
	rfp16 = xfrndtest1bf16(limit, rndfp, rndfpx);
        rerr16 = fabs((rfp-rfp16)/rfp);
	rlns16monte = xlns162fp(xlns16_vec_dot_monte(rndlns16, rndlns16x, limit));
        rerrmonte   = fabs((rfp-rlns16monte)/rfp);
	rlns32lpvip = xlns162fp(xlns16_vec_dot_lpvip32(rndlns16, rndlns16x, limit));
        rerr32lpvip = fabs((rfp-rlns32lpvip)/rfp);
        printf("%5i %12.6f   %12.6f %8.6f   %12.6f %8.6f  %12.6f %8.6f  %12.6f %8.6f\n",
              limit,rfp,rfp16,rerr16,rlns,rerr,rlns16monte,rerrmonte,rlns32lpvip,rerr32lpvip);
     }
     return 1;

}


