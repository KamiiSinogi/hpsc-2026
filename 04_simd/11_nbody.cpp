#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

void print(__m512 stp)
{
	float num[16];
	_mm512_store_ps(num,stp);
	for(int i=0;i<16;i++) printf("%.4f ",num[i]);
	printf("\n");
}
int main() {
	const int N = 16;
	float x[N], y[N], m[N], fx[N], fy[N];
	for(int i=0; i<N; i++)
	{
		x[i]=drand48();
		y[i]=drand48();
		m[i]=drand48();
		fx[i]=fy[i]=0;
	}
	__m512 xvec=_mm512_load_ps(x);
	__m512 yvec=_mm512_load_ps(y);
	__m512 mvec=_mm512_load_ps(m);
	for(int i=0; i<N; i++)
	{
		__m512 stpx=_mm512_set1_ps(x[i]);
		__m512 stpy=_mm512_set1_ps(y[i]);
		__m512 rx=_mm512_sub_ps(stpx,xvec);
		__m512 ry=_mm512_sub_ps(stpy,yvec);
		__m512 r2=_mm512_add_ps(_mm512_mul_ps(rx,rx),_mm512_mul_ps(ry,ry));
		__m512 r_1=_mm512_rsqrt14_ps(_mm512_mask_blend_ps(1<<i,r2,_mm512_set1_ps(1)));
		__m512 r_2=_mm512_mul_ps(r_1,r_1);
		__m512 r_3=_mm512_mul_ps(r_1,r_2);
		__m512 dx=_mm512_mul_ps(_mm512_mul_ps(rx,mvec),r_3);
		__m512 dy=_mm512_mul_ps(_mm512_mul_ps(ry,mvec),r_3);
		fx[i]-=_mm512_reduce_add_ps(dx);
		fy[i]-=_mm512_reduce_add_ps(dy);
		printf("%d %g %g\n",i,fx[i],fy[i]);
	}
}
