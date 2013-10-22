#ifndef CHAMBER_H
#define CHAMBER_H


#include <iostream>
#include <fstream>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "DTRandom.h"	//Adapted some code for GPGPU


#define V_PI 3.141592653589793f		//3.1415926535897932384626433832795028841971
#define V_E  2.718281828459045f


using namespace std;


unsigned long long int Rand_SetUp()
{
	srand(time(NULL));				//Seed the "rand()" function
	unsigned long long int a;
	for(unsigned short i=0;i<100;i++) {rand();}		//Generate more ran numbers to better randomize than just from time seed.
	a = (double(rand())/RAND_MAX)*(LONG_MAX);		//a better random number
	return a;
}

__global__ void RNGinit(curandState *Sts, long long seed) {
	const unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
	curand_init(seed, tid, 0, Sts+tid);				//Initialize: Seed, Sequence #, Order #, Address of PRNG_State		&Sts[tid] = Sts+tid;
	/*curand_init(seed+tid, 0, 0, Sts+tid);*/	//A MUCH faster set-up method (potentially rare cases of some threads getting correlated RNs)
}

__device__ float curndUniClosed(curandState *st) {
	return curand(st)*(1.0/4294967295.0);
}	//Uniform Rand # in [0,1]

__device__ float curndUniOpen(curandState *st) {
	return (double(curand(st)) + 0.5)*(1.0/4294967296.0);
}	//Uniform Rand # in (0,1)

__device__ float curndUniHalf(curandState *st) {
	return curand(st)*(1.0/4294967296.0);
}	//Uniform Rand # in [0,1)

__device__ float curndUniHalf53(curandState *st) {
	unsigned long a=curand(st)>>5, b=curand(st)>>6;
	return (a*67108864.0+b)*(1.0/9007199254740992.0);
}	//Uniform Rand # in [0,1) with 53-bit resolution

__device__ double curndGamInt(int na, curandState *st) {
	if(na < 12) {
		int i;
		double product = 1.0;
		for(i=0; i<na; i++) {
			product *= curndUniOpen(st);
		}
		return -log(product);
	}
	else {
		//Want 'return GammaLarge(na);'
		double x, y, v;
		double sqa = sqrt(2.0*na - 1);

		do {
			do {
				y = tan(V_PI*curndUniOpen(st));
				x = sqa*y + na-1.0;
			} while(!(x > 0));

			v = curndUniOpen(st);
		} while(v > (1 + y*y)*exp((na-1.0)*log(x/(na-1.0)) - sqa*y));

		return x;		//Equivalent to 'return GammaLarge(na);'
	}
}	//Folded GammaLarge into GammaInteger from DTRandom files

__device__ double curndGam(double a, curandState *st) {
	unsigned int na=(int)(floor((double) (a)));
	double toReturn;

	if(a==na) {
		toReturn = curndGamInt(na, st);	//Equivalent to 'x1 = Gamma(a, 1.0);'
	}
	else if(na==0) {
		//Want 'toReturn = GammaFraction(a);'
		double q, u, v;
		double p = V_E/(a+V_E);
		do {
			u = curndUniOpen(st);
			v = curndUniOpen(st);
			if(u < p) {
				toReturn = exp((1/a) * log(v));
				q = exp(-toReturn);
			}
			else {
				toReturn = 1.0 - log(v);
				q = exp((a-1.0) * log(toReturn));
			}
		} while(curndUniOpen(st) >= q);
		//toReturn has been set to GammaFraction(a)
	}
	else {
		//Want 'toReturn = GammaFraction(a - na);'
		double q, u, v;
		double p = V_E/(a-na+V_E);
		do {
			u = curndUniOpen(st);
			v = curndUniOpen(st);
			if(u < p) {
				toReturn = exp((1/(a-na)) * log(v));
				q = exp(-toReturn);
			}
			else {
				toReturn = 1.0 - log(v);
				q = exp((a-na-1.0) * log(toReturn));
			}
		} while(curndUniOpen(st) >= q);
		//toReturn has been set to GammaFraction(a-na);
		toReturn += curndGamInt(na, st);
	}

	return toReturn;
}	//Folded GammaFraction (calls GammaInteger) into Gamma from DTRandom (assume b=1 in Gamma)

__device__ unsigned curndPoisson(double mu, curandState *st) {
	unsigned int m, k=0u;
	double X;

	while(mu>10.0) {
		m = (unsigned int) (mu*(7.0/8.0));
		X = curndGamInt(m, st);
	
		if(X >= mu) {
			//Want 'return k + Binomial(mu/X, m-1);'
			double BX, p=mu/X;
			int n=m-1;
			unsigned int a, b, Bk=0u;

			while(n>10) {
				a = 1+(n/2);
				b = 1+n-a;
				//Want 'BX = Beta(a, b);'
				double x1=curndGam(a, st), x2=curndGam(b, st);
				BX = x1/(x1+x2);	//Equivalent to 'BX = Beta(a, b);'

				if(BX >= p) {
					n = a-1;
					p /= BX;
				}
				else {
					Bk += a;
					n = b-1;
					p = (p-BX)/(1-BX);
				}
			}

			int i;
			for(i=0; i<n; i++) {
				if(curndUniClosed(st) < p) {	Bk++;	}
			}

			return k + Bk;	//Equivalent to 'return k + Binomial(mu/X, m-1);'
		}
		else {
			k += m;
			mu -= X;
		}
	}

	double emu = exp(-mu), prod = 1.0;
	do {
		prod *= curndUniClosed(st);
		k++;
	} while(prod>emu);

	return k-1;
}	//Poisson Rand # with average = mu		Folded Binomial & Beta (calls GammaInteger & Gamma) into Poisson from DTRandom

/*struct Mutex {
	int *mutex;

	Mutex(void) {
		int state=0;
		cudaMalloc((void **) &mutex, sizeof(int));
		cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
	}
	~Mutex(void) {
		cudaFree(mutex);
	}

	__device__ void lock(void) {
		while(atomicCAS(mutex, 0, 1) != 0) {}
	}
	__device__ void unlock(void) {
		atomicExch(mutex, 0);
	}
};*/

__global__ void ABlank(float *Ax, float *Ay, float *Az, int *aopen, int *anear) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if(tid < AMAX) {
		Ax[tid] = FLT_MAX;
		Ay[tid] = -1.0f;
		Az[tid] = -1.0f;
	}	//Blank A data
	if(tid < OPNMAX) {	aopen[tid] = -1;	}	//Blank Open IDs
	if(tid < NRMAX) {	anear[tid] = -1;	}	//Blank Near IDs
}

__global__ void AInit(const int ainit, float *Ax, float *Ay, float *Az, curandState *Sts) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	float ex=0.0f, wy=0.0f, ze=0.0f, buff=0.0f;
	curandState tState = Sts[tid];

	if(tid < ainit) {
		while(buff <= CLLR2) {	
			ex = curndUniClosed(&tState)*LENGTH;	//create locally
			wy = curndUniClosed(&tState)*LENGTH;
			ze = curndUniClosed(&tState)*LENGTH;
			buff = (ex-CLLX)*(ex-CLLX) + (wy-CLLY)*(wy-CLLY) + (ze-CLLZ)*(ze-CLLZ);	//distance from cell center ^2
		}	//try until outside cell
		Ax[tid] = ex;
		Ay[tid] = wy;
		Az[tid] = ze;		//save globally
	}	//Create A particle

	Sts[tid] = tState;
}

__global__ void AOpenNearInit(const int ainit, float *Ax, float *Ay, float *Az, int *Naopen, int *aopen, int *Nanear, int *anear) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x, i=0;
	float dst2=0.0f, temp=0.0f;
	__shared__ int lclNaopen, lclaopen[ATHREADS], j;

	if(threadIdx.x==0)	{	lclNaopen=0;	j=0;	}
	lclaopen[threadIdx.x] = -1;
	__syncthreads();

	if(tid<ainit) {
		temp = Ax[tid];
		dst2 = (temp-CLLX)*(temp-CLLX);	//dx^2
		if(dst2 < NRTHRSH2) {
			temp = Ay[tid];
			dst2 = dst2 + (temp-CLLY)*(temp-CLLY);	//dx^2 + dy^2
			if(dst2 < NRTHRSH2) {
				temp = Az[tid];
				dst2 = dst2 + (temp-CLLZ)*(temp-CLLZ);	//dx^2 + dy^2 + dz^2
				if(dst2 < NRTHRSH2) {
					i = atomicAdd(Nanear, 1);
					anear[i] = tid;		//Save ID in global NearID array
				}	//near z
			}	//near y
		}	//near x
	}	//Get & Check if position is near

	if((tid >= ainit) && (tid<AMAX)) {
		i = atomicAdd(&lclNaopen, 1);
		lclaopen[i] = tid;
	}	//Save ID in local OpenID array
	__syncthreads();

	if((threadIdx.x == 0) && (lclNaopen > 0)) {
		j = atomicAdd(Naopen, lclNaopen);
	}	//Get last size of global OpenID array
	__syncthreads();
	if(threadIdx.x < lclNaopen) {
		aopen[j + threadIdx.x] = lclaopen[threadIdx.x];
	}	//Move local OpenID to global OpenID
}

__global__ void BCInit(const int bcinit, float *BCx, float *BCy, float *BCz, bool *bnd, const int first, const bool whch, curandState *Sts) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	float theta=0.0f, phi=0.0f;
	curandState tState = Sts[tid];

	if(tid<bcinit) {
		theta = 2.0f*curndUniOpen(&tState) - 1.0f;
		theta = acosf(theta);
		phi = 2.0f*V_PI*curndUniOpen(&tState);
		BCx[first+tid] = CLLX + sinf(theta)*cosf(phi)*CLLR;
		BCy[first+tid] = CLLY + sinf(theta)*sinf(phi)*CLLR;
		BCz[first+tid] = CLLZ + cosf(theta)*CLLR;

		bnd[first+tid] = whch;
	}

	Sts[tid] = tState;
}

__global__ void BCBoundInit(const int bcinit, const int first, bool *bnd, const bool whch) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if(tid<bcinit) {
		bnd[first+tid] = whch;
	}
}

__global__ void Binding(float *Ax, float *Ay, float *Az, bool *bnd, const float *BCx, const float *BCy, const float *BCz, int *Naopen, int *aopen, int *opnmutex, int *Nanear, int *anear, curandState *Sts) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x, i=0, j=0;
	float dst2=0.0f, temp=0.0f;
	bool bound=true;	//default = do nothing
	__shared__ int lclanear[NRMAX], lclNanear;
	__shared__ float lclax[NRMAX], lclay[NRMAX], lclaz[NRMAX];

	if(threadIdx.x == 0) {	lclNanear = *Nanear;	}
	__syncthreads();
	for(i=threadIdx.x; i<lclNanear; i+=blockDim.x) {
		lclanear[i] = anear[i];
		if(lclanear[i] >= 0) {	lclax[i] = Ax[lclanear[i]];	}	//valid near ID
		else {	lclax[i] = FLT_MAX;	}	//invalid near ID
		lclay[i] = -1.0f;
		lclaz[i] = -1.0f;
	}	//Get the IDs & x-values of near As
	__syncthreads();
	if(tid < BCMAX) {	bound = bnd[tid];	}	//get your bound state

	if(!bound) {
		temp = BCx[tid];	//get your x-value
		for(i=0; i<lclNanear; i++) {

			dst2 = (lclax[i] - temp)*(lclax[i] - temp);	//dx^2
			if(dst2 <= RBIND2) {
				j = lclanear[i];	//secondary purpose of 'j' (primary is in binding case)
				//printf("tid: %i\tis close in x: %f\n\tB: (%f, ?, ?)\n\tA: (%f, ?, ?)\n", tid, dst2, temp, lclax[i]);
				if(lclay[i] < 0.0f) {	lclay[i] = Ay[j];	}	//download lclay[i]
				dst2 = dst2 + (lclay[i] - BCy[tid])*(lclay[i] - BCy[tid]);	//dx^2 + dy^2
				if(dst2 <= RBIND2) {
					if(lclaz[i] < 0.0f) {	lclaz[i] = Az[j];	}	//download lclaz[i]
					dst2 = dst2 + (lclaz[i] - BCz[tid])*(lclaz[i] - BCz[tid]);	//dx^2 + dy^2 + dz^2
					if((dst2 <= RBIND2) && (curndUniHalf(Sts+tid) < PON)) {
						if(bound) {	printf("Competition type 2 on BCid = %i\n", tid);	}
						else if(atomicCAS(anear+i, j, -1) != j) {	printf("Competition type 1 on BCid = %i\n", tid);	}	//well well, aren't YOU a fancy pants
						else {
							bound = true;
							bnd[tid] = true;	//mark BC particle bound
							Ax[j] = FLT_MAX;	//mark A particle invalid
							anear[i] = -1;		//clear the index in global nearID array
							j=-1;
							while(j<0) {
								if(atomicCAS(opnmutex, 0, 1) == 0) {
									j = atomicAdd(Naopen, 1);			//increment count of openIDs	//primary purpose of 'j'
									//atomicExch(aopen+j, lclanear[i]);	//append this nearID to openID array
									if(j<OPNMAX) {
										atomicExch(aopen+j, lclanear[i]);	//append this nearID to openID array
									}
									else {	printf("Open array Overflow: %i\n", j+1);	}
									atomicExch(opnmutex, 0);		//unlock
									//printf("+1 Ste2*\n");
								}		//lock successful
							}	//spin until lock
						}
					}	//near z & successful rxn
				}	//near y
			}	//near x

		}	//compare each near A
	}	//Continue iff unbound
}

__global__ void Unbinding(float *Ax, float *Ay, float *Az, bool *bnd, const float *BCx, const float *BCy, const float *BCz, int *Naopen, int *aopen, int *opnmutex, curandState *Sts) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x, i=-1;	//i=-1 to enter lock loop
	float ex=0.0f, wy=0.0f, ze=0.0f, theta=0.0f, phi=0.0f, nwex=0.0f, nwwy=0.0f, nwze=0.0f;
	bool bound=false;	//default = do nothing

	if(tid < BCMAX) {	bound = bnd[tid];	}
	if((bound) && (curndUniHalf53(Sts+tid) < POFF)) {
		ex = BCx[tid];	//download receptor position
		wy = BCy[tid];
		ze = BCz[tid];
		theta = 2.0f*curndUniOpen(Sts+tid) - 1.0f;	//roll rand theta & phi
		theta = acosf(theta);
		phi = 2.0f*V_PI*curndUniOpen(Sts+tid);
		nwex = ex + sinf(theta)*cosf(phi)*RUNBIND;	//calculate new A position
		nwwy = wy + sinf(theta)*sinf(phi)*RUNBIND;
		nwze = ze + cosf(theta)*RUNBIND;

		theta = (nwex-CLLX)*(nwex-CLLX) + (nwwy-CLLY)*(nwwy-CLLY) + (nwze-CLLZ)*(nwze-CLLZ);	//distance to cell center
		if(theta < CLLR2) {
			//receptor position: (ex, wy, ze) = vector x (intersection point)
			theta = (2.0f/CLLR2)*( (nwex-ex)*(CLLX-ex) + (nwwy-wy)*(CLLY-wy) + (nwze-ze)*(CLLZ-ze) );	//magnitude of vector from p to f
			nwex = nwex + (theta)*(ex-CLLX);
			nwwy = nwwy + (theta)*(wy-CLLY);
			nwze = nwze + (theta)*(ze-CLLZ);	//Final Position
		}	//reflect out of cell

		while(bound) {
			if(atomicCAS(opnmutex, 0, 1) == 0) {
				i = atomicSub(Naopen, 1);		//decrease count of openIDs
				//i = atomicExch(aopen+i-1, -1);	//get last openID & replace with invalid
				if(i>0) {
					i = atomicExch(aopen+i-1, -1);	//get last openID & replace with invalid
				}
				else {	printf("Open array Underflow: %i\n", i-1);	}
				atomicExch(opnmutex, 0);	//unlock
				bound=false;	//exit loop
				//printf("-1 Ste2*\n");
			}	//lock successful
		}	//spin until lock
		Ax[i] = nwex;	//write & validate new A particle
		Ay[i] = nwwy;
		Az[i] = nwze;
		bnd[tid] = false;	//mark BC particle unbound
	}	//Continue iff bound & successful unbinding
}

__global__ void DiffusionA(float *Ax, float *Ay, float *Az, int *Nanear, int *anear, curandState *Sts) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x, i=0;
	float2 pair=make_float2(0.0f, 0.0f);
	float3 init=make_float3(FLT_MAX, 0.0f, 0.0f), prop=make_float3(0.0f, 0.0f, 0.0f);	//default = do nothing	(init.x = FLT_MAX)
	//curandState tState;

	if(tid < AMAX) {	init.x = Ax[tid];	}	//download x-position
	if(init.x < 2*LENGTH) {
		//tState = Sts[tid];	//Get RNG
		init.y = Ay[tid];	//download rest of position
		init.z = Az[tid];
		pair = curand_normal2(Sts+tid);	//Diffuse in 3D
		prop.x = init.x + pair.x*DA_FACTOR;
		prop.y = init.y + pair.y*DA_FACTOR;
		pair.x = curand_normal(Sts+tid);
		prop.z = init.z + pair.x*DA_FACTOR;

		pair.x = (prop.x-CLLX)*(prop.x-CLLX);	//dx^2
		if(pair.x < NRTHRSH2) {
			pair.x = pair.x + (prop.y-CLLY)*(prop.y-CLLY);	//dx^2 + dy^2
			if(pair.x < NRTHRSH2) {
				pair.x = pair.x + (prop.z-CLLZ)*(prop.z-CLLZ);	//dx^2 + dy^2 + dz^2
				if(pair.x < CLLR2) {
					pair.x = (prop.x-init.x)*(prop.x-init.x) + (prop.y-init.y)*(prop.y-init.y) + (prop.z-init.z)*(prop.z-init.z);	//temporarily, pair.x = ||(p-i)||^2 = ||v||^2
					pair.y = (prop.x-init.x)*(CLLX-init.x) + (prop.y-init.y)*(CLLY-init.y) + (prop.z-init.z)*(CLLZ-init.z);		//temporarily, pair.y = v DOT (c - i)
					pair.y = ( pair.y - sqrtf(pair.y*pair.y + pair.x*(CLLR2-(CLLX-init.x)*(CLLX-init.x)-(CLLY-init.y)*(CLLY-init.y)-(CLLZ-init.z)*(CLLZ-init.z))) )/pair.x;
					if( !(pair.y == pair.y) ) {
						printf("%u QNAN:\tinit = (%f, %f, %f), prop = (%f, %f, %f)\n", tid, init.x, init.y, init.z, prop.x, prop.y, prop.z);
						pair.y = 0.0f;
					}	//pair.y = QNAN
						//temporarily, pair.y = tx parameter value at intersection
					init.x = init.x + pair.y*(prop.x - init.x);
					init.y = init.y + pair.y*(prop.y - init.y);
					init.z = init.z + pair.y*(prop.z - init.z);
						//init = vector x (intersection point)
					pair.x = (2.0f/CLLR2)*( (prop.x-init.x)*(CLLX-init.x) + (prop.y-init.y)*(CLLY-init.y) + (prop.z-init.z)*(CLLZ-init.z) );	//magnitude of vector from p to f
					prop.x = prop.x + (pair.x)*(init.x-CLLX);
					prop.y = prop.y + (pair.x)*(init.y-CLLY);
					prop.z = prop.z + (pair.x)*(init.z-CLLZ);	//Final Position
					pair.x = (prop.x-CLLX)*(prop.x-CLLX) + (prop.y-CLLY)*(prop.y-CLLY) + (prop.z-CLLZ)*(prop.z-CLLZ);	//new distance^2 to center of cell
				}	//inside the cell, reflect out
				if(pair.x < NRTHRSH2) {
					i = atomicAdd(Nanear, 1);
					//anear[i] = tid;
					if(i<NRMAX) {	anear[i] = tid;	}
					else {	printf("Near array overflow: %i.\t\t(%f, %f, %f) => %f\n", i+1, prop.x, prop.y, prop.z, pair.x);	}
				}	//near z (but not inside cell), save ID to near array
			}	//near y
		}	//near x

		Ax[tid] = prop.x;	//update new position
		Ay[tid] = prop.y;
		Az[tid] = prop.z;
		//Sts[tid] = tState;	//save RNG globally
	}	//Diffuse iff valid
}

__global__ void DiffusionBC(float *BCx, float *BCy, float *BCz, curandState *Sts) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	float2 temp=make_float2(0.0f, 0.0f);
	float3 init=make_float3(0.0f, 0.0f, 0.0f);
	//curandState tState;
	
	if(tid < BCMAX) {
		//tState = Sts[tid];	//Get RNG
		init.x = BCx[tid];	//download position
		init.y = BCy[tid];
		init.z = BCz[tid];
		temp = curand_normal2(Sts+tid);	//Diffuse in 3D
		init.x = init.x + temp.x*DBC_FACTOR;
		init.y = init.y + temp.y*DBC_FACTOR;
		temp.x = curand_normal(Sts+tid);
		init.z = init.z + temp.x*DBC_FACTOR;

		temp.x =(init.x-CLLX)*(init.x-CLLX) + (init.y-CLLY)*(init.y-CLLY) + (init.z-CLLZ)*(init.z-CLLZ);
		temp.x = sqrtf(temp.x);
		temp.y = CLLR/temp.x;	//temp.y is scaling factor
		BCx[tid] = CLLX + temp.y*(init.x - CLLX);	//update new position
		BCy[tid] = CLLY + temp.y*(init.y - CLLY);
		BCz[tid] = CLLZ + temp.y*(init.z - CLLZ);
		//init.x = CLLX + temp.y*(init.x - CLLX);
		//init.y = CLLY + temp.y*(init.y - CLLY);
		//init.z = CLLZ + temp.y*(init.z - CLLZ);

		//BCx[tid] = init.x;	//update new position
		//BCy[tid] = init.y;
		//BCz[tid] = init.z;
		//Sts[tid] = tState;
	}	//Diffuse iff valid
}

__global__ void EjectA(float *Ax, float *Ay, float *Az, int *Naopen, int *aopen, int *opnmutex) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x, i=-1;	//i=-1 to enter lock loop
	float temp=FLT_MAX;	//default = do nothing

	if(tid < AMAX) {	temp = Ax[tid];	}
	if(temp < 2*LENGTH) {
		if((temp > LENGTH) || (temp < 0.0f)) {
			Ax[tid] = FLT_MAX;	//mark A particle invalid
			while(i<0) {
				if(atomicCAS(opnmutex, 0, 1) == 0) {
					i = atomicAdd(Naopen, 1);	//increment count of openIDs
					//atomicExch(aopen+i, tid);	//append this nearID to openID array
					if(i<OPNMAX) {
						atomicExch(aopen+i, tid);	//append this nearID to openID array
					}
					else {	printf("Open array Overflow: %i\n", i+1);	}
					atomicExch(opnmutex, 0);	//unlock
				}	//lock successful
			}	//spin until lock
		}	//oob x-axis -> eject
		else {
			temp = Ay[tid];	//check oob y-axis
			if(temp > LENGTH) {
				Ay[tid] = 2*LENGTH - temp;
			}	//too large, reflect
			else if(temp < 0.0f) {
				Ay[tid] = -temp;
			}	//too small, reflect
			else {}
			temp = Az[tid];	//check oob z-axis;
			if(temp > LENGTH) {
				Az[tid] = 2*LENGTH - temp;
			}	//too large, reflect
			else if(temp < 0.0f) {
				Az[tid] = -temp;
			}	//too small, reflect
			else {}
		}
	}	//check ejection iff valid
}

__global__ void InjectA(float *Ax, float *Ay, float *Az, int *Naopen, int *aopen, int *opnmutex, const int *Ninjds, const float *injds, curandState *Sts) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x, i=0, j=0;
	curandState tState = Sts[tid];
	__shared__ int Ntoinj, k;

	if(threadIdx.x == 0) {
		if(blockIdx.x == 0) {
			Ntoinj = curndPoisson(FLUXL, &tState);
			while(atomicCAS(opnmutex, 0, 1) != 0) {}	//spin until lock
			k = atomicSub(Naopen, Ntoinj);		//decrease count of openIDs
		}	//Inject Low Side
		else if(blockIdx.x == 1) {
			Ntoinj = curndPoisson(FLUXH, &tState);
			while(atomicCAS(opnmutex, 0, 1) != 0) {}	//spin until lock
			k = atomicSub(Naopen, Ntoinj);		//decrease count of openIDs
		}	//Inject High Side
		else {	Ntoinj=0;	}
	}
	__syncthreads();
	if(threadIdx.x < Ntoinj) {
		//i = atomicExch(aopen+k-1-threadIdx.x, -1);	//get last openID & replace with invalid
		if(k>Ntoinj) {
			i = atomicExch(aopen+k-1-threadIdx.x, -1);	//get last openID & replace with invalid
		}
		else {	printf("Open array Underflow: %i\n", k-Ntoinj);	}
	}
	__syncthreads();
	if((threadIdx.x == 0) && (blockIdx.x < INJBLOCKS)) {
		atomicExch(opnmutex, 0);	//unlock
	}

	if(threadIdx.x < Ntoinj) {
		j = floor(curndUniHalf(&tState)*(*Ninjds));		//roll for index of injection distance
		if(blockIdx.x == 0) {
			Ax[i] = injds[j];
		}	//x=0.0f
		else if(blockIdx.x == 1) {
			Ax[i] = LENGTH - injds[j];
		}	//x=LENGTH
		else {}

		Ay[i] = curndUniClosed(&tState)*LENGTH;
		Az[i] = curndUniClosed(&tState)*LENGTH;
	}	//Make a new particle

	Sts[tid] = tState;
}

__global__ void BounceA(float *Ax, float *Ay, float *Az) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	float temp=FLT_MAX;	//default = do nothing

	if(tid < AMAX) {	temp = Ax[tid];	}
	if(temp < 2*LENGTH) {
		//check oob x-axis
		if(temp > LENGTH) {
			Ax[tid] = 2*LENGTH - temp;
		}	//too large, reflect
		else if(temp < 0.0f) {
			Ax[tid] = -temp;
		}	//too small, reflect
		else {}
		temp = Ay[tid];	//check oob y-axis
		if(temp > LENGTH) {
			Ay[tid] = 2*LENGTH - temp;
		}	//too large, reflect
		else if(temp < 0.0f) {
			Ay[tid] = -temp;
		}	//too small, reflect
		else {}
		temp = Az[tid];	//check oob z-axis;
		if(temp > LENGTH) {
			Az[tid] = 2*LENGTH - temp;
		}	//too large, reflect
		else if(temp < 0.0f) {
			Az[tid] = -temp;
		}	//too small, reflect
		else {}
	}	//check bounce iff valid
}


#endif	//CHAMBER_H
