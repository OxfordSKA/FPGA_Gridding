#pragma once
#pragma OPENCL_EXTENSION cl_khr_fp64 : enable  //this is only required for Double precision logic
#include "arith.h"  //  This includes the addacc32 function given below. You don't need this library
#pragma OPENCL EXTENSION cl_altera_channels : enable  //you need this

#define LOOKUP_MEM_TYPE __global const double *

#ifndef EMULATOR
#define EMULATOR 0
#endif

#define VBITS 3 //double8 usage -- don't change

#ifndef PBITS
#define PBITS 2 
#endif

#ifndef NODE
#define NODE 1
#endif

#ifndef CDEPTH
#define CDEPTH 8  //16 
#endif

#define MPORT 1

#define MAXBOOLFACTORS 16
#define NMAX (1<<MAXBOOLFACTORS)
#define MMAX 512
#define NMAX_SPLIT (1<<(MAXBOOLFACTORS - (PBITS+VBITS)))
#define PE (1<<PBITS)
#define VE (1<<VBITS)
#define PEVE (1<<(PBITS+VBITS))
#define PMASK (PE-1) // for PBITS=1, PMASK is 0x1
#define VMASK (VE -1)


typedef unsigned int BitVector32;

struct SuperSetIterator
{
    BitVector32 _bitvector;
    BitVector32 _current;
    unsigned int _mask;
};


struct chwritecfg_struct
{
	__global double * restrict M;
};

struct chconfig_struct
{
	__global double * restrict v;
	short numBoolFactors;
};

struct chdata_struct
{
	struct SuperSetIterator citer;
	//bool PEVEcond[PEVE*MPORT];
	uchar pattern;
	ushort mask;
	unsigned offset; //MSB is a flag for final_iter
	#if EMULATOR > 0
	double accref;
	unsigned pBitVectorv;
	#endif
};

struct chMdata_struct
{
	unsigned offset; //MSB is a flag for final_iter
	double result;
};

channel struct chwritecfg_struct cfgwr_ch __attribute__((depth(1)));
channel struct chconfig_struct config_ch[NODE] __attribute__((depth(1)));
channel struct chdata_struct data_ch[NODE] __attribute__((depth(CDEPTH)));
channel struct chMdata_struct write_ch[NODE] __attribute__((depth(8)));
channel uchar notify_ch __attribute__((depth(1)));


inline BitVector32 SuperSetIterator_get(struct SuperSetIterator * p)
{
    BitVector32 val = p->_mask;
    if (p->_bitvector == ~p->_current) 
        val =  ~p->_current & val;

    return val;
}

inline void SuperSetIterator_next(struct SuperSetIterator * p)
{
    p->_current = (~p->_bitvector);
}

inline bool SuperSetIterator_isValid(struct SuperSetIterator * p)
{
    return 0 != p->_current;
}

#define SUPERCAP(x, i) ((x) | (~((1 << (i)) )))

inline void SuperSetIterator_init(struct SuperSetIterator * p, unsigned bv, unsigned numBF)
{
    const unsigned x = SUPERCAP(bv, numBF);
    p->_bitvector = x;
    p->_current = (~x) & (-(x));
    p->_mask = (1<<numBF) - 1;
}

inline __global double * restrict getElement(__global double * restrict Mbig, unsigned m, unsigned i, unsigned j)
{
    return Mbig + m*i + j;
}

inline double addacc (double * in, char ctrl, double acc) { 
#if ((PBITS+VBITS)==5)
   return addacc32(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], 
                  in[8], in[9], in[10], in[11], in[12], in[13], in[14], in[15], 
                  in[16], in[17], in[18], in[19], in[20], in[21], in[22], in[23], 
                  in[24], in[25], in[26], in[27], in[28], in[29], in[30], in[31], ctrl, acc);
#endif

}




