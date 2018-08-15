#include "MyKernel.h"

__attribute__((max_global_work_dim(0)))
__kernel void MyKernel(
                  __global double * restrict Result,
                  __global double * restrict Coefs, 
                  unsigned numBoolFactors,
                  __global const unsigned * restrict knownPatterns, //size m
                  unsigned const m 
                  )
{

#if EMULATOR > 0
    printf("OpenCL: MyKernel, m=%d numBF=%d\n",(unsigned)m, numBoolFactors);
#endif
	uchar bitwidth = numBoolFactors - PBITS - VBITS;
	ushort bitmask = (1 << bitwidth) - 1;
	//send the config data to the autorun kernels
    struct chconfig_struct cfg_val;
    cfg_val.v = (__global double * restrict)v;
    cfg_val.numBoolFactors = (short)numBoolFactors;
    #pragma unroll
    for (char id=0; id < NODE; id++) {
		write_channel_intel(config_ch[id], cfg_val);
	}

#if EMULATOR > 2
    printf("config information sent to autorun kernels\n");
#endif	

	//send the Result pointer to the writer kernel
	struct chwritecfg_struct cfgwr_val;
	cfgwr_val.M= (__global double * restrict)Result;
	write_channel_intel(cfgwr_ch, cfgwr_val);
	
	//fetch knownPatterns
    
    unsigned kP_local[MMAX];
    
    unsigned wr_index = 0;	
	
	for (unsigned int i=0; i < (unsigned)m; i++) {
	
		kP_local[i]=knownPatterns[i];

    }
#if EMULATOR > 2
    printf("knownPattern Loaded\n");
#endif	

 //form the nested loop to iterate on Mbig matrix (size m by m) 
	   //make sure you cover triangular 
  	unsigned j=0;
  	unsigned count=0;
  	unsigned maxcount = (((unsigned)m*(ushort)m) >> 1) + ((unsigned)m>>1) - NODE;
    for (unsigned int i=0; i < (unsigned)m; ) {

		//calculate t_x and then lbitVector (all local mem)
		unsigned t_i = kP_local[i];
		unsigned t_j = kP_local[j];
		BitVector32 lBitVector = t_i | t_j;
		//modify the lbitVector for split
		BitVector32 pBitVector = lBitVector & ((1<<(numBoolFactors - PBITS))-1);
		BitVector32 pBitVectorv = pBitVector >> VBITS;
		BitVector32 mBitVector = pBitVectorv;
		
		#if EMULATOR > 1
			printf("i=%d j=%d lBitVector=%X pBitVector=%X pBitVectorv=%X mBitVector=%X mask=%X\n",
			i,j,lBitVector,pBitVector, pBitVectorv, (mBitVector & bitmask), ((mBitVector >> 16) & (bitmask)));
		#endif
		
		struct chdata_struct chdata;
		chdata.pattern = (((lBitVector >> (numBoolFactors - PBITS))<<VBITS) | (pBitVector & VMASK )); 
		chdata.mask = (mBitVector >> 16) & bitmask;


		//form the superset iterator 
		
		SuperSetIterator_init(&chdata.citer, (mBitVector & bitmask), bitwidth);

	
			
		unsigned offset = ((unsigned)m*(ushort)i) + j; // since m and i are less 9 bits each, the MSB is not used for offset value.
		chdata.offset = offset;
		if (count >= maxcount) chdata.offset=(offset | (1<<31)); //set MSB if final iteration
		switch (count & (NODE -1)) {
			case 0: write_channel_intel(data_ch[0],chdata);	break; //bool write_valid=write_channel_nb_intel
			#if NODE > 1
			case 1: write_channel_intel(data_ch[1],chdata); break;
			#endif
					
		}
	   	j++; 
	   	count++;

	   	if (j== (unsigned)m) {
	   		i++;
	   		j=i;
   		}

		#if EMULATOR > 1
		//printf("iteration data sent to autorun kernels\n");
		#endif	   		
	} 
	
	uchar notify_val;
	notify_val=read_channel_intel(notify_ch);

	
}    



__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(NODE)))
__kernel void superset_iterator() {

	double v_local[NMAX_SPLIT][PEVE]; //__attribute__ ((numbanks(PEVE))); 

	int node_id = get_compute_id(0);
	__global double8* restrict ddr_access_pointer;
	bool final_iter =1;
	//#pragma max_concurrency 1
	while(1) {
	
		//read config channel
		#if EMULATOR > 2
			printf("[%d] waiting for cfg data\n",node_id);
		#endif	
		struct chconfig_struct cfg_val = read_channel_intel(config_ch[node_id]);
		#if EMULATOR > 2
			printf("[%d] cfg received \n",node_id);
		#endif	
				
		unsigned int n = 1 << cfg_val.numBoolFactors;
		unsigned n_split = n / PEVE;
		ddr_access_pointer = (__global double8 *restrict)cfg_val.v;
		
		
		//load v_local
		for (unsigned int wr_index=0; wr_index < n/8; wr_index++) {

			double8 temp_val = *ddr_access_pointer;

			#pragma unroll
			for (char vi=0; vi < VE; vi++) {		
				v_local[wr_index & (n_split-1)][(wr_index >> (cfg_val.numBoolFactors - PBITS - VBITS))*VE + vi ]=temp_val[vi];
			}

			ddr_access_pointer++;
			final_iter = 0;
		}
		
		#if EMULATOR > 2
			printf("[%d] cfg is complete \n",node_id);
		#endif				
		#if EMULATOR > 0
			unsigned count =0;
		#endif
		
		
		while (final_iter == 0) {	

			//read data channel
			struct chdata_struct chdata = read_channel_intel(data_ch[node_id]);
			#if EMULATOR > 1
			printf("[%d]: count=%d, received new data packet, offset = %x, mbitVector = %x \n", node_id, count, chdata.offset, (chdata.citer._bitvector & chdata.citer._mask));
			#endif
			
			#if EMULATOR > 0
			if (chdata.pBitVectorv != (chdata.citer._bitvector & chdata.citer._mask)) {
				printf("[%d]: count=%d, mBitVectorv=%x, _bitvector=%x, _mask=%x \n",node_id,count,chdata.pBitVectorv, chdata.citer._bitvector, chdata.citer._mask);
			}
			#endif
			
			bool PEVEcond[PEVE*MPORT];
			#pragma unroll
			for (uchar pi=0; pi < PEVE; pi++) {
	   	 		PEVEcond[pi*MPORT]=((chdata.pattern | pi) == pi) ? 1 : 0;

			}	
					
			
			#if EMULATOR > 1
			printf("PEVEcond=");
			for (ushort pi=0; pi < PEVE*MPORT; pi++) {
				printf("%d",PEVEcond[pi]);
			}
			printf("\n");
			#endif 			
			//do the processing
			double acc = 0;
					
			#pragma unroll 1
			for (unsigned iter=0; ((iter == 0) || SuperSetIterator_isValid(&chdata.citer)); iter++)  {
				ushort perm;
				ushort perm0=0;					
				char ctrl=0;
				if (iter==0) {
					perm = (ushort)(chdata.citer._bitvector & chdata.citer._mask);
					ctrl = 1; }
				else {
					perm =(ushort)SuperSetIterator_get(&chdata.citer);
				}
				
	
				
				#if EMULATOR > 1
					printf("[%d]: count=%d iter=%d  perm=%X perm0=%X mask<<1=%X ~mask=%X \n",node_id, count, iter,perm, perm0,(ushort)(chdata.mask << 1), (ushort)(~chdata.mask) );
				#endif

				double preadd[PEVE*MPORT];
				
				#pragma unroll
		   	 	for (uchar pi=0; pi < PEVE; pi++) {
		   	 		preadd[MPORT*pi]=(PEVEcond[MPORT*pi]==1) ? v_local[perm][pi] : 0;
		   	 		#if MPORT == 2
					preadd[MPORT*pi+1]=(PEVEcond[MPORT*pi+1]==1) ? v_local[perm0][pi] : 0;
					#endif
				}
			
				acc = addacc (preadd,  ctrl
				#if EMULATOR > 0 
		          	            , acc
		        #endif  
		        #if EMULATOR == 0
		                        , 0
		        #endif
				);
				if (iter > 0) {
					SuperSetIterator_next(&chdata.citer);
				}
			}
			#if EMULATOR > 0
			if (fabs((acc-chdata.accref)/acc)>0.01)
            {
                printf("[%d]: count=%d acc and accRef do not match %f %f\n", node_id, count,  acc, chdata.accref);
                acc = chdata.accref; // fix it
            }
            #endif
            struct chMdata_struct wrdata;
            wrdata.offset = chdata.offset;
            wrdata.result = acc;
			write_channel_intel(write_ch[node_id],wrdata);
			
			final_iter = (chdata.offset >> 31);// use the MSB
			#if EMULATOR > 1
				printf("[%d]: iteration %d is complete final_iter=%d\n",node_id, count,final_iter);
			#endif		
			#if EMULATOR > 0
				count++;
			#endif		
		}
		#if EMULATOR > 1
			printf("[%d]: completed %d iterations \n",node_id, count--);
		#endif			

	}
}	


__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void Mwriter() {

	__global double* restrict M;
	uchar final_iter =1;
	
	while(1) {
		//read cfgwr channel
		#if EMULATOR > 2
			printf("[WR] waiting for write cfg data\n");
		#endif	
		struct chwritecfg_struct cfg_val = read_channel_intel(cfgwr_ch);
		#if EMULATOR > 2
			printf("[WR] wr cfg received \n");
		#endif		
		M = (__global double *restrict)cfg_val.M;
		if (M > 0) final_iter = 0;
		uchar count = 0;
		
		while (final_iter < NODE) {
			
			bool read_valid=0;
			unsigned offset=0;
			double result;
			struct chMdata_struct wrdata;
		
			switch (count & (NODE -1)) {
				case 0: {
					wrdata = read_channel_nb_intel(write_ch[0], &read_valid);
					break;
				}
				#if NODE > 1
				case 1: {
					wrdata = read_channel_nb_intel(write_ch[1], &read_valid);
					break;
				}
				#endif
						
			}
			
			if (read_valid == 1) {
				M[wrdata.offset & 0x7FFFFFFF] = wrdata.result;
				final_iter += (wrdata.offset >> 31);
			}
			count++;
		}
		#if EMULATOR > 1
			printf("[WR]: completed writing to memory \n");
		#endif			
		write_channel_intel(notify_ch,final_iter);		
	}
	
}


