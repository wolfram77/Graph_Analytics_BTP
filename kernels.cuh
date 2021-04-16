__global__ void kerneltest(long *cstart, long *cend, long *cmemsz, long *cmember, long *crcw, 
	double *cinitial, double *crank, long *rcwgraph, long *outdeg, long *corder, long *ctemp, long *ctempg)
{
	long w = blockIdx.z*blockDim.z + threadIdx.z + (*cstart);
	long num_threads_z = blockDim.z * gridDim.z;
	for(;w < (*cend);w+=num_threads_z)
	{
		long size = cmemsz[w];
		long j = blockIdx.x*blockDim.x+threadIdx.x;
		long num_threads_x = blockDim.x * gridDim.x;
		for(;j<size;j+=num_threads_x){
			long node = cmember[ctemp[w]+j];
			long k = blockIdx.y*blockDim.y+threadIdx.y;
			long size1 = crcw[node];
			long num_threads_y = blockDim.y * gridDim.y;
			for(;k<size1;k+=num_threads_y){
				cinitial[ctempg[node]+k]=0.85*crank[rcwgraph[ctempg[node]+k]]/outdeg[rcwgraph[ctempg[node]+k]];
			}
		}
	}
}
__global__ void kerneltest1(long *cstart, long *cend, long *cmemsz, long *cmember, long *crcw, 
	double *cinitial, double *crank, long *rcwgraph, long *outdeg, long *corder, long *ctemp, long *ctempg)
{
	long w = (*cstart);
	long size = cmemsz[w];
	long j = blockIdx.x*blockDim.x+threadIdx.x;
	long num_threads_x = blockDim.x * gridDim.x;
	for(;j<size;j+=num_threads_x){
		long node = cmember[ctemp[w]+j];
		long k = blockIdx.y*blockDim.y+threadIdx.y;
		long size1 = crcw[node];
		long num_threads_y = blockDim.y * gridDim.y;
		for(;k<size1;k+=num_threads_y){
			cinitial[ctempg[node]+k]=0.85*crank[rcwgraph[ctempg[node]+k]]/outdeg[rcwgraph[ctempg[node]+k]];
		}
	}
}
__global__ void kernel1test(long *cn, long *cmem, long *cgraph, 
							long *ctemp, double *ccurr, double *crank, long *coutdeg, long *cparent)
{
	long w = blockIdx.x*blockDim.x + threadIdx.x;
	long num_threads_x = blockDim.x * gridDim.x;
	for(;w<(*cn);w+=num_threads_x){
		long size = ctemp[w+1]-ctemp[w];
		long j = blockIdx.y*blockDim.y + threadIdx.y;
		long num_threads_y = blockDim.y * gridDim.y;
		for(;j<size;j+=num_threads_y){
			long ind = ctemp[w]+j;
			long node = cgraph[ind];
			long par = cparent[node];
			ccurr[ind]=crank[par]/coutdeg[node];
		}
	}
}
__global__ void kernel1test1(long *cn, long *cmem, long *cgraph, 
							long *ctemp, double *ccurr, double *crank, long *coutdeg, long *cparent)
{
	long w = *cn;
	long size = ctemp[w+1]-ctemp[w];
	long j = blockIdx.x*blockDim.x + threadIdx.x;
	long num_threads_x = blockDim.x * gridDim.x;
	for(;j<size;j+=num_threads_x){
		long node = cgraph[ctemp[w]+j];
		ccurr[ctemp[w]+j]=crank[cparent[node]]/coutdeg[node];
	}
}

__global__ void kernel2test(long *cn, long *cmem, long *cgraph, 
							long *ctemp, double *ccurr, double *crank, long *coutdeg, long *cparent, long *cmarked)
{
 
	long w = blockIdx.x*blockDim.x + threadIdx.x;
	long num_threads_x = blockDim.x * gridDim.x;
	for(;w<(*cn);w+=num_threads_x){
		if(cmarked[w] != 0) continue; 
		long size = ctemp[w+1]-ctemp[w];
		long j = blockIdx.y*blockDim.y + threadIdx.y;
		long num_threads_y = blockDim.y * gridDim.y;	
		for(;j<size;j+=num_threads_y){
			long node = cgraph[ctemp[w]+j];
			ccurr[ctemp[w]+j]=crank[cparent[node]]/coutdeg[node];
		}
	}
}

__global__ void kernel2test1(long *cn, long *cmem, long *cgraph, 
							long *ctemp, double *ccurr, double *crank, long *coutdeg, long *cparent, long *cmarked)
{
 
	long w = *cn;
	if(cmarked[w] == 0){
		long size = ctemp[w+1]-ctemp[w];
		long j = blockIdx.y*blockDim.y + threadIdx.y;
		long num_threads_y = blockDim.y * gridDim.y;	
		for(;j<size;j+=num_threads_y){
			long node = cgraph[ctemp[w]+j];
			ccurr[ctemp[w]+j]=crank[cparent[node]]/coutdeg[node];
		}
	}
}
 
__global__ void kernel3test(long *cn, long *cmem, long *cgraph, 
							long *ctemp, double *ccurr, double *crank, long *coutdeg)
{
	long w = blockIdx.x*blockDim.x + threadIdx.x;
	long num_threads_x = blockDim.x * gridDim.x;
	for(;w<(*cn);w+=num_threads_x){
		long size = ctemp[w+1]-ctemp[w];
		long j = blockIdx.y*blockDim.y + threadIdx.y;
		long num_threads_y = blockDim.y * gridDim.y;
		for(;j<size;j+=num_threads_y){
			long node = cgraph[ctemp[w]+j];
			ccurr[ctemp[w]+j]=crank[node]/coutdeg[node];
		}
	}
}


__global__ void kernel3test1(long *cn, long *cmem, long *cgraph, 
							long *ctemp, double *ccurr, double *crank, long *coutdeg)
{
	long w = *cn;
	long size = ctemp[w+1]-ctemp[w];
	long j = blockIdx.y*blockDim.y + threadIdx.y;
	long num_threads_y = blockDim.y * gridDim.y;
	for(;j<size;j+=num_threads_y){
		long node = cgraph[ctemp[w]+j];
		ccurr[ctemp[w]+j]=crank[node]/coutdeg[node];
	}
}
 
__global__ void kernel4test(long *cn, long *cmem, long *cgraph, 
							long *ctemp, double *ccurr, double *crank, long *coutdeg, long *cmarked)
{
	long w = blockIdx.x*blockDim.x + threadIdx.x;
	long num_threads_x = blockDim.x * gridDim.x;
	for(;w<(*cn);w+=num_threads_x){
		if(cmarked[w] != 0) continue;
		long size = ctemp[w+1]-ctemp[w];
		long j = blockIdx.y*blockDim.y + threadIdx.y;	
		long num_threads_y = blockDim.y * gridDim.y;
		for(;j<size;j+=num_threads_y){
			long node = cgraph[ctemp[w]+j];
			ccurr[ctemp[w]+j]=crank[node]/coutdeg[node];
		}
	}
}

__global__ void kernel4test1(long *cn, long *cmem, long *cgraph, 
							long *ctemp, double *ccurr, double *crank, long *coutdeg, long *cmarked)
{
	long w = *cn;
	if(cmarked[w]==0){
		long size = ctemp[w+1]-ctemp[w];
		long j = blockIdx.y*blockDim.y + threadIdx.y;	
		long num_threads_y = blockDim.y * gridDim.y;
		for(;j<size;j+=num_threads_y){
			long node = cgraph[ctemp[w]+j];
			ccurr[ctemp[w]+j]=crank[node]/coutdeg[node];
		}
	}
}