#include "kernels.cuh"
using namespace std;
#define MAX 10000
double computeparalleli(vector<vector<long>> &graph, long *parent, vector<long> left, long n, long *outdeg, vector<long> &mapit, double *rank,double *initial, long nn)
{
	double total = 0.0;
	long i, iterations = 0;
	double damp=0.85, thres=1e-10, error = 0;
	double randomp=(1-damp)/graph.size();
	long thresh=MAX;
	long pivot=0;
	for(i=0;i<n;i++)
	{
		long node=mapit[i];
		if(graph[node].size()<thresh)
		{
			long temp=mapit[pivot];
			mapit[pivot]=node;
			mapit[i]=temp;
			pivot++;
		}   
	}
	double *curr = (double *)malloc(n*sizeof(double));
	for(long i=0;i<n;i++){
		curr[i]=0;
	}
	long *mem = (long *)malloc(n*sizeof(long));
	for(i=0;i<n;i++){
		mem[i]=mapit[i];
	}
	long *temp = (long *)malloc((n+1)*sizeof(long));
	long szz=0;
	for(i=0;i<n;i++){
		if(i){
			temp[i]=temp[i-1]+graph[mapit[i-1]].size();
		}else{
			temp[i]=0;
		}
		szz+=graph[mapit[i]].size();
	}
	temp[n]=temp[n-1]+graph[mapit[n-1]].size();
	long *graphh = (long *)malloc(szz*sizeof(long));
	long k=0;
	for(i=0;i<n;i++){
		for(auto c:graph[mapit[i]]){
			graphh[k++]=c;
		}
	}
	long *cn, *cm, *cmem, *coutdeg, *cparent, *ctemp, *cgraph;
	double *ccurr, *crank;
	double *currRank = (double *)malloc(szz * sizeof(double));
	double *ccurrRank;
	cudaMalloc((void**)&ccurrRank, szz * sizeof(double));
	cudaMalloc((void**)&cn, sizeof(long));
	cudaMalloc((void**)&cm, sizeof(long));
	cudaMalloc((void**)&cmem, n*sizeof(long));
	cudaMalloc((void**)&ccurr, n*sizeof(double));
	cudaMalloc((void**)&crank, nn*sizeof(double));
	cudaMalloc((void**)&coutdeg, nn*sizeof(long));
	cudaMalloc((void**)&cparent, nn*sizeof(long));
	cudaMalloc((void**)&ctemp, (n+1)*sizeof(long));
	cudaMalloc((void**)&cgraph, szz*sizeof(long));

	cudaMemcpy(cn, &pivot, sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cmem, mem, n*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(coutdeg, outdeg, nn*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cparent, parent, nn*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(ctemp, temp, (n+1)*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cgraph, graphh, szz*sizeof(long), cudaMemcpyHostToDevice);

	do  
	{
		error=0;
		for(i=0;i<n;i++){
			curr[i]=0;
		}

		for(i=0;i<szz;i++){
			currRank[i]=0;
		}
		cudaMemcpy(ccurrRank, currRank, szz*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(crank, rank, nn*sizeof(double), cudaMemcpyHostToDevice);

		long bx = pivot/1024+1;
		dim3 threadB(1024, 1);
		dim3 blockB(bx, 1024/bx);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);
		kernel1test<<<blockB ,threadB>>>(cn, cmem, cgraph, ctemp, ccurrRank, crank, coutdeg, cparent);
		
		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		total += elapsedTime;
		for(i=pivot;i<n;i++)
		{   
			{   
				cudaMemcpy(cm, &i, sizeof(long), cudaMemcpyHostToDevice);
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start, 0);
				kernel1test1<<<1,1024>>>(cm, cmem, cgraph, ctemp, ccurrRank, crank, coutdeg, cparent);
				cudaDeviceSynchronize();
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
			}   
		}   
		cudaMemcpy(currRank, ccurrRank, szz*sizeof(double) , cudaMemcpyDeviceToHost);
		for(i=0;i<n;i++){
			for(long kk=0;kk<graph[mapit[i]].size();kk++){
				curr[i]+=currRank[temp[i]+kk];
			}
		}
		double anse=0;
		for(i=0;i<n;i++){
			anse=max(anse, fabs(randomp+initial[mapit[i]]+damp*curr[i]-rank[mapit[i]]));
		}
		for(i=0;i<n;i++)
		{   
			{
				rank[mapit[i]]=damp*curr[i]+randomp+initial[mapit[i]];
			}   
		}
		iterations++;
		error = anse;
	}while(error > thres);
	for(i=0;i<left.size();i++)
		rank[left[i]]=rank[parent[left[i]]];
	cudaFree(cn);
	cudaFree(cm);
	cudaFree(cmem);
	cudaFree(coutdeg);
	cudaFree(cparent);
	cudaFree(ctemp);
	cudaFree(cgraph);
	cudaFree(ccurr);
	cudaFree(crank);
	return total;
}

void computeranki(vector < vector < long > > & graph, long *parent,vector < long > left,long n,long *outdeg,vector < long > &  mapit,double *rank,double *initial, long nn)
{
	double damp=0.85;
	double thres=1e-10;
	long i,j;
	vector < double > curr(n);
	double error=0;
	double randomp=(1-damp)/graph.size();
	do
	{
		error=0;
		for(i=0;i<n;i++)
		{
			{
				long node=mapit[i];
				double ans=0;
				for(j=0;j<graph[node].size();j++){
					ans=ans+rank[parent[graph[node][j]]]/outdeg[graph[node][j]];
				}
				curr[i]=randomp+damp*ans+initial[mapit[i]];
				error=max(error,fabs(curr[i]-rank[node]));
			}
		}
		for(i=0;i<n;i++)
		{
			{
				rank[mapit[i]]=curr[i];
			}
		}
	}while(error > thres);
	for(i=0;i<left.size();i++)
		rank[left[i]]=rank[parent[left[i]]];
	curr.clear();
}

double computeparallelid(vector < vector < long > > & graph,long *parent,vector < long > & left,long n,long *outdeg,vector < long > &  mapit,double *rank,double *initial, long nn)
{
	double total = 0.0;
	double thres=1e-10;
	double dis=1e-12;
	double value=((dis)*10.0)/n;
	long i;
	double *prev = (double *)malloc(n*sizeof(double));
	double *curr = (double *)malloc(n*sizeof(double));
	for(i=0;i<n;i++)
		prev[i]=rank[mapit[i]];
	long *marked = (long *)malloc(n*sizeof(long));
	memset(marked,0,n*sizeof(long));
	double error=0;
	long iterations=0;
	double damp = 0.85;
	double randomp=(1-damp)/graph.size();

	long *mem = (long *)malloc(n*sizeof(long));
	for(i=0;i<n;i++){
		mem[i]=mapit[i];
	}
	long *temp = (long *)malloc((n+1)*sizeof(long));
	long szz=0;
	for(i=0;i<n;i++){
		if(i){
			temp[i]=temp[i-1]+graph[mapit[i-1]].size();
		}else{
			temp[i]=0;
		}
		szz+=graph[mapit[i]].size();
	}
	temp[n]=temp[n-1]+graph[mapit[n-1]].size();
	long *graphh = (long *)malloc(szz*sizeof(long));
	long k=0;
	for(i=0;i<n;i++){
		for(auto c:graph[mapit[i]]){
			graphh[k++]=c;
		}
	}

	long pivot=0;
	long thresh=MAX;
	for(i=0;i<n;i++)
	{   
		long node=mapit[i];
		if(graph[node].size()<thresh)
		{   
			long temp=mapit[pivot];
			mapit[pivot]=node;
			mapit[i]=temp;
			pivot++;
		}   
	}

	long *cn, *cm, *cmem, *coutdeg, *cparent, *ctemp, *cgraph, *cmarked;;
	double *crank;

	double *currRank = (double *)malloc(szz * sizeof(double));
	double *ccurrRank;

	cudaMalloc((void**)&cn, sizeof(long));
	cudaMalloc((void**)&cm, sizeof(long));
	cudaMalloc((void**)&cmem, n*sizeof(long));
	cudaMalloc((void**)&coutdeg, nn*sizeof(long));
	cudaMalloc((void**)&cparent, nn*sizeof(long));
	cudaMalloc((void**)&ctemp, n*sizeof(long));
	cudaMalloc((void**)&cgraph, szz*sizeof(long));
	cudaMalloc((void**)&cmarked, n*sizeof(long));
	cudaMalloc((void**)&crank, nn*sizeof(double));

	cudaMemcpy(cn, &pivot, sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cmem, mem, n*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(coutdeg, outdeg, nn*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cparent, parent, nn*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(ctemp, temp, n*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cgraph, graphh, szz*sizeof(long), cudaMemcpyHostToDevice);

	do  
	{
		error=0;
		for(i=0;i<n;i++){
			curr[i]=0;
		}
		for(i=0;i<szz;i++){
			currRank[i]=0;
		}
		cudaMemcpy(ccurrRank, currRank, szz*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(crank, rank, nn*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(cmarked, marked, n*sizeof(long), cudaMemcpyHostToDevice);

		long bx = pivot/1024+1;
		dim3 threadB(1024, 1);
		dim3 blockB(bx, 1024/bx);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);

		kernel2test<<<blockB,threadB>>>(cn, cmem, cgraph, ctemp, ccurrRank, crank, coutdeg, cparent, cmarked);
		
		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		total += elapsedTime;

		for(i=pivot;i<n;i++)
		{   
			{   
				cudaMemcpy(cm, &i, sizeof(long), cudaMemcpyHostToDevice);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);

				kernel2test1<<<1,1024>>>(cm, cmem, cgraph, ctemp, ccurrRank, crank, coutdeg, cparent, cmarked);
				
				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
			}   
		}  
		cudaMemcpy(currRank, ccurrRank, szz*sizeof(double) , cudaMemcpyDeviceToHost);
		for(i=0;i<n;i++){
			for(long kk=0;kk<graph[mapit[i]].size();kk++){
				curr[i]+=currRank[temp[i]+kk];
			}
		}

		double anse=0;
		for(i=0;i<n;i++){
			if(!marked[i]){
				anse=max(anse, fabs(randomp+initial[mapit[i]]+damp*curr[i]-rank[mapit[i]]));
			}
		}
		iterations++;
		for(i=0;i<n;i++){
			if(!marked[i])   {
				rank[mapit[i]]=damp*curr[i]+randomp+initial[mapit[i]];
			}   
		}
		if(iterations%20==0){   
			for(i=0;i<n;i++){   
				if(!marked[i]){
					if(fabs(prev[i]-curr[i]) < value)
						marked[i]=1;
					else
						prev[i]=curr[i];
				}   
			}   
		}   
		error = anse;
	}while(error > thres );
	for(i=0;i<left.size();i++)
		rank[left[i]]=rank[parent[left[i]]];
	cudaFree(cn);
	cudaFree(cm);
	cudaFree(cmem);
	cudaFree(coutdeg);
	cudaFree(cparent);
	cudaFree(ctemp);
	cudaFree(cgraph);
	cudaFree(crank);
	cudaFree(cmarked);
	cudaFree(ccurrRank);
	return total;
}

void computerankid(vector < vector < long > > & graph,long *parent,vector < long > & left, long n,long *outdeg,vector < long > &  mapit,double *rank,double *initial, long nn)
{
	double damp = 0.85;
	double thres=1e-10;
	long i,j;
	vector < double > curr(n);
	vector < double > prev(n);
	for(i=0;i<n;i++)
		prev[i]=rank[mapit[i]];
	double dis=1e-12;
	double value=(dis*10.0)/n;
	bool *marked = (bool *)malloc(n*sizeof(bool));
	memset(marked,0,n*sizeof(bool));
	double error=0;
	long  iterations=0;
	double randomp=(1-damp)/graph.size();
	do
	{
		error=0;
		for(i=0;i<n;i++)
		{
			if(!marked[i])
			{
				long node=mapit[i];
				double ans=0;
				for(j=0;j<graph[node].size();j++)
				{
					ans=ans+rank[parent[graph[node][j]]]/outdeg[graph[node][j]];
				}
				curr[i]=randomp+damp*ans+initial[mapit[i]];
				error=max(error,fabs(curr[i]-rank[node]));
			}
		}
		for(i=0;i<n;i++)
			if(!marked[i]) rank[mapit[i]]=curr[i];
		iterations++;
		if(iterations%20==0)
		{
			for(i=0;i<n;i++)
			{
				if(!marked[i])
				{
					if(fabs(prev[i]-curr[i]) < value )marked[i]=1;
					else prev[i]=curr[i];
				}
			}
		}
	}while(error > thres );
	for(i=0;i<left.size();i++)
		rank[left[i]]=rank[parent[left[i]]];
}

double computeparallel(vector < vector < long > > & graph,long n,long *outdeg,vector < long > &  mapit,double *rank,double *initial, long nn)
{
	double total = 0.0;
	double damp=0.85;
	double thres=1e-10;
	long i;
	double *curr = (double *)malloc(n*sizeof(double));
	double error=0;
	long iterations=0;
	double randomp=(1-damp)/graph.size();

	long *mem = (long *)malloc(n*sizeof(long));
	for(i=0;i<n;i++){
		mem[i]=mapit[i];
	}

	long *temp = (long *)malloc((n+1)*sizeof(long));
	long szz=0;
	for(i=0;i<n;i++){
		if(i){
			temp[i]=temp[i-1]+graph[mapit[i-1]].size();
		}else{
			temp[i]=0;
		}
		szz+=graph[mapit[i]].size();
	}
	temp[n]=temp[n-1]+graph[mapit[n-1]].size();

	long *graphh = (long *)malloc(szz*sizeof(long));
	long k=0;
	for(i=0;i<n;i++){
		for(auto c:graph[mapit[i]]){
			graphh[k++]=c;
		}
	}

	long pivot=0;
	long thresh=MAX;
	for(i=0;i<n;i++)
	{   
		long node=mapit[i];
		if(graph[node].size()<thresh)
		{   
			long temp=mapit[pivot];
			mapit[pivot]=node;
			mapit[i]=temp;
			pivot++;
		}   
	}

	long *cn, *cm, *cmem, *coutdeg, *ctemp, *cgraph;
	double *crank;

	double *currRank = (double *)malloc(szz*sizeof(double));
	double *ccurrRank;
	cudaMalloc((void**)&ccurrRank, szz*sizeof(double));

	cudaMalloc((void**)&cn, sizeof(long));
	cudaMalloc((void**)&cm, sizeof(long));
	cudaMalloc((void**)&cmem, n*sizeof(long));
	cudaMalloc((void**)&coutdeg, nn*sizeof(long));
	cudaMalloc((void**)&ctemp, n*sizeof(long));
	cudaMalloc((void**)&cgraph, szz*sizeof(long));
	cudaMalloc((void**)&crank, nn*sizeof(double));

	cudaMemcpy(cn, &pivot, sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cmem, mem, n*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(coutdeg, outdeg, nn*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(ctemp, temp, n*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cgraph, graphh, szz*sizeof(long), cudaMemcpyHostToDevice);

	do
	{
		error=0;
		
		for(i=0;i<n;i++){
			curr[i]=0;
		}
		for(i=0;i<szz;i++){
			currRank[i]=0;
		}

		cudaMemcpy(crank, rank, nn*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(ccurrRank, currRank, szz*sizeof(double), cudaMemcpyHostToDevice);

		long bx = pivot/1024+1;
		dim3 threadB(1024, 1);
		dim3 blockB(bx, 1024/bx);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);

		kernel3test<<<blockB,threadB>>>(cn, cmem, cgraph, ctemp, ccurrRank, crank, coutdeg);

		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		total += elapsedTime;
		cout << elapsedTime << "\n";

		for(i=pivot;i<n;i++)
		{   
			{   
				cudaMemcpy(cm, &i, sizeof(long), cudaMemcpyHostToDevice);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);

				kernel3test1<<<1,1024>>>(cm, cmem, cgraph, ctemp, ccurrRank, crank, coutdeg);
				
				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
			}   
		}  
		cudaMemcpy(currRank, ccurrRank, szz*sizeof(double) , cudaMemcpyDeviceToHost);
		for(i=0;i<n;i++){
			for(long kk=0;kk<graph[mapit[i]].size();kk++){
				curr[i]+=currRank[temp[i]+kk];
			}
		}
		double anse=0;
		for(i=0;i<n;i++){
			anse=max(anse, fabs(randomp+initial[mapit[i]]+damp*curr[i]-rank[mapit[i]]));
		}
		for(i=0;i<n;i++){
			rank[mapit[i]]=damp*curr[i]+randomp+initial[mapit[i]];
		}
		iterations++;
		error = anse;
	}while(error > thres );
	cudaFree(cn);
	cudaFree(cm);
	cudaFree(cmem);
	cudaFree(coutdeg);
	cudaFree(ctemp);
	cudaFree(cgraph);
	cudaFree(crank);
	cudaFree(ccurrRank);
	cout << "It: " << iterations << "\n";
	return total;
}


void computerank(vector < vector < long > > & graph,long n,long *outdeg,vector < long > &  mapit,double *rank,double *initial)
{
	double damp=0.85;
	double thres=1e-10;
	long i,j;
	vector < double > curr(n);
	double error=0;
	double randomp=(1-damp)/graph.size();
	do
	{
		error=0;
		for(i=0;i<n;i++)
		{
			{
				long node=mapit[i];
				double ans=0;
				for(j=0;j<graph[node].size();j++)
					ans=ans+rank[graph[node][j]]/outdeg[graph[node][j]];
				curr[i]=randomp+damp*ans+initial[mapit[i]];
				error=max(error,fabs(curr[i]-rank[node]));
			}
		}
		for(i=0;i<n;i++)
		{
			{
				rank[mapit[i]]=curr[i];
			}
		}
	}while(error > thres );
	curr.clear();
}

double computeparalleld(vector < vector < long > > & graph,long n,long *outdeg,vector < long > &  mapit,double *rank,double *initial, long nn)
{
	double total = 0.0;
	double thres=1e-10;
	double dis=1e-12;
	double value=((dis)*10.0)/n;
	long i;
	double *curr = (double *)malloc(n*sizeof(double));
	double *prev = (double *)malloc(n*sizeof(double));
	for(i=0;i<n;i++)
		prev[i]=rank[mapit[i]];
	long *marked = (long *)malloc(n*sizeof(long));
	memset(marked,0,n*sizeof(long));
	double error=0;
	long iterations=0;
	double damp = 0.85;
	double randomp=(1-damp)/graph.size();

	long *mem = (long *)malloc(n*sizeof(long));
	for(i=0;i<n;i++){
		mem[i]=mapit[i];
	}
	long *temp = (long *)malloc((n+1)*sizeof(long));
	long szz=0;
	for(i=0;i<n;i++){
		if(i){
			temp[i]=temp[i-1]+graph[mapit[i-1]].size();
		}else{
			temp[i]=0;
		}
		szz+=graph[mapit[i]].size();
	}
	temp[n]=temp[n-1]+graph[mapit[n-1]].size();
	long *graphh = (long *)malloc(szz*sizeof(long));
	long k=0;
	for(i=0;i<n;i++){
		for(auto c:graph[mapit[i]]){
			graphh[k++]=c;
		}
	}

	long thresh=MAX;
	long pivot=0;
	for(i=0;i<n;i++)
	{   
		long node=mapit[i];
		if(graph[node].size()<thresh)
		{   
			long temp=mapit[pivot];
			mapit[pivot]=node;
			mapit[i]=temp;
			pivot++;
		}   
	}

	long *cn, *cm, *cmem, *coutdeg, *ctemp, *cgraph, *cmarked;
	double *crank;

	double *currRank = (double *)malloc(szz*sizeof(double));
	double *ccurrRank;
	cudaMalloc((void**)&ccurrRank, szz*sizeof(double));

	cudaMalloc((void**)&cn, sizeof(long));
	cudaMalloc((void**)&cm, sizeof(long));
	cudaMalloc((void**)&cmem, n*sizeof(long));
	cudaMalloc((void**)&coutdeg, nn*sizeof(long));
	cudaMalloc((void**)&ctemp, n*sizeof(long));
	cudaMalloc((void**)&cgraph, szz*sizeof(long));
	cudaMalloc((void**)&cmarked, n*sizeof(long));
	cudaMalloc((void**)&crank, nn*sizeof(double));

	cudaMemcpy(cn, &pivot, sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cmem, mem, n*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(coutdeg, outdeg, nn*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(ctemp, temp, n*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cgraph, graphh, szz*sizeof(long), cudaMemcpyHostToDevice);

	do
	{
		error=0;
		for(i=0;i<n;i++){
			curr[i]=0;
		}
		for(i=0;i<szz;i++){
			currRank[i]=0;
		}
		
		cudaMemcpy(crank, rank, nn*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(cmarked, marked, n*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(ccurrRank, currRank, szz*sizeof(double), cudaMemcpyHostToDevice);

		long bx = pivot/1024+1;
		dim3 threadB(1024, 1);
		dim3 blockB(bx, 1024/bx);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);

		kernel4test<<<blockB,threadB>>>(cn, cmem, cgraph, ctemp, ccurrRank, crank, coutdeg, cmarked);
		
		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		total += elapsedTime;
		

		for(i=pivot;i<n;i++)
		{   
			{   
				cudaMemcpy(cm, &i, sizeof(long), cudaMemcpyHostToDevice);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);

				kernel4test1<<<1,1024>>>(cm, cmem, cgraph, ctemp, ccurrRank, crank, coutdeg, cmarked);
				
				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
			}   
		}  
		cudaMemcpy(currRank, ccurrRank, szz*sizeof(double) , cudaMemcpyDeviceToHost);
		for(i=0;i<n;i++){
			for(long kk=0;kk<graph[mapit[i]].size();kk++){
				curr[i]+=currRank[temp[i]+kk];
			}
		}

		double anse=0;
		for(i=0;i<n;i++)
			if(!marked[i])
				anse=max(anse, fabs(randomp+initial[mapit[i]]+damp*curr[i]-rank[mapit[i]]));
		iterations++;
		for(i=0;i<n;i++)
		{
			if(!marked[i])   
			{
				rank[mapit[i]]=damp*curr[i]+randomp+initial[mapit[i]];
			}   
		}
		if(iterations%20==0)
		{   
			for(i=0;i<n;i++)
			{   
				if(!marked[i])
				{   

					if(fabs(prev[i]-curr[i]) < value )marked[i]=1;
					else
						prev[i]=curr[i];
				}   
			}   
		}   
		error = anse;
	}while(error > thres );
	cudaFree(cn);
	cudaFree(cm);
	cudaFree(cmem);
	cudaFree(coutdeg);
	cudaFree(ctemp);
	cudaFree(cgraph);
	cudaFree(cmarked);
	cudaFree(crank);
	cudaFree(ccurrRank);
	return total;
}

void computerankd(vector < vector < long > > & graph,long n,long *outdeg,vector < long > &  mapit,double *rank,double *initial)
{
	double damp = 0.85;
	double thres=1e-10;
	long i,j;
	vector < double > curr(n);
	vector < double > prev(n);
	for(i=0;i<n;i++)
		prev[i]=rank[mapit[i]];
	double dis=1e-12;
	double value=(dis*10.0)/n;
	// double bound=1e-5;
	bool *marked = (bool *)malloc(n*sizeof(bool));
	memset(marked,0,n*sizeof(bool));
	double error=0;
	long  iterations=0;
	double randomp=(1-damp)/graph.size();
	do
	{
		error=0;
		for(i=0;i<n;i++)
		{
			if(!marked[i])
			{
				long node=mapit[i];
				double ans=0;
				for(j=0;j<graph[node].size();j++)
					ans=ans+rank[graph[node][j]]/outdeg[graph[node][j]];
				curr[i]=randomp+damp*ans+initial[mapit[i]];
				error=max(error,fabs(curr[i]-rank[node]));
			}
		}
		for(i=0;i<n;i++)
			if(!marked[i]) rank[mapit[i]]=curr[i];
		iterations++;
		if(iterations%20==0)
		{
			for(i=0;i<n;i++)
			{
				if(!marked[i])
				{
					if(fabs(prev[i]-curr[i]) < value )marked[i]=1;
					else prev[i]=curr[i];
				}
			}
		}
	}while(error > thres );
	curr.clear();
}

double computeparallelc(vector < vector < long > > & graph,long n,long *outdeg,vector < long > &  mapit,double *rank,double *initial,long *level,long *redir,double *powers, long nn)
{
	double total = 0.0;
	double damp=0.85;
	double thres=1e-10;
	long i;
	double *curr = (double *)malloc(n*sizeof(double));
	double error=0;
	long iterations=0;
	double randomp=(1-damp)/graph.size();
	long limit=0;
	for(i=0;i<n;i++)
	{   
		long node=mapit[i];
		if(node==redir[node])
		{   
			long temp=mapit[limit];
			mapit[limit]=node;
			mapit[i]=temp;
			limit++;
		}   
	} 

	long *mem = (long *)malloc(n*sizeof(long));
	for(i=0;i<n;i++){
		mem[i]=mapit[i];
	}
	long *temp = (long *)malloc((n+1)*sizeof(long));
	long szz=0;
	for(i=0;i<n;i++){
		if(i){
			temp[i]=temp[i-1]+graph[mapit[i-1]].size();
		}else{
			temp[i]=0;
		}
		szz+=graph[mapit[i]].size();
	}
	temp[n]=temp[n-1]+graph[mapit[n-1]].size();
	long *graphh = (long *)malloc(szz*sizeof(long));
	long k=0;
	for(i=0;i<n;i++){
		for(auto c:graph[mapit[i]]){
			graphh[k++]=c;
		}
	}

	long pivot=0;
	long thresh = 10000;
	for(i=0;i<limit;i++)
	{   
		long node=mapit[i];
		if(graph[node].size()<thresh)
		{   
			long temp=mapit[pivot];
			mapit[pivot]=node;
			mapit[i]=temp;
			pivot++;
		}   
	}   

	long *cn, *cm, *cmem, *coutdeg, *ctemp, *cgraph;
	double *crank;

	double *currRank = (double *)malloc(sizeof(double)*szz);
	double *ccurrRank;

	cudaMalloc((void**)&ccurrRank, szz*sizeof(double));

	cudaMalloc((void**)&cn, sizeof(long));
	cudaMalloc((void**)&cm, sizeof(long));
	cudaMalloc((void**)&cmem, n*sizeof(long));
	cudaMalloc((void**)&coutdeg, nn*sizeof(long));
	cudaMalloc((void**)&ctemp, n*sizeof(long));
	cudaMalloc((void**)&cgraph, szz*sizeof(long));
	cudaMalloc((void**)&crank, nn*sizeof(double));

	cudaMemcpy(cn, &pivot, sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cmem, mem, n*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(coutdeg, outdeg, nn*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(ctemp, temp, n*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cgraph, graphh, szz*sizeof(long), cudaMemcpyHostToDevice);

	do  
	{   
		error=0;
		for(i=0;i<n;i++){
			curr[i]=0;
		}
		for(i=0;i<szz;i++){
			currRank[i]=0;
		}
		
		cudaMemcpy(crank, rank, nn*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(ccurrRank, currRank, szz*sizeof(double), cudaMemcpyHostToDevice);
		

		long bx = pivot/1024+1;
		dim3 threadB(1024, 1);
		dim3 blockB(bx, 1024/bx);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);

		kernel3test<<<blockB,threadB>>>(cn, cmem, cgraph, ctemp, ccurrRank, crank, coutdeg);
		
		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		total += elapsedTime;

		for(i=pivot;i<limit;i++)
		{   
			{   
				cudaMemcpy(cm, &i, sizeof(long), cudaMemcpyHostToDevice);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);

				kernel3test1<<<1,1024>>>(cm, cmem, cgraph, ctemp, ccurrRank, crank, coutdeg);

				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
			}   
		}  
		cudaMemcpy(currRank, ccurrRank, szz*sizeof(double) , cudaMemcpyDeviceToHost);
		for(i=0;i<n;i++){
			for(long kk=0;kk<graph[mapit[i]].size();kk++){
				curr[i]+=currRank[temp[i]+kk];
			}
		}

		double anse=0;
		for(i=0;i<limit;i++){
			anse=max(anse, fabs(randomp+initial[mapit[i]]+damp*curr[i]-rank[mapit[i]]));
		}

		for(i=0;i<limit;i++){
				rank[mapit[i]]=damp*curr[i]+randomp+initial[mapit[i]];
		}
		iterations++;
		error = anse; 
	}while(error > thres);
	for(i=limit;i<n;i++)
	{   
		long node=mapit[i];
		double val=powers[level[node]];
		rank[node]=rank[redir[node]]*val+(1.0-val)/graph.size();
	}   
	cudaFree(cn);
	cudaFree(cm);
	cudaFree(cmem);
	cudaFree(coutdeg);
	cudaFree(ctemp);
	cudaFree(cgraph);
	cudaFree(crank);
	cudaFree(ccurrRank);
	return total;
}

void computerankc(vector < vector < long > > & graph,long n,long *outdeg,vector < long > &  mapit,double *rank,double *initial,long *level,long *redir,double *powers)
{
	double damp=0.85;
	double thres=1e-10;
	long i,j;
	vector < double > curr(n);
	double error=0;
	long  iterations=0;
	double randomp=(1-damp)/graph.size();
	long limit=0;
	for(i=0;i<n;i++)
	{
		long node=mapit[i];
		if(node==redir[node])
		{
			long temp=mapit[limit];
			mapit[limit]=node;
			mapit[i]=temp;
			limit++;
		}
	}
	do
	{
		error=0;
		for(i=0;i<limit;i++)
		{
			long node=mapit[i];
			double ans=0;
			for(j=0;j<graph[node].size();j++)
			{
				ans=ans+rank[graph[node][j]]/outdeg[graph[node][j]];
			}
			curr[i]=randomp+damp*ans+initial[mapit[i]];
			error=max(error,fabs(curr[i]-rank[node]));
		}
		for(i=0;i<limit;i++)
			rank[mapit[i]]=curr[i];
		iterations++;
	}while(error > thres );
	for(i=limit;i<n;i++)
	{
		long node=mapit[i];
		double val=powers[level[node]];
		rank[node]=rank[redir[node]]*val+(1.0-val)/graph.size();
	}
}

double computeparalleldc(vector < vector < long > > & graph,long n,long *outdeg,vector < long > &  mapit,double *rank,double *initial,long *level,long *redir,double *powers, long nn)
{
	double total = 0.0;
	double damp=0.85;
	double thres=1e-10;
	double value=((1e-12)*10.0)/double(n);
	long i;
	double *curr = (double *)malloc(n*sizeof(double));
	double *prev = (double *)malloc(n*sizeof(double));
	for(i=0;i<n;i++){
		prev[i]=1.0/n;
	}
	long *marked = (long *)malloc(n*sizeof(long));
	memset(marked,0,n*sizeof(long));
	double error=0;
	long iterations=0;
	double randomp=(1-damp)/graph.size();
	long limit=0;
	for(i=0;i<n;i++)
	{   
		long node=mapit[i];
		if(node==redir[node])
		{   
			long temp=mapit[limit];
			mapit[limit]=node;
			mapit[i]=temp;
			limit++;
		}   
	}

	long *mem = (long *)malloc(n*sizeof(long));
	for(i=0;i<n;i++){
		mem[i]=mapit[i];
	}
	long *temp = (long *)malloc((n+1)*sizeof(long));
	long szz=0;
	for(i=0;i<n;i++){
		if(i){
			temp[i]=temp[i-1]+graph[mapit[i-1]].size();
		}else{
			temp[i]=0;
		}
		szz+=graph[mapit[i]].size();
	}
	temp[n]=temp[n-1]+graph[mapit[n-1]].size();
	long *graphh = (long *)malloc(szz*sizeof(long));
	long k=0;
	for(i=0;i<n;i++){
		for(auto c:graph[mapit[i]]){
			graphh[k++]=c;
		}
	}

	long pivot=0;
	long thresh=MAX;
	for(i=0;i<limit;i++)
	{   
		long node=mapit[i];
		if(graph[node].size()<thresh)
		{   
			long temp=mapit[pivot];
			mapit[pivot]=node;
			mapit[i]=temp;
			pivot++;
		}   
	}   

	long *cn, *cm, *cmem, *coutdeg, *ctemp, *cgraph, *cmarked;
	double *crank;

	double *currRank, *ccurrRank;
	currRank = (double *)malloc(szz*sizeof(double));
	cudaMalloc((void**)&ccurrRank, szz*sizeof(double));

	cudaMalloc((void**)&cn, sizeof(long));
	cudaMalloc((void**)&cm, sizeof(long));
	cudaMalloc((void**)&cmem, n*sizeof(long));
	cudaMalloc((void**)&coutdeg, nn*sizeof(long));
	cudaMalloc((void**)&ctemp, n*sizeof(long));
	cudaMalloc((void**)&cgraph, szz*sizeof(long));
	cudaMalloc((void**)&cmarked, n*sizeof(long));
	cudaMalloc((void**)&crank, nn*sizeof(double));

	cudaMemcpy(cn, &pivot, sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cmem, mem, n*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(coutdeg, outdeg, nn*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(ctemp, temp, n*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cgraph, graphh, szz*sizeof(long), cudaMemcpyHostToDevice);

	do  
	{   
		error=0;
		for(i=0;i<n;i++){
			curr[i]=0;
		}
		for(i=0;i<szz;i++){
			currRank[i]=0;
		}
		
		cudaMemcpy(crank, rank, nn*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(cmarked, marked, n*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(ccurrRank, currRank, szz*sizeof(double), cudaMemcpyHostToDevice);

		long bx = pivot/1024+1;
		dim3 threadB(1024, 1);
		dim3 blockB(bx, 1024/bx);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);

		kernel4test<<<blockB,threadB>>>(cn, cmem, cgraph, ctemp, ccurrRank, crank, coutdeg, cmarked);
		
		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		total += elapsedTime;

		for(i=pivot;i<limit;i++)
		{   
			{   
				cudaMemcpy(cm, &i, sizeof(long), cudaMemcpyHostToDevice);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);

				kernel4test1<<<1,1024>>>(cm, cmem, cgraph, ctemp, ccurrRank, crank, coutdeg, cmarked);
				
				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
			}   
		}

		cudaMemcpy(currRank, ccurrRank, szz*sizeof(double) , cudaMemcpyDeviceToHost);
		for(i=0;i<n;i++){
			for(long kk=0;kk<graph[mapit[i]].size();kk++){
				curr[i]+=currRank[temp[i]+kk];
			}
		}


		double anse=0;
		for(i=0;i<limit;i++)
			if(!marked[i])
				anse=max(anse, fabs(randomp+initial[mapit[i]]+damp*curr[i]-rank[mapit[i]]));
		iterations++;
		for(i=0;i<limit;i++){
			if(!marked[i]){
				rank[mapit[i]]=damp*curr[i]+randomp+initial[mapit[i]];
			}
		}
		if(iterations%20==0)
		{   
			for(i=0;i<limit;i++)
			{   
				if(!marked[i])
				{   
					if(fabs(prev[i]-curr[i]) < value )marked[i]=1;
					else
						prev[i]=curr[i];
				}   
			}   
		}
		error = anse;
	}while(error > thres);
	for(i=limit;i<n;i++)
	{   
		long node=mapit[i];
		double val=powers[level[node]];
		rank[node]=rank[redir[node]]*val+(1.0-val)/graph.size();
	}   
	cudaFree(cn);
	cudaFree(cm);
	cudaFree(cmem);
	cudaFree(coutdeg);
	cudaFree(ctemp);
	cudaFree(cgraph);
	cudaFree(cmarked);
	cudaFree(crank);
	cudaFree(ccurrRank);
	return total;
}

void computerankdc(vector < vector < long > > & graph,long n,long *outdeg,vector < long > &  mapit,double *rank,double *initial,long *level,long *redir,double *powers)
{
	double damp=0.85;
	double thres=1e-10;
	long i, j;
	vector < double > curr(n);
	vector < double > prev(n,1.0/n);
	double value=((1e-12)*10.0)/double ( n );
	bool *marked = (bool *)malloc(n*sizeof(bool));
	memset(marked,0,n*sizeof(bool));
	double error=0;
	long  iterations=0;
	double randomp=(1-damp)/graph.size();
	long limit=0;
	for(i=0;i<n;i++)
	{
		long node=mapit[i];
		if(node==redir[node])
		{
			long temp=mapit[limit];
			mapit[limit]=node;
			mapit[i]=temp;
			limit++;
		}
	}
	do
	{
		error=0;
		for(i=0;i<limit;i++)
		{
			long node=mapit[i];
			if(!marked[i])
			{
				double ans=0;
				for(j=0;j<graph[node].size();j++)
				{
					ans=ans+rank[graph[node][j]]/outdeg[graph[node][j]];
				}
				curr[i]=randomp+damp*ans+initial[mapit[i]];
				error=max(error,fabs(curr[i]-rank[node]));
			}
		}
		for(i=0;i<limit;i++) if(!marked[i])
			rank[mapit[i]]=curr[i];
		iterations++;
		if(iterations%20==0){
			for(i=0;i<limit;i++)
			{
				if(!marked[i])
				{
					if(fabs(prev[i]-curr[i]) < value )marked[i]=1;
					else prev[i]=curr[i];
				}
			}
		}
	}while(error > thres );
	for(i=limit;i<n;i++)
	{
		long node=mapit[i];
		double val=powers[level[node]];
		rank[node]=rank[redir[node]]*val+(1.0-val)/graph.size();
	}
}

double computeparallelic(vector < vector < long > > & graph,long *parent,vector <long > & left, long n,long *outdeg,vector < long > &  mapit,double *rank,double *initial,long *level,long *redir,double *powers, long nn)
{
	double total = 0.0;
	double damp=0.85;
	double thres=1e-10;
	long i;
	double *curr = (double *)malloc(n*sizeof(double));
	double error=0;
	long iterations=0;
	double randomp=(1-damp)/graph.size();
	long limit=0;
	for(i=0;i<n;i++)
	{   
		long node=mapit[i];
		if(node==redir[node])
		{   
			long temp=mapit[limit];
			mapit[limit]=node;
			mapit[i]=temp;
			limit++;
		}   
	}
	long *mem = (long *)malloc(n*sizeof(long));
	for(i=0;i<n;i++){
		mem[i]=mapit[i];
	}
	long *temp = (long *)malloc((n+1)*sizeof(long));
	long szz=0;
	for(i=0;i<n;i++)
	{
		if(i)
		{
			temp[i]=temp[i-1]+graph[mapit[i-1]].size();
		}
		else
		{
			temp[i]=0;
		}
		szz+=graph[mapit[i]].size();
	}
	temp[n]=temp[n-1]+graph[mapit[n-1]].size();
	long *graphh = (long *)malloc(szz*sizeof(long));
	long k=0;
	for(i=0;i<n;i++){
		for(auto c:graph[mapit[i]]){
			graphh[k++]=c;
		}
	}

	long pivot=0;
	long thresh=MAX;
	for(i=0;i<limit;i++)
	{   
		long node=mapit[i];
		if(graph[node].size()<thresh)
		{   
			long temp=mapit[pivot];
			mapit[pivot]=node;
			mapit[i]=temp;
			pivot++;
		}   
	}   

	long *cn, *cm, *cmem, *coutdeg, *cparent, *ctemp, *cgraph;
	double *crank;

	double *currRank, *ccurrRank;
	currRank = (double *)malloc(szz*sizeof(double));
	cudaMalloc((void**)&ccurrRank, szz*sizeof(double));

	cudaMalloc((void**)&cn, sizeof(long));
	cudaMalloc((void**)&cm, sizeof(long));
	cudaMalloc((void**)&cmem, n*sizeof(long));
	cudaMalloc((void**)&coutdeg, nn*sizeof(long));
	cudaMalloc((void**)&cparent, nn*sizeof(long));
	cudaMalloc((void**)&ctemp, n*sizeof(long));
	cudaMalloc((void**)&cgraph, szz*sizeof(long));
	cudaMalloc((void**)&crank, nn*sizeof(double));
	cudaMalloc((void**)&cparent, nn*sizeof(long));

	cudaMemcpy(cn, &pivot, sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cmem, mem, n*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(coutdeg, outdeg, nn*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cparent, parent, nn*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(ctemp, temp, n*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cgraph, graphh, szz*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cparent, parent, nn*sizeof(long), cudaMemcpyHostToDevice);
	do  
	{   
		error=0;
		for(i=0;i<n;i++){
			curr[i]=0;
		}
		for(i=0;i<szz;i++){
			currRank[i]=0;
		}

		cudaMemcpy(crank, rank, nn*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(ccurrRank, currRank, szz*sizeof(double), cudaMemcpyHostToDevice);

		long bx = pivot/1024+1;
		dim3 threadB(1024, 1);
		dim3 blockB(bx, 1024/bx);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);

		kernel1test<<<blockB,threadB>>>(cn, cmem, cgraph, ctemp, ccurrRank, crank, coutdeg, cparent);
		
		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		total += elapsedTime;

		for(i=pivot;i<limit;i++)
		{   
			{   
				cudaMemcpy(cm, &i, sizeof(long), cudaMemcpyHostToDevice);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);

				kernel1test1<<<1,1024>>>(cm, cmem, cgraph, ctemp, ccurrRank, crank, coutdeg, cparent);
				
				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
			}   
		} 

		cudaMemcpy(currRank, ccurrRank, szz*sizeof(double) , cudaMemcpyDeviceToHost);
		for(i=0;i<n;i++){
			for(long kk=0;kk<graph[mapit[i]].size();kk++){
				curr[i]+=currRank[temp[i]+kk];
			}
		}

		double anse=0;
		for(i=0;i<limit;i++){
			anse=max(anse, fabs(randomp+initial[mapit[i]]+damp*curr[i]-rank[mapit[i]]));
		}

		for(i=0;i<limit;i++)
		{   
			{
				rank[mapit[i]]=damp*curr[i]+randomp+initial[mapit[i]];
			}   
		}
		iterations++;
		error = anse;
	}while(error > thres );
	for(i=limit;i<n;i++)
	{   
		long node=mapit[i];
		double val=powers[level[node]];
		rank[node]=rank[redir[node]]*val+(1.0-val)/graph.size();
	}   
	for(i=0;i<left.size();i++)
		rank[left[i]]=rank[parent[left[i]]];
	cudaFree(cn);
	cudaFree(cm);
	cudaFree(cmem);
	cudaFree(coutdeg);
	cudaFree(ctemp);
	cudaFree(cgraph);
	cudaFree(cparent);
	cudaFree(crank);
	cudaFree(ccurrRank);
	return total;
}

void computerankic(vector < vector < long > > & graph,long *parent,vector < long > & left,long n,long *outdeg,vector < long > &  mapit,double *rank,double *initial,long *level,long *redir,double *powers)
{
	double damp=0.85;
	double thres=1e-10;
	long i, j;
	vector < double > curr(n);
	double error=0;
	long  iterations=0;
	double randomp=(1-damp)/graph.size();
	long limit=0;
	for(i=0;i<n;i++)
	{
		long node=mapit[i];
		if(node==redir[node])
		{
			long temp=mapit[limit];
			mapit[limit]=node;
			mapit[i]=temp;
			limit++;
		}
	}
	do
	{
		error=0;
		for(i=0;i<limit;i++)
		{
			long node=mapit[i];
			double ans=0;
			for(j=0;j<graph[node].size();j++)
			{
				ans=ans+rank[parent[graph[node][j]]]/outdeg[graph[node][j]];
			}
			curr[i]=randomp+damp*ans+initial[mapit[i]];
			error=max(error,fabs(curr[i]-rank[node]));
		}
		for(i=0;i<limit;i++)
			rank[mapit[i]]=curr[i];
		iterations++;
	}while(error > thres );
	for(i=limit;i<n;i++)
	{
		long node=mapit[i];
		double val=powers[level[node]];
		rank[node]=rank[redir[node]]*val+(1.0-val)/graph.size();
	}
	for(i=0;i<left.size();i++)
		rank[left[i]]=rank[parent[left[i]]];
}

double computeparallelidc(vector < vector < long > > & graph, long *parent,vector <long> & left,long n,long *outdeg,vector < long > &  mapit,double *rank,double *initial,long *level,long *redir,double *powers, long nn)
{
	double total = 0.0;
	double damp=0.85;
	double thres=1e-10;
	double value=((1e-12)*10.0)/double ( n );
	long i;
	double *curr = (double *)malloc(n*sizeof(double));
	double *prev = (double *)malloc(n*sizeof(double));
	for(i=0;i<n;i++){
		prev[i]=1.0/n;
	}
	long *marked = (long *)malloc(n*sizeof(long));
	memset(marked,0,n*sizeof(long));
	double error=0;
	long iterations=0;
	double randomp=(1-damp)/graph.size();
	long limit=0;
	for(i=0;i<n;i++)
	{   
		long node=mapit[i];
		if(node==redir[node])
		{   
			long temp=mapit[limit];
			mapit[limit]=node;
			mapit[i]=temp;
			limit++;
		}   
	}

	long *mem = (long *)malloc(n*sizeof(long));
	for(i=0;i<n;i++){
		mem[i]=mapit[i];
	}
	long *temp = (long *)malloc((n+1)*sizeof(long));
	long szz=0;
	for(i=0;i<n;i++){
		if(i){
			temp[i]=temp[i-1]+graph[mapit[i-1]].size();
		}else{
			temp[i]=0;
		}
		szz+=graph[mapit[i]].size();
	}
	temp[n]=temp[n-1]+graph[mapit[n-1]].size();
	long *graphh = (long *)malloc(szz*sizeof(long));
	long k=0;
	for(i=0;i<n;i++){
		for(auto c:graph[mapit[i]]){
			graphh[k++]=c;
		}
	}

	long pivot=0;
	long thresh=MAX;
	for(i=0;i<limit;i++)
	{   
		long node=mapit[i];
		if(graph[node].size()<thresh)
		{   
			long temp=mapit[pivot];
			mapit[pivot]=node;
			mapit[i]=temp;
			pivot++;
		}   
	}   

	long *cn, *cm, *cmem, *coutdeg, *cparent, *ctemp, *cgraph, *cmarked;
	double *crank;

	double *currRank, *ccurrRank;
	currRank = (double *)malloc(szz*sizeof(double));
	cudaMalloc((void**)&ccurrRank, szz*sizeof(double));

	cudaMalloc((void**)&cn, sizeof(long));
	cudaMalloc((void**)&cm, sizeof(long));
	cudaMalloc((void**)&cmem, n*sizeof(long));
	cudaMalloc((void**)&coutdeg, nn*sizeof(long));
	cudaMalloc((void**)&cparent, nn*sizeof(long));
	cudaMalloc((void**)&ctemp, n*sizeof(long));
	cudaMalloc((void**)&cgraph, szz*sizeof(long));
	cudaMalloc((void**)&cmarked, n*sizeof(long));
	cudaMalloc((void**)&crank, nn*sizeof(double));
	cudaMalloc((void**)&cparent, nn*sizeof(long));

	cudaMemcpy(cn, &pivot, sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cmem, mem, n*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(coutdeg, outdeg, nn*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cparent, parent, nn*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(ctemp, temp, n*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cgraph, graphh, szz*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(cparent, parent, nn*sizeof(long), cudaMemcpyHostToDevice);

	do  
	{   
		error=0;
		for(i=0;i<n;i++){
			curr[i]=0;
		}
		for(i=0;i<szz;i++){
			currRank[i]=0;
		}
		
		cudaMemcpy(crank, rank, nn*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(cmarked, marked, n*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(ccurrRank, currRank, szz*sizeof(double), cudaMemcpyHostToDevice);

		long bx = pivot/1024+1;
		dim3 threadB(1024, 1);
		dim3 blockB(bx, 1024/bx);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);

		kernel2test<<<blockB,threadB>>>(cn, cmem, cgraph, ctemp, ccurrRank, crank, coutdeg, cparent, cmarked);
		
		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		total += elapsedTime;

		for(i=pivot;i<limit;i++)
		{   
			{   
				cudaMemcpy(cm, &i, sizeof(long), cudaMemcpyHostToDevice);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);

				kernel2test1<<<1,1024>>>(cm, cmem, cgraph, ctemp, ccurrRank, crank, coutdeg, cparent, cmarked);
				
				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
			}   
		} 
		cudaMemcpy(currRank, ccurrRank, szz*sizeof(double) , cudaMemcpyDeviceToHost);
		for(i=0;i<n;i++){
			for(long kk=0;kk<graph[mapit[i]].size();kk++){
				curr[i]+=currRank[temp[i]+kk];
			}
		}
		double anse=0;
		for(i=0;i<limit;i++)
			if(!marked[i])
				anse=max(anse, fabs(randomp+initial[mapit[i]]+damp*curr[i]-rank[mapit[i]]));
		iterations++;
		for(i=0;i<limit;i++)
		{
			if(!marked[i])   
			{
				rank[mapit[i]]=damp*curr[i]+randomp+initial[mapit[i]];
			}   
		}
		if(iterations%20==0)
		{   
			for(i=0;i<limit;i++)
			{   
				if(!marked[i])
				{   

					if(fabs(prev[i]-curr[i]) < value )marked[i]=1;
					else
						prev[i]=curr[i];
				}   
			}   
		}   
		error = anse;
	}while(error > thres);
	for(i=limit;i<n;i++)
	{   
		long node=mapit[i];
		double val=powers[level[node]];
		rank[node]=rank[redir[node]]*val+(1.0-val)/graph.size();
	}   
	for(i=0;i<left.size();i++)
		rank[left[i]]=rank[parent[left[i]]];
	cudaFree(cn);
	cudaFree(cm);
	cudaFree(cmem);
	cudaFree(coutdeg);
	cudaFree(cparent);
	cudaFree(ctemp);
	cudaFree(cgraph);
	cudaFree(crank);
	cudaFree(cmarked);
	cudaFree(ccurrRank);
	return total;
}

void computerankidc(vector < vector < long > > & graph,long *parent,vector < long > & left,long n,long *outdeg,vector < long > &  mapit,double *rank,double *initial,long *level,long *redir,double *powers)
{
	double damp=0.85;
	double thres=1e-10;
	long i, j;
	vector < double > curr(n);
	vector < double > prev(n,1.0/n);
	double value=((1e-12)*10.0)/double ( n );
	bool *marked = (bool *)malloc(n*sizeof(bool));
	memset(marked,0,n*sizeof(bool));
	double error=0;
	long  iterations=0;
	double randomp=(1-damp)/graph.size();
	long limit=0;
	for(i=0;i<n;i++)
	{
		long node=mapit[i];
		if(node==redir[node])
		{
			long temp=mapit[limit];
			mapit[limit]=node;
			mapit[i]=temp;
			limit++;
		}
	}
	do
	{
		error=0;
		for(i=0;i<limit;i++)
		{
			long node=mapit[i];
			if(!marked[i])
			{
				double ans=0;
				for(j=0;j<graph[node].size();j++)
				{
					ans=ans+rank[parent[graph[node][j]]]/outdeg[graph[node][j]];
				}
				curr[i]=randomp+damp*ans+initial[mapit[i]];
				error=max(error,fabs(curr[i]-rank[node]));
			}
		}
		for(i=0;i<limit;i++) if(!marked[i])
			rank[mapit[i]]=curr[i];
		iterations++;
		if(iterations%20==0){
			for(i=0;i<limit;i++)
			{
				if(!marked[i])
				{
					if(fabs(prev[i]-curr[i]) < value )marked[i]=1;
					else prev[i]=curr[i];
				}
			}
		}
	}while(error > thres );
	for(i=limit;i<n;i++)
	{
		long node=mapit[i];
		double val=powers[level[node]];
		rank[node]=rank[redir[node]]*val+(1.0-val)/graph.size();
	}
	for(i=0;i<left.size();i++)
		rank[left[i]]=rank[parent[left[i]]];
}