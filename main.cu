#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <fstream>
#include <set>
#include <map>
#include <chrono>
#include "functions.cuh"

using namespace std;
using namespace std::chrono;

#define _MAX 100000

// total time taken by kernel computations
double total = 0.0;

long inc=0;

// disjolong set union
// find the parent
long find(long u,long *parent){
	if(parent[u]<0) return u;
	return parent[u]=find(parent[u],parent);
}

// union by size 
// parent[u] = -1 if parent
// 		size if not parent (part of some set)
long unionit(long u,long v,long *parent){
	long pu=find(u,parent);
	long pv=find(v,parent);
	// same set
	if(pu==pv) return 0;
	if(-parent[pu]>-parent[pv]){   
		parent[pu]=parent[pu]+parent[pv];
		parent[pv]=pu;
	}   
	else{   
		parent[pv]=parent[pu]+parent[pv];
		parent[pu]=pv;
	}   
	return 1;
}

// dfs 
// visit = stores the values the nodes when they leave the computation (same as stack)
// kosaraju's algorithm's part
// nvisit = visited or not
// dfs uses original graph
void dfs(vector < vector < long > > & graph,long *visit,long *nvisit,long node){
	nvisit[node]=1; // visited
	for(long i=0;i<graph[node].size();i++)
		if(nvisit[graph[node][i]]==-1) // if not visited
			dfs(graph,visit,nvisit,graph[node][i]);
	visit[inc++]=node; // enter the stack 
}

// SCC computations
// com = component number
// nvisit = nodes
// rdfs uses graph with reverse edges
// kosaraju's algorithm's part
void rdfs(vector < vector < long > > & graph,long *nvisit,long node,long *component,long com){
	nvisit[node]=1; // visited
	component[node]=com; // set component of the node
	for(long i=0;i<graph[node].size();i++)
		if(nvisit[graph[node][i]]==-1) // if not visited
			rdfs(graph,nvisit,graph[node][i],component,com);
}

// topological order of SCCs
// graph = component graph
//		where nodes = Components
//		      edges between component if there is an edge from one component to another
// order = topological order = level by level nodes
// visit = visited or not
void topobfs(vector < vector < long > > & graph, long *order, long *visit){
	long i,j;
	queue < long > line; // for bfs
	memset(visit, -1, sizeof(long)*graph.size());
	long indegree[graph.size()];
	memset(indegree,0,graph.size()*sizeof(long));
	// indegree of the component (number of components which have incoming edges to the component)
	for(i=0;i<graph.size();i++){
		for(j=0;j<graph[i].size();j++){
			indegree[graph[i][j]]++;
		}
	}
	for(i=0;i<graph.size();i++){
		// add the nodes which have indegree = 0 longo the line (queue)
		if(!indegree[i]) {
			line.push(i);
			visit[i]=0;
			indegree[i]--;
		}
	}
	// bfs
	// order = topological order of components
	while(!line.empty()){
		long node=line.front();
		line.pop();
		order[inc++]=node;
		for(i=0;i<graph[node].size();i++)
		{
			indegree[graph[node][i]]--;
			if(indegree[graph[node][i]]==0)
			{
				line.push(graph[node][i]);
				visit[graph[node][i]]=visit[node]+1;
			}
		}
	}
}

// optchain = chain node computation should be included or not
// optdead = dead node computation should be included or not
// optident = identical node computation should be included or not

// computation for optchain and optident is included
// for optdead, set optdead = 1 in main function
long optchain=0, optdead=0, optident=0;

int main(){
	// start time 
    auto start = high_resolution_clock::now();

	ifstream fin; // input file
	fin.open("./../input.txt");
	
	ofstream fout; // output file
	fout.open("./../outputTest3.txt");
	
	// n = number of nodes
	// m = number of edges
	long n,m;
	fin >> n >> m;
	
	// graph = original graph
	// rgraph = graph with reversed edges
	// rcgraph = graph with edges within component
	// rcwgraph = graph with edges (one component to another) (cross edges)
	long i,j;
	vector < vector < long > > graph(n), rgraph(n), rcgraph(n), rcwgraph(n);

	// outdegree of node
	long *outdeg = (long *)malloc(n*sizeof(long));
	memset(outdeg,0,n*sizeof(long));

	// below computation is to avoid any unusual node values
	/*
		for example,
			n = 5
			values of the nodes = 12 234 2312 123121 1232122121
			we convert it longo = 0 1 2 3 4 5
		we first add values of the nodes longo set
		assign each element of the set correspoing index values (map is used for this/ hash)
	*/
	// set to store the values of the nodes
	// set<long> s;
	// // edges of the node (u,v) (u->v)
	// vector<pair<long,long>> edgess;
	
	// for(i=0;i<m;i++){
	// 	long u,v;
	// 	fin >> u >> v; // input edges
	// 	s.insert(u); // insert node u longo set
	// 	s.insert(v); // insert node v longo set
	// 	edgess.push_back(make_pair(u,v)); // insert edge (u,v) longo edges 
	// }
	// // hash to assign 0 to n-1 to nodes
	// map<long,long> hash;
	// long cnt=0;
	// for(auto k:s){ // for every element of set
	// 	hash[k]=cnt++;
	// }

	// create graph using hash values
	for(i=0;i<m;i++){
		long u,v;
		fin >> u >> v;
		// long u=hash[edgess[i].first], v = hash[edgess[i].second];
		graph[u].push_back(v); // add edge longo the graph
		rgraph[v].push_back(u); // reversed edge
		outdeg[u]++; // outdegree
	}

	long *visit = (long *)malloc(n*sizeof(long));
	memset(visit, -1, n*sizeof(long));

	long *component = (long *)malloc(n*sizeof(long));
	memset(component, -1, n*sizeof(long));
	
	long *nvisit = (long *)malloc(n*sizeof(long));
	memset(nvisit, -1, n*sizeof(long));
	
	// kosaraju's algorithm computation
	for(i=0;i<n;i++){
		if(nvisit[i]==-1) { // if not visited
			dfs(graph,visit,nvisit,i);	
		}
	}

	memset(nvisit,-1,n*sizeof(long));
	
	// component number
	long com=0;
	for(i=n-1;i>=0;i--){
		if(nvisit[visit[i]]==-1){ // if not visited
			rdfs(rgraph,nvisit,visit[i],component,com); // scc computation
			com++;
		}
	}

	// create graph rcgraph and rcwgraph
	// rcgraph = graph with edges within components (using reverse edges)
	// rcwgraph = graph with edges (one component to another) (cross edges) (using reverse edges)
	for(i=0;i<n;i++){
		for(j=0;j<rgraph[i].size();j++){
			if(component[i]==component[rgraph[i][j]]){ 
				// if in the same component
				rcgraph[i].push_back(rgraph[i][j]);
			}
			else{ 
				// if in the diff components
				// cross edges
				rcwgraph[i].push_back(rgraph[i][j]);
			}
		}
	}
	
	// members[i] = members of the component i
	// compgr = component graph
	//		where nodes = Components
	//		      edges between component if there is an edge from one component to another
	vector < vector < long > > members(com), compgr(com);
	
	// create component graph
	for(i=0;i<n;i++){
		for(j=0;j<graph[i].size();j++){
			if(component[i]!=component[graph[i][j]]){ // if edge is cross edge == add edge longo the component graph
				// might be the case where multiple edges from one to another component
				// can be ignored as it doesn't affect the results
				compgr[component[i]].push_back(component[graph[i][j]]);
			}
		}
	}

	long *order = (long *)malloc(com*sizeof(long));
	memset(nvisit,0,n*sizeof(long));
	
	inc=0;
	// order = topological order = level by level components
	// nvisit = level of the component
	topobfs(compgr,order,nvisit);
	
	// calculation to include the optident or not
	long *number = (long *)malloc(n*sizeof(long));
	memset(number,0,n*sizeof(long));
	
	// number = number of nodes which have only one incoming edge from this node
	for(i=0;i<n;i++){
		if(rgraph[i].size()==1){
			number[rgraph[i][0]]++;
		}
	}

	// equiperc = number of nodes which will save computation
	long equiperc=0;
	for(i=0;i<n;i++){
		equiperc=equiperc+max((long)0,number[i]-1);
	}
	
	// saved computations nodes/total nodes
	double vai=double(equiperc)/n;
	// total edges/total nodes
	double ratio=double(m)/n;

	if(vai>0.06 && ratio>3.0)
		optident=1;
	
	// parent2 is parent of chain if any
	long *parent2 = (long *)malloc(n*sizeof(long));
	memset(parent2,-1,n*sizeof(long));
	
	// parent1 size of the chain
	long *parent1 = (long *)malloc(n*sizeof(long));
	memset(parent1,-1,n*sizeof(long));
	
	for(i=0;i<n;i++){
		if(rgraph[i].size()>1 || graph[i].size()>1 ) 
			continue;
		for(j=0;j<rcgraph[i].size();j++){
			if(graph[rcgraph[i][j]].size()>1 || rgraph[rcgraph[i][j]].size()>1) 
				continue;
			if(unionit(rcgraph[i][j],i,parent1)){
				// i -> rcgraph[i][j] in reverse graph
				// i <- rcgraph[i][j] in original graph
				// parent2[i] = rcgraph[i][j] if part of the chain
				parent2[i]=rcgraph[i][j];
			}
		}
	}
	
	// redir = head of the chain
	long *redir = (long *)malloc(n*sizeof(long));
	for(i=0;i<n;i++){
		redir[i]=i;
	}

	// levelz = level at which node appear in the node
	long *levelz = (long *)malloc(n*sizeof(long));
	memset(levelz,0,n*sizeof(long));

	// powers = used for pagerank computation of chain nodes
	// powers[k] = (0.85)^k;
	double *powers = (double *)malloc(n*sizeof(double));
	powers[0]=1;
	for(i=1;i<n;i++){
		powers[i]=powers[i-1]*0.85;
	}
	
	// vac = number of nodes in chain
	long vac=0;
	
	for(i=0;i<n;i++)
	{
		if(rgraph[i].size()>1 || graph[i].size()>1 ) continue;
		if(parent2[i]!=-1) continue;
		long node=i;
		long iterations=0;
		while(graph[node].size())
		{
			node=graph[node][0];
			if(component[node]!=component[i] || node==i || graph[node].size()>1 || rgraph[node].size()>1) break;
			iterations++;
			redir[node]=i;
			levelz[node]=iterations;
		}
		// iterations = chain size
		vac=vac+iterations;
	}

	// chain size/total nodes
	double rac=double(vac)/n;
	if(rac>0.2)
		optchain=1;

	// tempg, edges, rcw = for CSR
	// edges = edge list
	// rccw[i] = rcwgrah[i].size()
	// tempg = prefix sum of crcw
	long *tempg = (long *)malloc(n*sizeof(long));
	for(long i1=0;i1<n;i1++){
		if(i1) tempg[i1]=tempg[i1-1]+rcwgraph[i1-1].size();
		else tempg[i1]=0;
	}
	long szzz = tempg[n-1]+rcwgraph[n-1].size();
	long kkk=0;
	long *edges = (long *)malloc(szzz*sizeof(long));
	for(long i1=0;i1<n;i1++){
		for(long c:rcwgraph[i1]){
			edges[kkk++]=c;
		}
	}

	long *rcw = (long *)malloc(n*sizeof(long));
	for(long i1=0;i1<n;i1++){
		rcw[i1] = rcwgraph[i1].size();
	}

	long *cstart, *cend, *corder, *cmemsz, *ctemp, *coutdeg,
					*cmembers, *ctempg, *cedges, *crcw;
	double *crank;

	cudaMalloc((void**)&cstart, sizeof(long));
	cudaMalloc((void**)&cend, sizeof(long));
	cudaMalloc((void**)&corder, com*sizeof(long));
	cudaMalloc((void**)&cmemsz, com*sizeof(long));
	cudaMalloc((void**)&ctemp, com*sizeof(long));
	cudaMalloc((void**)&crcw, n*sizeof(long));
	cudaMalloc((void**)&crank, n*sizeof(double));
	cudaMalloc((void**)&coutdeg, n*sizeof(long));
	cudaMalloc((void**)&ctempg, n*sizeof(long));
	cudaMalloc((void**)&cedges, szzz*sizeof(long));

	// corder = level by level ordering of components
	cudaMemcpy(corder, order, com*sizeof(long), cudaMemcpyHostToDevice);
	// ctempg = prefix sum of crcw
	cudaMemcpy(ctempg, tempg, n*sizeof(long), cudaMemcpyHostToDevice);
	// cedges = edge list of rcwgraph
	cudaMemcpy(cedges, edges, szzz*sizeof(long), cudaMemcpyHostToDevice);
	// crcw[i] = size of rcwgraph[i]
	cudaMemcpy(crcw, rcw, n*sizeof(long), cudaMemcpyHostToDevice);
	// coutdeg = outdegree of nodes
	cudaMemcpy(coutdeg, outdeg, n*sizeof(long), cudaMemcpyHostToDevice);

	// rank = pageranks
	double *rank = (double *)malloc(n*sizeof(double));
	for(i=0;i<n;i++){
		// initialize with 1.0/n
		rank[i]=1.0/n;
	}

	long check_size=szzz;
	double *check_initial = (double *)malloc(check_size * sizeof(long));
	double *ccheck_initial;
	cudaMalloc((void**)&ccheck_initial, check_size*sizeof(double));
	cout << check_size << "\n";
	cudaMemcpy(ccheck_initial, check_initial, check_size*sizeof(double), cudaMemcpyHostToDevice);
	cout << optident << " " << optchain << " " << optdead << "\n";

	if(optident==1 && optchain==0 && optdead==0)
	{
		// parent/head of the identical nodes
		long *parent = (long *)malloc(n*sizeof(long));
		// identical nodes whose pagerank won't be calculated
		// they would just be equated to pagerank of parent/head node
		vector < vector < long > > left(com);
		for(i=0;i<n;i++)
			parent[i]=i;
		// hash values to find identical nodes
		vector < vector <  pair  <  pair < long , long > , long >  > > hvalues(n);
		for(i=0;i<n;i++)
		{
			if(rgraph[i].size()!=1 && rgraph[i].size()!=2) continue;
			if(rgraph[i].size()==1)
			{
				hvalues[(rgraph[i][0])%n].push_back(make_pair(make_pair(rgraph[i][0],component[i]),i));
			}
			else
			{
				long long val=(long long)max(rgraph[i][1]+1,rgraph[i][0]+1)*(n+1)+min(rgraph[i][0]+1,rgraph[i][1]+1);
				hvalues[(val)%n].push_back(make_pair(make_pair(val,component[i]),i));
			}
		}

		for(i=0;i<n;i++)
			sort(hvalues[i].begin(),hvalues[i].end());
		for(long k=0;k<n;k++)
		{
			for(i=0;i<hvalues[k].size();i++)
			{
				for(j=i;j<hvalues[k].size() && hvalues[k][j].first==hvalues[k][i].first ;j++)
				{
					parent[hvalues[k][j].second]=hvalues[k][i].second;
				}
				i=j-1;
			}
		}
		hvalues.clear();
		long noo=0;
		for(i=0;i<n;i++){
			if(parent[i]==i) 
			{
				// members of the component whose pagerank will be computed
				members[component[i]].push_back(i);
			}
			else
			{
				// members of the component which are identical to some other node
				left[component[i]].push_back(i);
				noo++;
			}
		}
		// par stores the first component number of every level 
		vector < long > par;
		par.push_back(0);
		for(i=0;i<com;i++)
		{
			long j=i;
			while(j<com && nvisit[order[j]]==nvisit[order[i]]) // component are on same level increment j
				j++;
			// now j is on the next level
			par.push_back(j);
			i=j-1;
		}
		long thresh=_MAX; // to sort the components
		// initial array for storing the contribution of one component to another (pagerank of nodes)
		double *initial = (double *)malloc(n * sizeof(double));
		memset(initial,0,n * sizeof(double));
		long w;

		// memsz, mem, temp for CSR
		// memsz = size of members
		// temp = prefix sum of memsz
		// mem = list of members
		long *memsz = (long *)malloc(com*sizeof(long));
		long *temp = (long *)malloc(com*sizeof(long));
		long szz=0;
		for(long i1=0;i1<com;i1++){
			memsz[i1]=members[order[i1]].size();
			szz+=members[order[i1]].size();
		}

		for(long i1=0;i1<com;i1++){
			if(i1) temp[i1]=temp[i1-1]+memsz[i1-1];
			else temp[i1]=0;
		}

		long kk=0;
		long *mem = (long *)malloc(szz*sizeof(long));

		for(long i1=0;i1<com;i1++){
			for(long c:members[order[i1]]){
				mem[kk++]=c;
			}
		}

		// cout << "Check\n";

		cudaMalloc((void**)&cmembers, szz*sizeof(long));
		
		cudaMemcpy(cmemsz, memsz, com*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(ctemp, temp, com*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(cmembers, mem, szz*sizeof(long), cudaMemcpyHostToDevice);

		// cout << "Problem?\n";

		

		// cout << "Now?\n";
		// cudaMemcpy(ccheck_pre, check_pre, 
			// n * sizeof(long), cudaMemcpyHostToDevice);

		cout << "Level: " << par.size() << " " << check_size << "\n";

		for(i=0;i<par.size()-1;i++)
		{
			// [par[i], par[i+1]) components on the same level
			long pivot=par[i];
			// [par[i], pivot) components with total edges > thresh
			// [pivot, par[i+1]) components with total edges < thresh
			
			// sorting
			for(w=par[i];w<par[i+1];w++)
			{
				// sum = total edges
				long sum=0;
				for(j=0;j<members[order[w]].size();j++)
					sum=sum+rgraph[members[order[w]][j]].size();
				if(sum>thresh)
				{
					long temp=order[pivot];
					order[pivot]=order[w];
					order[w]=temp;
					pivot++;
				}
			}
			
			// Pagerank computation (Contribution of nodes of one component to another)
			// par[i] to pivot => Computation for all components separatly
			for(w=par[i];w<pivot;w++)
			{
				// cn = current component
				// cout << "Entering...\n";

				long *cn;
				cudaMalloc((void**)&cn, sizeof(long));
				cudaMemcpy(cn, &w, sizeof(long), cudaMemcpyHostToDevice);

				dim3 threadB(32,32);
				dim3 blockB(32,32);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);

				kerneltest1<<<blockB,threadB>>>(cn, cend, cmemsz, cmembers, 
													crcw, ccheck_initial, crank,
														cedges, coutdeg, corder, ctemp, ctempg);

				cudaDeviceSynchronize();
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
				cudaFree(cn);
			}
			
			// cout << "Cleared phase 1\n";

			// pivot to par[i+1] => Computation for all components at the same time (by one kernel)
			if(pivot < par[i+1])
			{
				cudaMemcpy(cstart, &pivot, sizeof(long), cudaMemcpyHostToDevice);
				cudaMemcpy(cend, &par[i+1], sizeof(long), cudaMemcpyHostToDevice);
				// cudaMemcpy(cinitial, initial, n*sizeof(double), cudaMemcpyHostToDevice);
				cudaMemcpy(crank, rank, n*sizeof(double), cudaMemcpyHostToDevice);

				dim3 threadB(8,8,16);
				dim3 blockB(8,8,16);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);

				// cstart = pivot
				// cend = par[i+1]
				// cmemsz = member size
				// cmembers = list of all members
				// crcw = cross edge graph size
				// cinitial = contribution of one node to another (different components) 
				// crank = Pagerank
				// cedges = edge list
				// coutdeg = outdegree of the nodes
				// corder = level by level topological component ordering
				// ctemp = prefix sum of cmemsz
				// ctempg = prefix sum of crcw
				kerneltest<<<blockB,threadB>>>(cstart, cend, cmemsz, cmembers, crcw, ccheck_initial, crank,
								cedges, coutdeg, corder, ctemp, ctempg);

				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
				// cudaMemcpy(initial, cinitial, n*sizeof(double), cudaMemcpyDeviceToHost);
			}

			cudaMemcpy(check_initial, ccheck_initial, check_size * sizeof(double), cudaMemcpyDeviceToHost);
			for(j=par[i];j<par[i+1];j++){
				for(long kk=0;kk<members[order[j]].size();kk++){
					long node = members[order[j]][kk];
					for(long ll=0;ll<rcwgraph[node].size();ll++){
						initial[node] += check_initial[tempg[node]+ll];
					}
				}
			}

			// cout << "Cleared phase 2\n";

			// cout << "Not here\n";
			
			// Pagerank computation within the component
			for(j=par[i];j<pivot;j++){
				total += computeparalleli(rcgraph,parent,left[order[j]],members[order[j]].size(),outdeg,members[order[j]],rank,initial, n);
			}
			for(j=pivot;j<par[i+1];j++){
				computeranki(rcgraph,parent,left[order[j]],members[order[j]].size(),outdeg,members[order[j]],rank,initial, n);
			}
		}
	}

	if(optident==1 && optchain==0 && optdead==1)	
	{
		long *parent = (long *)malloc(n*sizeof(long));
		vector < vector < long > > left(com);
		for(i=0;i<n;i++){
			parent[i]=i;
		}
		vector < vector <  pair  <  pair < long , long > , long >  > > hvalues(n);
		for(i=0;i<n;i++){
			if(rgraph[i].size()!=1 && rgraph[i].size()!=2) 
				continue;
			if(rgraph[i].size()==1){
				hvalues[(rgraph[i][0])%n].push_back(make_pair(make_pair(rgraph[i][0],component[i]),i));
			}
			else{
				long val=max(rgraph[i][1]+1,rgraph[i][0]+1)*(long)(n+1)+min(rgraph[i][0]+1,rgraph[i][1]+1);
				hvalues[(val)%(long)n ].push_back(make_pair(make_pair(val,component[i]),i));
			}
		}
		for(i=0;i<n;i++){
			sort(hvalues[i].begin(),hvalues[i].end());
		}
		for(long k=0;k<n;k++){
			for(i=0;i<hvalues[k].size();i++){
				for(j=i;j<hvalues[k].size() && hvalues[k][j].first==hvalues[k][i].first ;j++){
					parent[hvalues[k][j].second]=hvalues[k][i].second;
				}
				i=j-1;
			}
		}
		hvalues.clear();
		long noo=0;
		for(i=0;i<n;i++){
			if(parent[i]==i){
				members[component[i]].push_back(i);
			}
			else{
				left[component[i]].push_back(i);
				noo++;
			}
		}

		vector < long > par;
		par.push_back(0);
		for(i=0;i<com;i++){
			long j=i;
			while(j<com && nvisit[order[j]]==nvisit[order[i]]){
				j++;
			}
			par.push_back(j);
			i=j-1;
		}
		
		double *initial = (double *)malloc(n*sizeof(double));
		memset(initial,0,n*sizeof(double));

		long *memsz = (long *)malloc(com*sizeof(long));
		long *temp = (long *)malloc(com*sizeof(long));
		long szz=0;
		for(long i1=0;i1<com;i1++){
			memsz[i1]=members[order[i1]].size();
			szz+=members[order[i1]].size();
		}
		for(long i1=0;i1<com;i1++){
			if(i1) temp[i1]=temp[i1-1]+memsz[i1-1];
			else temp[i1]=0;
		}
		long kk=0;
		long *mem = (long *)malloc(szz*sizeof(long));
		for(long i1=0;i1<com;i1++){
			for(long c:members[order[i1]]){
				mem[kk++]=c;
			}
		}

		cudaMalloc((void**)&cmembers, szz*sizeof(long));
		
		cudaMemcpy(cmemsz, memsz, com*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(ctemp, temp, com*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(cmembers, mem, szz*sizeof(long), cudaMemcpyHostToDevice);
		
		long w;
		long thresh=_MAX;

		for(i=0;i<par.size()-1;i++){
			long pivot=par[i];
			for(w=par[i];w<par[i+1];w++)
			{
				long sum=0;
				for(j=0;j<members[order[w]].size();j++)
					sum=sum+rgraph[members[order[w]][j]].size();
				if(sum>thresh)
				{
					long temp=order[pivot];
					order[pivot]=order[w];
					order[w]=temp;
					pivot++;
				}
			}
			for(w=par[i];w<pivot;w++)
			{
				long *cn;
				cudaMalloc((void**)&cn, sizeof(long));
				cudaMemcpy(cn, &w, sizeof(long), cudaMemcpyHostToDevice);

				dim3 threadB(32,32);
				dim3 blockB(32,32);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);
				
				// cn = w
				// cmemsz = member size
				// cmembers = list of all members
				// crcw = cross edge graph size
				// cinitial = contribution of one node to another (different components) 
				// crank = Pagerank
				// cedges = edge list
				// coutdeg = outdegree of the nodes
				// corder = level by level topological component ordering
				// ctemp = prefix sum of cmemsz
				// ctempg = prefix sum of crcw
				kerneltest1<<<blockB,threadB>>>(cn, cend, cmemsz, cmembers, crcw, ccheck_initial, crank,
								cedges, coutdeg, corder, ctemp, ctempg);

				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;

				cudaFree(cn);
			}

			if(pivot < par[i+1]){
				cudaMemcpy(cstart, &pivot, sizeof(long), cudaMemcpyHostToDevice);
				cudaMemcpy(cend, &par[i+1], sizeof(long), cudaMemcpyHostToDevice);
				cudaMemcpy(crank, rank, n*sizeof(double), cudaMemcpyHostToDevice);

				dim3 threadB(8,8,16);
				dim3 blockB(8,8,16);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);
				
				// cstart = pivot
				// cend = par[i+1]
				// cmemsz = member size
				// cmembers = list of all members
				// crcw = cross edge graph size
				// cinitial = contribution of one node to another (different components) 
				// crank = Pagerank
				// cedges = edge list
				// coutdeg = outdegree of the nodes
				// corder = level by level topological component ordering
				// ctemp = prefix sum of cmemsz
				// ctempg = prefix sum of crcw
				kerneltest<<<blockB,threadB>>>(cstart, cend, cmemsz, cmembers, crcw, ccheck_initial, crank,
								cedges, coutdeg, corder, ctemp, ctempg);

				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
			}
			cudaMemcpy(check_initial, ccheck_initial, check_size * sizeof(double), cudaMemcpyDeviceToHost);
			for(j=par[i];j<par[i+1];j++){
				for(long kk=0;kk<members[order[j]].size();kk++){
					long node = members[order[j]][kk];
					for(long ll=0;ll<rcwgraph[node].size();ll++){
						initial[node] += check_initial[tempg[node]+ll];
					}
				}
			}
			for(j=par[i];j<pivot;j++){
				total += computeparallelid(rcgraph,parent,left[order[j]],members[order[j]].size(),outdeg,members[order[j]],rank,initial, n);
			}
			for(j=pivot;j<par[i+1];j++){
				computerankid(rcgraph,parent,left[order[j]],members[order[j]].size(),outdeg,members[order[j]],rank,initial, n);
			}
		}
	}

	if(optident==0 && optchain==0 && optdead==0)
	{
		vector < long > par;
		par.push_back(0);

		for(i=0;i<com;i++){
			long j=i;
			while(j<com && nvisit[order[j]]==nvisit[order[i]]){
				j++;
			}
			par.push_back(j);
			i=j-1;
		}
		
		double *initial = (double *)malloc(n*sizeof(double));
		memset(initial,0,n*sizeof(double));
		
		for(i=0;i<n;i++){
			members[component[i]].push_back(i);
		}

		long *memsz = (long *)malloc(com*sizeof(long));
		long *temp = (long *)malloc(com*sizeof(long));
		long szz=0;

		for(long i1=0;i1<com;i1++){
			memsz[i1]=members[order[i1]].size();
			szz+=members[order[i1]].size();
		}
		for(long i1=0;i1<com;i1++){
			if(i1) temp[i1]=temp[i1-1]+memsz[i1-1];
			else temp[i1]=0;
		}
		long kk=0;
		long *mem = (long *)malloc(szz*sizeof(long));
		for(long i1=0;i1<com;i1++){
			for(long c:members[order[i1]]){
				mem[kk++]=c;
			}
		}

		cudaMalloc((void**)&cmembers, szz*sizeof(long));
		
		cudaMemcpy(cmemsz, memsz, com*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(ctemp, temp, com*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(cmembers, mem, szz*sizeof(long), cudaMemcpyHostToDevice);

		long w;
		long thresh=_MAX;

		for(i=0;i<par.size()-1;i++){
			long pivot=par[i];
			for(w=par[i];w<par[i+1];w++)
			{
				long sum=0;
				for(j=0;j<members[order[w]].size();j++)
					sum=sum+rgraph[members[order[w]][j]].size();
				if(sum>thresh)
				{
					long temp=order[pivot];
					order[pivot]=order[w];
					order[w]=temp;
					pivot++;
				}
			}
			for(w=par[i];w<pivot;w++)
			{
				long *cn;
				cudaMalloc((void**)&cn, sizeof(long));
				cudaMemcpy(cn, &w, sizeof(long), cudaMemcpyHostToDevice);

				dim3 threadB(32,32);
				dim3 blockB(32,32);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);

				// cn = w
				// cmemsz = member size
				// cmembers = list of all members
				// crcw = cross edge graph size
				// cinitial = contribution of one node to another (different components) 
				// crank = Pagerank
				// cedges = edge list
				// coutdeg = outdegree of the nodes
				// corder = level by level topological component ordering
				// ctemp = prefix sum of cmemsz
				// ctempg = prefix sum of crcw
				kerneltest1<<<blockB,threadB>>>(cn, cend, cmemsz, cmembers, crcw, ccheck_initial, crank,
								cedges, coutdeg, corder, ctemp, ctempg);

				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;

				// cudaMemcpy(initial, cinitial, n*sizeof(double), cudaMemcpyDeviceToHost);
				cudaFree(cn);
			}

			if(pivot < par[i+1]){
				cudaMemcpy(cstart, &pivot, sizeof(long), cudaMemcpyHostToDevice);
				cudaMemcpy(cend, &par[i+1], sizeof(long), cudaMemcpyHostToDevice);
				// cudaMemcpy(cinitial, initial, n*sizeof(double), cudaMemcpyHostToDevice);
				cudaMemcpy(crank, rank, n*sizeof(double), cudaMemcpyHostToDevice);

				dim3 threadB(8,8,16);
				dim3 blockB(8,8,16);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);
				
				// cstart = pivot
				// cend = par[i+1]
				// cmemsz = member size
				// cmembers = list of all members
				// crcw = cross edge graph size
				// cinitial = contribution of one node to another (different components) 
				// crank = Pagerank
				// cedges = edge list
				// coutdeg = outdegree of the nodes
				// corder = level by level topological component ordering
				// ctemp = prefix sum of cmemsz
				// ctempg = prefix sum of crcw
				kerneltest<<<blockB,threadB>>>(cstart, cend, cmemsz, cmembers, crcw, ccheck_initial, crank,
								cedges, coutdeg, corder, ctemp, ctempg);

				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
				// cudaMemcpy(initial, cinitial, n*sizeof(double), cudaMemcpyDeviceToHost);
			}
			cudaMemcpy(check_initial, ccheck_initial, check_size * sizeof(double), cudaMemcpyDeviceToHost);
			for(j=par[i];j<par[i+1];j++){
				for(long kk=0;kk<members[order[j]].size();kk++){
					long node = members[order[j]][kk];
					for(long ll=0;ll<rcwgraph[node].size();ll++){
						initial[node] += check_initial[tempg[node]+ll];
					}
				}
			}
			for(j=par[i];j<pivot;j++){
				total += computeparallel(rcgraph,members[order[j]].size(),outdeg,members[order[j]],rank,initial, n);
			}
			for(j=pivot;j<par[i+1];j++){
				computerank(rcgraph,members[order[j]].size(),outdeg,members[order[j]],rank,initial);
			}
		}
	}

	if(optident==0 && optchain==0 && optdead==1)
	{
		vector < long > par;
		par.push_back(0);
		for(i=0;i<com;i++){
			long j=i;
			while(j<com && nvisit[order[j]]==nvisit[order[i]]){
				j++;
			}
			par.push_back(j);
			i=j-1;
		}

		double *initial = (double *)malloc(n*sizeof(double));
		memset(initial,0,n*sizeof(double));
		
		for(i=0;i<n;i++){
			members[component[i]].push_back(i);
		}

		long *memsz = (long *)malloc(com*sizeof(long));
		long *temp = (long *)malloc(com*sizeof(long));
		long szz=0;
		for(long i1=0;i1<com;i1++){
			memsz[i1]=members[order[i1]].size();
			szz+=members[order[i1]].size();
		}
		for(long i1=0;i1<com;i1++){
			if(i1) temp[i1]=temp[i1-1]+memsz[i1-1];
			else temp[i1]=0;
		}
		long kk=0;
		long *mem = (long *)malloc(szz*sizeof(long));
		for(long i1=0;i1<com;i1++){
			for(long c:members[order[i1]]){
				mem[kk++]=c;
			}
		}
		
		cudaMalloc((void**)&cmembers, szz*sizeof(long));
		
		cudaMemcpy(cmemsz, memsz, com*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(ctemp, temp, com*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(cmembers, mem, szz*sizeof(long), cudaMemcpyHostToDevice);
		
		long w;
		long thresh=_MAX;

		for(i=0;i<par.size()-1;i++){
			long pivot=par[i];
			for(w=par[i];w<par[i+1];w++)
			{
				long sum=0;
				for(j=0;j<members[order[w]].size();j++)
					sum=sum+rgraph[members[order[w]][j]].size();
				if(sum>thresh)
				{
					long temp=order[pivot];
					order[pivot]=order[w];
					order[w]=temp;
					pivot++;
				}
			}
			for(w=par[i];w<pivot;w++)
			{
				long *cn;
				cudaMalloc((void**)&cn, sizeof(long));
				cudaMemcpy(cn, &w, sizeof(long), cudaMemcpyHostToDevice);

				dim3 threadB(32,32);
				dim3 blockB(32,32);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);
				
				// cn = w
				// cmemsz = member size
				// cmembers = list of all members
				// crcw = cross edge graph size
				// cinitial = contribution of one node to another (different components) 
				// crank = Pagerank
				// cedges = edge list
				// coutdeg = outdegree of the nodes
				// corder = level by level topological component ordering
				// ctemp = prefix sum of cmemsz
				// ctempg = prefix sum of crcw
				kerneltest1<<<blockB,threadB>>>(cn, cend, cmemsz, cmembers, crcw, ccheck_initial, crank,
								cedges, coutdeg, corder, ctemp, ctempg);

				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;

				// cudaMemcpy(initial, cinitial, n*sizeof(double), cudaMemcpyDeviceToHost);
				cudaFree(cn);
			}

			if(pivot < par[i+1]){
				cudaMemcpy(cstart, &pivot, sizeof(long), cudaMemcpyHostToDevice);
				cudaMemcpy(cend, &par[i+1], sizeof(long), cudaMemcpyHostToDevice);
				// cudaMemcpy(cinitial, initial, n*sizeof(double), cudaMemcpyHostToDevice);
				cudaMemcpy(crank, rank, n*sizeof(double), cudaMemcpyHostToDevice);

				dim3 threadB(8,8,16);
				dim3 blockB(8,8,16);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);
				
				// cstart = pivot
				// cend = par[i+1]
				// cmemsz = member size
				// cmembers = list of all members
				// crcw = cross edge graph size
				// cinitial = contribution of one node to another (different components) 
				// crank = Pagerank
				// cedges = edge list
				// coutdeg = outdegree of the nodes
				// corder = level by level topological component ordering
				// ctemp = prefix sum of cmemsz
				// ctempg = prefix sum of crcw
				kerneltest<<<blockB,threadB>>>(cstart, cend, cmemsz, cmembers, crcw, ccheck_initial, crank,
								cedges, coutdeg, corder, ctemp, ctempg);

				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
				// cudaMemcpy(initial, cinitial, n*sizeof(double), cudaMemcpyDeviceToHost);
			}
			cudaMemcpy(check_initial, ccheck_initial, check_size * sizeof(double), cudaMemcpyDeviceToHost);
			for(j=par[i];j<par[i+1];j++){
				for(long kk=0;kk<members[order[j]].size();kk++){
					long node = members[order[j]][kk];
					for(long ll=0;ll<rcwgraph[node].size();ll++){
						initial[node] += check_initial[tempg[node]+ll];
					}
				}
			}
				
			for(j=par[i];j<pivot;j++){
				total += computeparalleld(rcgraph,members[order[j]].size(),outdeg,members[order[j]],rank,initial, n);
			}
			for(j=pivot;j<par[i+1];j++){
				computerankd(rcgraph,members[order[j]].size(),outdeg,members[order[j]],rank,initial);
			}
		}
	}

	if(optident==0 && optchain==1 && optdead==0)
	{
		vector < long > par;
		par.push_back(0);
		for(i=0;i<com;i++){
			long j=i;
			while(j<com && nvisit[order[j]]==nvisit[order[i]]){
				j++;
			}
			par.push_back(j);
			i=j-1;
		}

		for(i=0;i<n;i++){
			members[component[i]].push_back(i);
		}

		double *initial = (double *)malloc(n*sizeof(double));
		memset(initial,0,n*sizeof(double));

		long *memsz = (long *)malloc(com*sizeof(long));
		long *temp = (long *)malloc(com*sizeof(long));
		long szz=0;
		for(long i1=0;i1<com;i1++){
			memsz[i1]=members[order[i1]].size();
			szz+=members[order[i1]].size();
		}
		for(long i1=0;i1<com;i1++){
			if(i1) temp[i1]=temp[i1-1]+memsz[i1-1];
			else temp[i1]=0;
		}
		long kk=0;
		long *mem = (long *)malloc(szz*sizeof(long));
		for(long i1=0;i1<com;i1++){
			for(long c:members[order[i1]]){
				mem[kk++]=c;
			}
		}

		cudaMalloc((void**)&cmembers, szz*sizeof(long));
		
		cudaMemcpy(cmemsz, memsz, com*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(ctemp, temp, com*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(cmembers, mem, szz*sizeof(long), cudaMemcpyHostToDevice);
		
		long w;
		long thresh=_MAX;

		for(i=0;i<par.size()-1;i++){
			long pivot=par[i];
			for(w=par[i];w<par[i+1];w++)
			{
				long sum=0;
				for(j=0;j<members[order[w]].size();j++)
					sum=sum+rgraph[members[order[w]][j]].size();
				if(sum>thresh)
				{
					long temp=order[pivot];
					order[pivot]=order[w];
					order[w]=temp;
					pivot++;
				}
			}
			for(w=par[i];w<pivot;w++)
			{
				long *cn;
				cudaMalloc((void**)&cn, sizeof(long));
				cudaMemcpy(cn, &w, sizeof(long), cudaMemcpyHostToDevice);

				dim3 threadB(32,32);
				dim3 blockB(32,32);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);
				
				// cn = w
				// cmemsz = member size
				// cmembers = list of all members
				// crcw = cross edge graph size
				// cinitial = contribution of one node to another (different components) 
				// crank = Pagerank
				// cedges = edge list
				// coutdeg = outdegree of the nodes
				// corder = level by level topological component ordering
				// ctemp = prefix sum of cmemsz
				// ctempg = prefix sum of crcw
				kerneltest1<<<blockB,threadB>>>(cn, cend, cmemsz, cmembers, crcw, ccheck_initial, crank,
								cedges, coutdeg, corder, ctemp, ctempg);

				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;

				// cudaMemcpy(initial, cinitial, n*sizeof(double), cudaMemcpyDeviceToHost);
				cudaFree(cn);
			}

			if(pivot < par[i+1]){
				cudaMemcpy(cstart, &pivot, sizeof(long), cudaMemcpyHostToDevice);
				cudaMemcpy(cend, &par[i+1], sizeof(long), cudaMemcpyHostToDevice);
				// cudaMemcpy(cinitial, initial, n*sizeof(double), cudaMemcpyHostToDevice);
				cudaMemcpy(crank, rank, n*sizeof(double), cudaMemcpyHostToDevice);

				dim3 threadB(8,8,16);
				dim3 blockB(8,8,16);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);
				
				// cstart = pivot
				// cend = par[i+1]
				// cmemsz = member size
				// cmembers = list of all members
				// crcw = cross edge graph size
				// cinitial = contribution of one node to another (different components) 
				// crank = Pagerank
				// cedges = edge list
				// coutdeg = outdegree of the nodes
				// corder = level by level topological component ordering
				// ctemp = prefix sum of cmemsz
				// ctempg = prefix sum of crcw
				kerneltest<<<blockB,threadB>>>(cstart, cend, cmemsz, cmembers, crcw, ccheck_initial, crank,
								cedges, coutdeg, corder, ctemp, ctempg);

				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
				// cudaMemcpy(initial, cinitial, n*sizeof(double), cudaMemcpyDeviceToHost);
			}
			cudaMemcpy(check_initial, ccheck_initial, check_size * sizeof(double), cudaMemcpyDeviceToHost);
			for(j=par[i];j<par[i+1];j++){
				for(long kk=0;kk<members[order[j]].size();kk++){
					long node = members[order[j]][kk];
					for(long ll=0;ll<rcwgraph[node].size();ll++){
						initial[node] += check_initial[tempg[node]+ll];
					}
				}
			}
				
			for(j=par[i];j<pivot;j++){
				total += computeparallelc(rcgraph,members[order[j]].size(),outdeg,members[order[j]],rank,initial,levelz,redir,powers, n);
			}
			for(j=pivot;j<par[i+1];j++){
				computerankc(rcgraph,members[order[j]].size(),outdeg,members[order[j]],rank,initial, levelz, redir, powers);
			}
		}
	}

	if(optident==0 && optchain==1 && optdead==1)
	{
		vector < long > par;
		par.push_back(0);
		for(i=0;i<com;i++){
			long j=i;
			while(j<com && nvisit[order[j]]==nvisit[order[i]]){
				j++;
			}
			par.push_back(j);
			i=j-1;
		}

		for(i=0;i<n;i++){
			members[component[i]].push_back(i);
		}

		double *initial = (double *)malloc(n*sizeof(double));
		memset(initial,0,n*sizeof(double));

		long *memsz = (long *)malloc(com*sizeof(long));
		long *temp = (long *)malloc(com*sizeof(long));
		long szz=0;
		for(long i1=0;i1<com;i1++){
			memsz[i1]=members[order[i1]].size();
			szz+=members[order[i1]].size();
		}
		for(long i1=0;i1<com;i1++){
			if(i1) temp[i1]=temp[i1-1]+memsz[i1-1];
			else temp[i1]=0;
		}
		long kk=0;
		long *mem = (long *)malloc(szz*sizeof(long));
		for(long i1=0;i1<com;i1++){
			for(long c:members[order[i1]]){
				mem[kk++]=c;
			}
		}
		
		cudaMalloc((void**)&cmembers, szz*sizeof(long));
		
		cudaMemcpy(cmemsz, memsz, com*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(ctemp, temp, com*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(cmembers, mem, szz*sizeof(long), cudaMemcpyHostToDevice);
		
		long w;
		long thresh=_MAX;

		for(i=0;i<par.size()-1;i++){
			long pivot=par[i];
			for(w=par[i];w<par[i+1];w++)
			{
				long sum=0;
				for(j=0;j<members[order[w]].size();j++)
					sum=sum+rgraph[members[order[w]][j]].size();
				if(sum>thresh)
				{
					long temp=order[pivot];
					order[pivot]=order[w];
					order[w]=temp;
					pivot++;
				}
			}
			for(w=par[i];w<pivot;w++)
			{
				long *cn;
				cudaMalloc((void**)&cn, sizeof(long));
				cudaMemcpy(cn, &w, sizeof(long), cudaMemcpyHostToDevice);

				dim3 threadB(32,32);
				dim3 blockB(32,32);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);
				
				// cn = w
				// cmemsz = member size
				// cmembers = list of all members
				// crcw = cross edge graph size
				// cinitial = contribution of one node to another (different components) 
				// crank = Pagerank
				// cedges = edge list
				// coutdeg = outdegree of the nodes
				// corder = level by level topological component ordering
				// ctemp = prefix sum of cmemsz
				// ctempg = prefix sum of crcw
				kerneltest1<<<blockB,threadB>>>(cn, cend, cmemsz, cmembers, crcw, ccheck_initial, crank,
								cedges, coutdeg, corder, ctemp, ctempg);

				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;

				// cudaMemcpy(initial, cinitial, n*sizeof(double), cudaMemcpyDeviceToHost);
				cudaFree(cn);
			}

			if(pivot < par[i+1]){
				cudaMemcpy(cstart, &pivot, sizeof(long), cudaMemcpyHostToDevice);
				cudaMemcpy(cend, &par[i+1], sizeof(long), cudaMemcpyHostToDevice);
				// cudaMemcpy(cinitial, initial, n*sizeof(double), cudaMemcpyHostToDevice);
				cudaMemcpy(crank, rank, n*sizeof(double), cudaMemcpyHostToDevice);

				dim3 threadB(8,8,16);
				dim3 blockB(8,8,16);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);
				
				// cstart = pivot
				// cend = par[i+1]
				// cmemsz = member size
				// cmembers = list of all members
				// crcw = cross edge graph size
				// cinitial = contribution of one node to another (different components) 
				// crank = Pagerank
				// cedges = edge list
				// coutdeg = outdegree of the nodes
				// corder = level by level topological component ordering
				// ctemp = prefix sum of cmemsz
				// ctempg = prefix sum of crcw
				kerneltest<<<blockB,threadB>>>(cstart, cend, cmemsz, cmembers, crcw, ccheck_initial, crank,
								cedges, coutdeg, corder, ctemp, ctempg);

				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
				// cudaMemcpy(initial, cinitial, n*sizeof(double), cudaMemcpyDeviceToHost);
			}
			cudaMemcpy(check_initial, ccheck_initial, check_size * sizeof(double), cudaMemcpyDeviceToHost);
			for(j=par[i];j<par[i+1];j++){
				for(long kk=0;kk<members[order[j]].size();kk++){
					long node = members[order[j]][kk];
					for(long ll=0;ll<rcwgraph[node].size();ll++){
						initial[node] += check_initial[tempg[node]+ll];
					}
				}
			}
			
			for(j=par[i];j<pivot;j++){
				total += computeparalleldc(rcgraph,members[order[j]].size(),outdeg,members[order[j]],rank,initial,levelz,redir,powers, n);
			}
			for(j=pivot;j<par[i+1];j++){
				computerankdc(rcgraph,members[order[j]].size(),outdeg,members[order[j]],rank,initial, levelz, redir, powers);
			}
		}
	}
	
	if(optident==1 && optchain==1 && optdead==0)
	{
		long parent[n];
		vector < vector < long > > left(com);

		for(i=0;i<n;i++){
			parent[i]=i;
		}

		vector < vector <  pair  <  pair < long , long > , long >  > > hvalues(n);

		for(i=0;i<n;i++){
			if(rgraph[i].size()!=1 && rgraph[i].size()!=2) 
				continue;
			if(rgraph[i].size()==1){
				hvalues[(rgraph[i][0])%n].push_back(make_pair(make_pair(rgraph[i][0],component[i]),i));
			}
			else{
				long val=max(rgraph[i][1]+1,rgraph[i][0]+1)*(long)(n+1)+min(rgraph[i][0]+1,rgraph[i][1]+1);
				hvalues[(val)%(long)n ].push_back(make_pair(make_pair(val,component[i]),i));
			}
		}

		for(i=0;i<n;i++){
			sort(hvalues[i].begin(),hvalues[i].end());
		}

		for(long k=0;k<n;k++){
			for(i=0;i<hvalues[k].size();i++){
				for(j=i;j<hvalues[k].size() && hvalues[k][j].first==hvalues[k][i].first ;j++){
					parent[hvalues[k][j].second]=hvalues[k][i].second;
				}
				i=j-1;
			}
		}

		hvalues.clear();
		long noo=0;
		for(i=0;i<n;i++){
			if(parent[i]==i) 
			{
				members[component[i]].push_back(i);
			}
			else
			{
				left[component[i]].push_back(i);
				noo++;
			}
		}

		vector < long > par;
		par.push_back(0);
		for(i=0;i<com;i++){
			long j=i;
			while(j<com && nvisit[order[j]]==nvisit[order[i]]){
				j++;
			}
			par.push_back(j);
			i=j-1;
		}
		
		double *initial = (double *)malloc(n*sizeof(double));
		memset(initial,0,n*sizeof(double));

		long *memsz = (long *)malloc(com*sizeof(long));
		long *temp = (long *)malloc(com*sizeof(long));
		long szz=0;
		for(long i1=0;i1<com;i1++){
			memsz[i1]=members[order[i1]].size();
			szz+=members[order[i1]].size();
		}
		for(long i1=0;i1<com;i1++){
			if(i1) temp[i1]=temp[i1-1]+memsz[i1-1];
			else temp[i1]=0;
		}
		long kk=0;
		long *mem = (long *)malloc(szz*sizeof(long));
		for(long i1=0;i1<com;i1++){
			for(long c:members[order[i1]]){
				mem[kk++]=c;
			}
		}

		cudaMalloc((void**)&cmembers, szz*sizeof(long));
		
		cudaMemcpy(cmemsz, memsz, com*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(ctemp, temp, com*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(cmembers, mem, szz*sizeof(long), cudaMemcpyHostToDevice);
		
		long w;
		long thresh=_MAX;

		for(i=0;i<par.size()-1;i++){
			long pivot=par[i];
			for(w=par[i];w<par[i+1];w++)
			{
				long sum=0;
				for(j=0;j<members[order[w]].size();j++)
					sum=sum+rgraph[members[order[w]][j]].size();
				if(sum>thresh)
				{
					long temp=order[pivot];
					order[pivot]=order[w];
					order[w]=temp;
					pivot++;
				}
			}
			for(w=par[i];w<pivot;w++)
			{
				long *cn;
				cudaMalloc((void**)&cn, sizeof(long));
				cudaMemcpy(cn, &w, sizeof(long), cudaMemcpyHostToDevice);

				dim3 threadB(32,32);
				dim3 blockB(32,32);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);
				
				// cn = w
				// cmemsz = member size
				// cmembers = list of all members
				// crcw = cross edge graph size
				// cinitial = contribution of one node to another (different components) 
				// crank = Pagerank
				// cedges = edge list
				// coutdeg = outdegree of the nodes
				// corder = level by level topological component ordering
				// ctemp = prefix sum of cmemsz
				// ctempg = prefix sum of crcw
				kerneltest1<<<blockB,threadB>>>(cn, cend, cmemsz, cmembers, crcw, ccheck_initial, crank,
								cedges, coutdeg, corder, ctemp, ctempg);

				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;

				// cudaMemcpy(initial, cinitial, n*sizeof(double), cudaMemcpyDeviceToHost);
				cudaFree(cn);
			}

			if(pivot < par[i+1]){
				cudaMemcpy(cstart, &pivot, sizeof(long), cudaMemcpyHostToDevice);
				cudaMemcpy(cend, &par[i+1], sizeof(long), cudaMemcpyHostToDevice);
				// cudaMemcpy(cinitial, initial, n*sizeof(double), cudaMemcpyHostToDevice);
				cudaMemcpy(crank, rank, n*sizeof(double), cudaMemcpyHostToDevice);

				dim3 threadB(8,8,16);
				dim3 blockB(8,8,16);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);
				
				// cstart = pivot
				// cend = par[i+1]
				// cmemsz = member size
				// cmembers = list of all members
				// crcw = cross edge graph size
				// cinitial = contribution of one node to another (different components) 
				// crank = Pagerank
				// cedges = edge list
				// coutdeg = outdegree of the nodes
				// corder = level by level topological component ordering
				// ctemp = prefix sum of cmemsz
				// ctempg = prefix sum of crcw
				kerneltest<<<blockB,threadB>>>(cstart, cend, cmemsz, cmembers, crcw, ccheck_initial, crank,
								cedges, coutdeg, corder, ctemp, ctempg);

				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
				// cudaMemcpy(initial, cinitial, n*sizeof(double), cudaMemcpyDeviceToHost);
			}

			for(j=par[i];j<pivot;j++){
				total += computeparallelic(rcgraph,parent,left[order[j]],members[order[j]].size(),outdeg,members[order[j]],rank,initial,levelz,redir,powers, n);
			}
			for(j=pivot;j<par[i+1];j++){
				computerankic(rcgraph, parent, left[order[j]],members[order[j]].size(),outdeg,members[order[j]],rank,initial, levelz, redir, powers);
			}
		}
	}

	if(optident==1 && optchain==1 && optdead==1)
	{
		long *parent = (long *)malloc(n*sizeof(long));
		vector < vector < long > > left(com);
		for(i=0;i<n;i++){
			parent[i]=i;
		}
		vector < vector <  pair  <  pair < long , long > , long >  > > hvalues(n);
		
		for(i=0;i<n;i++){
			if(rgraph[i].size()!=1 && rgraph[i].size()!=2) continue;
			if(rgraph[i].size()==1){
				hvalues[(rgraph[i][0])%n].push_back(make_pair(make_pair(rgraph[i][0],component[i]),i));
			}
			else{
				long val=max(rgraph[i][1]+1,rgraph[i][0]+1)*(long)(n+1)+min(rgraph[i][0]+1,rgraph[i][1]+1);
				hvalues[(val)%(long)n ].push_back(make_pair(make_pair(val,component[i]),i));
			}
		}
	
		for(i=0;i<n;i++){
			sort(hvalues[i].begin(),hvalues[i].end());
		}
		
		for(long k=0;k<n;k++){
			for(i=0;i<hvalues[k].size();i++){
				for(j=i;j<hvalues[k].size() && hvalues[k][j].first==hvalues[k][i].first ;j++){
					parent[hvalues[k][j].second]=hvalues[k][i].second;
				}
				i=j-1;
			}
		}
		hvalues.clear();

		long noo=0;
		for(i=0;i<n;i++){
			if(parent[i]==i){
				members[component[i]].push_back(i);
			}
			else{
				left[component[i]].push_back(i);
				noo++;
			}
		}
	
		vector < long > par;
		par.push_back(0);
		for(i=0;i<com;i++){
			long j=i;
			while(j<com && nvisit[order[j]]==nvisit[order[i]]){
				j++;
			}
			par.push_back(j);
			i=j-1;
		}
		
		double *initial = (double *)malloc(n*sizeof(double));
		memset(initial,0,n*sizeof(double));

		long *memsz = (long *)malloc(com*sizeof(long));
		long *temp = (long *)malloc(com*sizeof(long));
		long szz=0;
		for(long i1=0;i1<com;i1++){
			memsz[i1]=members[order[i1]].size();
			szz+=members[order[i1]].size();
		}
		for(long i1=0;i1<com;i1++){
			if(i1) temp[i1]=temp[i1-1]+memsz[i1-1];
			else temp[i1]=0;
		}
		long kk=0;
		long *mem = (long *)malloc(szz*sizeof(long));
		for(long i1=0;i1<com;i1++){
			for(long c:members[order[i1]]){
				mem[kk++]=c;
			}
		}
		
		cudaMalloc((void**)&cmembers, szz*sizeof(long));
		
		cudaMemcpy(cmemsz, memsz, com*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(ctemp, temp, com*sizeof(long), cudaMemcpyHostToDevice);
		cudaMemcpy(cmembers, mem, szz*sizeof(long), cudaMemcpyHostToDevice);
		
		long w;
		long thresh=_MAX;

		for(i=0;i<par.size()-1;i++){
			long pivot=par[i];
			for(w=par[i];w<par[i+1];w++)
			{
				long sum=0;
				for(j=0;j<members[order[w]].size();j++)
					sum=sum+rgraph[members[order[w]][j]].size();
				if(sum>thresh)
				{
					long temp=order[pivot];
					order[pivot]=order[w];
					order[w]=temp;
					pivot++;
				}
			}
			for(w=par[i];w<pivot;w++)
			{
				long *cn;
				cudaMalloc((void**)&cn, sizeof(long));
				cudaMemcpy(cn, &w, sizeof(long), cudaMemcpyHostToDevice);

				dim3 threadB(32,32);
				dim3 blockB(32,32);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);
				
				// cn = w
				// cmemsz = member size
				// cmembers = list of all members
				// crcw = cross edge graph size
				// cinitial = contribution of one node to another (different components) 
				// crank = Pagerank
				// cedges = edge list
				// coutdeg = outdegree of the nodes
				// corder = level by level topological component ordering
				// ctemp = prefix sum of cmemsz
				// ctempg = prefix sum of crcw
				kerneltest1<<<blockB,threadB>>>(cn, cend, cmemsz, cmembers, crcw, ccheck_initial, crank,
								cedges, coutdeg, corder, ctemp, ctempg);

				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;

				// cudaMemcpy(initial, cinitial, n*sizeof(double), cudaMemcpyDeviceToHost);
				cudaFree(cn);
			}

			if(pivot < par[i+1]){
				cudaMemcpy(cstart, &pivot, sizeof(long), cudaMemcpyHostToDevice);
				cudaMemcpy(cend, &par[i+1], sizeof(long), cudaMemcpyHostToDevice);
				// cudaMemcpy(cinitial, initial, n*sizeof(double), cudaMemcpyHostToDevice);
				cudaMemcpy(crank, rank, n*sizeof(double), cudaMemcpyHostToDevice);

				dim3 threadB(8,8,16);
				dim3 blockB(8,8,16);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);
				
				// cstart = pivot
				// cend = par[i+1]
				// cmemsz = member size
				// cmembers = list of all members
				// crcw = cross edge graph size
				// cinitial = contribution of one node to another (different components) 
				// crank = Pagerank
				// cedges = edge list
				// coutdeg = outdegree of the nodes
				// corder = level by level topological component ordering
				// ctemp = prefix sum of cmemsz
				// ctempg = prefix sum of crcw
				kerneltest<<<blockB,threadB>>>(cstart, cend, cmemsz, cmembers, crcw, ccheck_initial, crank,
								cedges, coutdeg, corder, ctemp, ctempg);

				cudaDeviceSynchronize();

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float elapsedTime;
				cudaEventElapsedTime(&elapsedTime, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
				total += elapsedTime;
				// cudaMemcpy(initial, cinitial, n*sizeof(double), cudaMemcpyDeviceToHost);
			}
	
			for(j=par[i];j<pivot;j++){
				total += computeparallelidc(rcgraph,parent,left[order[j]],members[order[j]].size(),outdeg,members[order[j]],rank,initial,levelz,redir,powers, n);
			}
			for(j=pivot;j<par[i+1];j++){
				computerankidc(rcgraph, parent, left[order[j]],members[order[j]].size(),outdeg,members[order[j]],rank,initial, levelz, redir, powers);
			}
		}
	}

	double sum=0;
	for(i=0;i<n;i++){
		sum=sum+rank[i];
	}
	for(i=0;i<n;i++){
		rank[i]=rank[i]/sum;
	}
	for(i=0;i<n;i++){
		fout << rank[i] << "\n";
	}
	auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken: "
         << duration.count() / 1000000.0 << "seconds" << "\n";

    cout << "kernel time: " << total << "\n\n\n";
    cudaFree(cstart);
    cudaFree(cend);
    cudaFree(corder);
    cudaFree(cmemsz);
    cudaFree(ctemp);
    cudaFree(coutdeg);
    cudaFree(cmembers);
    cudaFree(ctempg);
    cudaFree(cedges);
    cudaFree(crcw);
    cudaFree(crank);
	return 0;
}