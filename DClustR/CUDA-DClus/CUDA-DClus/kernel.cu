
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <time.h>
#include <sys/timeb.h>
#include <map>
#include <queue>
#include <windows.h>

// CUDA-DClus.cpp : Defines the entry point for the console application.
//

using namespace std;

#define tBuffer 100000
#define threadPerBlocksV 16
#define threadPerBlocksH 16
#define CENTER 1
#define SATELLITE 0


struct ConnectedComponent
{
	int** _adjacentList;
	int* _vertices;	
	int _countVertices;	
	bool _active;
	char* _path;
};

struct Node
{	
	int _pos;
	int _type;
	int _connectedComp;
	bool _analyzed;
	bool _proccessed;	
	bool _seek;
	bool _active;
	bool _weak;
	int* _linked;
	int _countLinked;
	int _degreeCenter;	
	float _relGreaterVertex;
};

struct Graph
{	
	int _countNode;
	int _countCennectedComp;
	Node* _nodes;	
	int* _countAdjPerNode;
	float* _aproxIntraSim;
	float* _relevance;
};

Graph *_graph;
int** _wordPosition = 0;
float** _frecuency = 0;


struct Document
{	
	int DocSize;
	float Norm;	
	int _index;
};

ConnectedComponent* connectedComponent = 0;
Document* _document = 0;
char** ID;
char** terminos = 0;
int cTERMINOS = 0;
vector<int> M;	
int option;
char* pathGraph = 0;
char* pathDocument = 0;
char* pathClusters = 0;
char* pathCC = 0;
vector<int> V_s;
vector<int> C_0;
int init = 0;
int _size = 0;

	 
__global__ void BuildSimilarityNode(Document* _devDoc, float* _frecD1, float* frec, int* wordPos, int indexDoc, float* result)
{
	int tid = threadIdx.x + blockDim.x* blockIdx.x;
	Document d = _devDoc[indexDoc + 1];		
	__shared__ float _frecPerDoc[threadPerBlocksV][threadPerBlocksH];

	Document d2;
	if(tid <= indexDoc)
	{		
		d2 = _devDoc[tid];
		int tidY = threadIdx.y;		
		float resultTemp = 0;
		while(d2._index + tidY < d2._index + d2.DocSize)
		{
			resultTemp += frec[d2._index + tidY]* _frecD1[wordPos[tidY + d2._index]];
			tidY += threadPerBlocksV;
		}
		_frecPerDoc[threadIdx.y][threadIdx.x] = resultTemp;
	}
	__syncthreads();
	
	int i = threadPerBlocksV/2;
	while(i != 0)
	{
		if(threadIdx.y < i)
			_frecPerDoc[threadIdx.y][threadIdx.x] += _frecPerDoc[threadIdx.y + i][threadIdx.x];
		__syncthreads();
		i/=2;
	}
	if(threadIdx.y == 0 && tid <= indexDoc)
		result[tid] = (d.Norm > 0 && d2.Norm > 0) ? _frecPerDoc[0][threadIdx.x] / (d.Norm*d2.Norm) : 0.0;			
}

__global__ void Initialize(Document* _devDoc, int indexDoc ,int* _devWordPosition, float* _devFrequency ,float* _devFreqDoc)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	Document d = _devDoc[indexDoc];
	while (tid < d.DocSize)
	{
		int pos = _devWordPosition[d._index + tid];
		float result = _devFrequency[d._index + tid];
		_devFreqDoc[pos] = result;
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void Finalize(float* _devFreqDoc, int size)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;	
	while (tid < size)
	{		
		_devFreqDoc[tid] = 0;
		tid += blockDim.x * gridDim.x;
	}
}

void Read(char* pathAdd, double threshold, int countAdd)
{	
	int cObjetos = 0;	
	FILE* f;
	char* buffer = 0;
	char* temp = 0;
	char* FIN = 0;

	if(option != 0)
	{
		cObjetos = _graph->_countNode;
		init = cObjetos;
	}

	if(countAdd > 0)
	{
		f = fopen(pathAdd, "r");		
		buffer = (char *)malloc(sizeof(char)*tBuffer);

		temp = (char *)malloc(sizeof(char)* 30);
		int i = 0, j = 0;
		FIN = fgets(buffer, tBuffer, f);

		while (FIN != 0)
		{
			if (cObjetos == 0)
			{
				_document = (Document*)malloc(sizeof(Document));
				_wordPosition = (int**)malloc(sizeof(int*));
				_frecuency = (float**)malloc(sizeof(float*));
				ID = (char**)malloc(sizeof(char*));
			}
			else
			{
				_document = (Document*)realloc(_document, sizeof(Document)*(cObjetos + 1));
				_wordPosition = (int**)realloc(_wordPosition, sizeof(int*)*(cObjetos + 1));
				_frecuency = (float**)realloc(_frecuency, sizeof(float*)*(cObjetos + 1));
				ID = (char**)realloc(ID, sizeof(char*)*(cObjetos + 1));
			}		

			ID[cObjetos] = 0;
			ID[cObjetos] = (char*)malloc(sizeof(char)* 100);
			
			i = 0; j = 0;
			while (buffer[i] != '|')
			{
				ID[cObjetos][j] = buffer[i++];
				j++;
			}

			i++;

			ID[cObjetos][j] = '\0';
			ID[cObjetos] = (char*)realloc(ID[cObjetos], sizeof(char)*(j + 1));
			_document[cObjetos].DocSize = 0;
			_document[cObjetos].Norm = 0;		
			

			while (buffer[i] != '\r' && buffer[i] != '\n')
			{
				if (cTERMINOS == 0) terminos = (char**)malloc(sizeof(char*));
				else terminos = (char**)realloc(terminos, sizeof(char*)*(cTERMINOS + 1));

				terminos[cTERMINOS] = 0;
				terminos[cTERMINOS] = (char*)malloc(sizeof(char)* 100);

				j = 0;
				while (buffer[i] != ':')
				{
					terminos[cTERMINOS][j] = buffer[i++];
					j++;
				}

				i++;
				terminos[cTERMINOS][j] = '\0';
				terminos[cTERMINOS] = (char*)realloc(terminos[cTERMINOS], sizeof(char)*(j + 1));

				j = 0;
				while (buffer[i] != '#')
				{
					temp[j] = buffer[i++];
					j++;
				}

				i++;
				temp[j] = '\0';

				int index = -1;

				for (int k = 0; k < cTERMINOS && index == -1; k++)
				{
					if (strcmp(terminos[k], terminos[cTERMINOS]) == 0)
						index = k;
				}

				if (index == -1)
				{
					index = cTERMINOS++;
				}
				else
				{
					free(terminos[cTERMINOS]);
					terminos[cTERMINOS] = 0;
					terminos = (char**)realloc(terminos, sizeof(char*)*cTERMINOS);
				}				
									

				if(_document[cObjetos].DocSize == 0)
				{
					_wordPosition[cObjetos] = (int*)malloc(sizeof(int));
					_frecuency[cObjetos] = (float*)malloc(sizeof(float));
				}
				else
				{
					_wordPosition[cObjetos] = (int*)realloc(_wordPosition[cObjetos], sizeof(int)*(_document[cObjetos].DocSize + 1));
					_frecuency[cObjetos] = (float*)realloc(_frecuency[cObjetos], sizeof(float)*(_document[cObjetos].DocSize + 1));
				}

				_wordPosition[cObjetos][_document[cObjetos].DocSize] = index;
				_frecuency[cObjetos][_document[cObjetos].DocSize] = atof(temp);

				_document[cObjetos].Norm += powf(_frecuency[cObjetos][_document[cObjetos].DocSize], 2);
				_document[cObjetos].DocSize++;
				_size++;
			}

			_document[cObjetos].Norm = sqrtf(_document[cObjetos].Norm);

			//Creo los nodos del grafo
			if (cObjetos == 0)
			{
				_graph = (Graph*)malloc(sizeof(Graph));
				_graph->_countNode = 0;
				_graph->_aproxIntraSim = (float*)calloc(1, sizeof(float));
				_graph->_countAdjPerNode = (int*)calloc(1, sizeof(int));				
				_graph->_relevance = 0;
				_graph->_countCennectedComp = 0;
				_graph->_nodes = (Node*)malloc(sizeof(Node));
			}
			else
			{
				_graph->_nodes = (Node*)realloc(_graph->_nodes, sizeof(Node)*(_graph->_countNode + 1));				
				_graph->_aproxIntraSim = (float*)realloc(_graph->_aproxIntraSim, (_graph->_countNode + 1)*sizeof(float));
				_graph->_aproxIntraSim[_graph->_countNode] = 0;
				_graph->_countAdjPerNode = (int*)realloc(_graph->_countAdjPerNode,(_graph->_countNode + 1)*sizeof(float));
				_graph->_countAdjPerNode[_graph->_countNode] = 0;				
			}			
			
			_graph->_nodes[_graph->_countNode]._active = false;
			_graph->_nodes[_graph->_countNode]._type = SATELLITE;			
			_graph->_nodes[_graph->_countNode]._weak = false;		
			_graph->_nodes[_graph->_countNode]._seek = false;				
			_graph->_nodes[_graph->_countNode]._proccessed = false;
			_graph->_nodes[_graph->_countNode]._analyzed = false;
			_graph->_nodes[_graph->_countNode]._countLinked = 0;
			_graph->_nodes[_graph->_countNode]._connectedComp = -1;
			_graph->_nodes[_graph->_countNode]._degreeCenter = 0;			

			M.push_back(_graph->_countNode);
			_graph->_countNode++;

			cObjetos++;

			FIN = fgets(buffer, tBuffer, f);
		}
		fclose(f);
		f = 0;

		free(buffer);
		buffer = 0;

		free(temp);
		temp = 0;
	}	
}

vector<int> GetConnectedComponent(int node, bool* mark)
{		
	queue<int> _queue;
	_queue.push(node);	
	vector<int> result;
	while (_queue.size() > 0)
	{
		int temp = _queue.front();
		_graph->_relevance[temp] = 0;
		_graph->_nodes[temp]._proccessed = false;
		mark[temp] = true;
		_queue.pop();
		result.push_back(temp);
		float density = 0;
		float compactness = 0;
		for (int i = 0; i < _graph->_countAdjPerNode[temp]; i++)
		{
			if(_graph->_nodes[temp]._seek && _graph->_nodes[temp]._type == SATELLITE)
				_graph->_nodes[connectedComponent[_graph->_nodes[temp]._connectedComp]._adjacentList[_graph->_nodes[temp]._pos][i]]._degreeCenter++;
			
			if (_graph->_countAdjPerNode[connectedComponent[_graph->_nodes[temp]._connectedComp]._adjacentList[_graph->_nodes[temp]._pos][i]] <= _graph->_countAdjPerNode[temp])
				density++;//_graph->_relevance[temp]++;
			if ((_graph->_aproxIntraSim[connectedComponent[_graph->_nodes[temp]._connectedComp]._adjacentList[_graph->_nodes[temp]._pos][i]]*1.0)/_graph->_countAdjPerNode[connectedComponent[_graph->_nodes[temp]._connectedComp]._adjacentList[_graph->_nodes[temp]._pos][i]] <= (_graph->_aproxIntraSim[temp]*1.0)/_graph->_countAdjPerNode[temp])
				compactness++;//_graph->_relevance[temp]++;

			if (!mark[connectedComponent[_graph->_nodes[temp]._connectedComp]._adjacentList[_graph->_nodes[temp]._pos][i]])
			{
				mark[connectedComponent[_graph->_nodes[temp]._connectedComp]._adjacentList[_graph->_nodes[temp]._pos][i]] = true;
				_queue.push(connectedComponent[_graph->_nodes[temp]._connectedComp]._adjacentList[_graph->_nodes[temp]._pos][i]);
			}
		}	

		density = (density*1.0) / _graph->_countAdjPerNode[temp];
		compactness = (compactness*1.0) / _graph->_countAdjPerNode[temp];
		_graph->_relevance[temp] = (density + compactness)/2.0; //(_graph->_countAdjPerNode[temp] > 0)? _graph->_relevance[temp]/(2.0*_graph->_countAdjPerNode[temp]): 0.0;										

		if(_graph->_nodes[temp]._seek)
		{
			if(_graph->_nodes[temp]._type == SATELLITE)
			{
				_graph->_nodes[temp]._type = CENTER;
				_graph->_nodes[temp]._degreeCenter++;
			}
			_graph->_nodes[temp]._seek = false;
			if(_graph->_nodes[temp]._countLinked > 0)
				free(_graph->_nodes[temp]._linked);
			_graph->_nodes[temp]._countLinked = 0;
		}

		//Construyo los conjuntos V_s y C'(C_0)
		if(_graph->_nodes[temp]._type == CENTER)
		{				
			_graph->_nodes[temp]._relGreaterVertex = _graph->_relevance[temp];			
			C_0.push_back(temp);
		}
		else 
		{			
			if(_graph->_relevance[temp] > 0)
				V_s.push_back(temp);
		}
	}
	return result;
}

void BuildAdjacentList(float beta)
{	
	//Construir vectores de posicion, frecuencia y actualizar los index de los documentos
	int* wordPosition = (int*)malloc(sizeof(int)*_size);
	float* frecuency = (float*)malloc(sizeof(float)*_size);
	int index = 0;
	for (int i = 0; i < _graph->_countNode; i++)
	{
		_document[i]._index = index;
		for (int j = 0; j < _document[i].DocSize; j++)
		{
			wordPosition[index] = _wordPosition[i][j];
			frecuency[index] = _frecuency[i][j];
			index++;
		}
	}

	int* _devWordPosition;
	float* _devFrecuency;
	float* _devSimilarityNode;
	Document* _devDoc;

	cudaError_t cudaStatus; //= cudaMemGetInfo(&freeM, &totalM);	
	//Copio para la memoria constante el array de documentos	
	cudaStatus = cudaMalloc((void**)&_devWordPosition, _size*sizeof(int));
	cudaStatus = cudaMalloc((void**)&_devFrecuency, _size* sizeof(float));	
	cudaStatus = cudaMalloc((void**)&_devDoc, _graph->_countNode* sizeof(Document));	

	//Creo la memoria necesaria en el GPU para el array de frecuencia y de la posicion de las palabras
	float* result = (float*)calloc(_graph->_countNode, sizeof(float));			
	//asigno memoria para el resultado
	cudaStatus = cudaMalloc((void**)&_devSimilarityNode, _graph->_countNode*sizeof(float));
	//copio el array de resultados para la memoria del dispositivo
	cudaStatus = cudaMemcpy(_devSimilarityNode, result, _graph->_countNode*sizeof(float), cudaMemcpyHostToDevice);

	//Copio los valores de los array para el GPU
	cudaStatus = cudaMemcpy(_devWordPosition, wordPosition, _size*sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(_devFrecuency, frecuency, _size*sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(_devDoc, _document, _graph->_countNode *sizeof(Document), cudaMemcpyHostToDevice);
	float* _devFrec;

	float* _frecD1 = (float*)calloc(cTERMINOS, sizeof(float));
	//asigno memoria para el array de terminos 
	cudaStatus = cudaMalloc((void**)&_devFrec, cTERMINOS*sizeof(float));			
	//copio la frecuencia del nodo i para la memoria del dispositivo
	cudaStatus = cudaMemcpy(_devFrec, _frecD1, cTERMINOS*sizeof(float), cudaMemcpyHostToDevice);
	free(_frecD1);	

	
	for(int i = _graph->_countNode - 1; i >= init; i--)
	{
		if(_graph->_nodes[i]._connectedComp < 0)
		{
			//CONSTRUIR LA COMPONENTES CONEXA DEL OBJETO I	
			if(_graph->_countCennectedComp == 0)
				connectedComponent = (ConnectedComponent*)malloc(sizeof(ConnectedComponent));
			else
				connectedComponent = (ConnectedComponent*)realloc(connectedComponent, (_graph->_countCennectedComp + 1)*sizeof(ConnectedComponent));
			connectedComponent[_graph->_countCennectedComp]._vertices = (int*)malloc(sizeof(int));
			connectedComponent[_graph->_countCennectedComp]._adjacentList = (int**)malloc(sizeof(int*));
			connectedComponent[_graph->_countCennectedComp]._vertices[0] = i;	
			//connectedComponent[_graph->_countCennectedComp]._countChildren = 0;
			connectedComponent[_graph->_countCennectedComp]._active = true;
			connectedComponent[_graph->_countCennectedComp]._path = (char*)malloc(sizeof(char)*100);
			int position = strlen(pathCC);
			strncpy(connectedComponent[_graph->_countCennectedComp]._path, pathCC, strlen(pathCC));
			char* tmp = (char*)malloc(sizeof(char)*10);
			itoa(_graph->_countCennectedComp, tmp, 10);			
			strncpy(&(connectedComponent[_graph->_countCennectedComp]._path[position]), tmp, strlen(tmp));
			position += strlen(tmp);
			strncpy(&(connectedComponent[_graph->_countCennectedComp]._path[position]), "_component.dat", strlen("_component.dat"));
			position += strlen("_component.dat");
			connectedComponent[_graph->_countCennectedComp]._path[position] = '\0';
			free(tmp);
			_graph->_nodes[i]._pos = 0;
			_graph->_nodes[i]._connectedComp = _graph->_countCennectedComp;
			connectedComponent[_graph->_countCennectedComp]._countVertices = 1;
			_graph->_countCennectedComp++;			
		}

		if(i > 0)
		{			
			Document d = _document[i];
			int grids = (d.DocSize + 256)/256;
			//Inicializo el vector de terminos
			Initialize<<<grids, 256>>>(_devDoc, i, _devWordPosition, _devFrecuency, _devFrec);

			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));			
			}

			//Ejecuto el Kernel
			dim3 blocks(threadPerBlocksH, threadPerBlocksV);
			grids = (i * 2 + (threadPerBlocksH - 1))/threadPerBlocksH;
			BuildSimilarityNode<<<grids, blocks>>>(_devDoc, _devFrec, _devFrecuency, _devWordPosition, i - 1, _devSimilarityNode);

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));			
			}
			//Copio el resultado para la memoria del CPU
			cudaStatus = cudaMemcpy(result, _devSimilarityNode, _graph->_countNode*sizeof(float), cudaMemcpyDeviceToHost);									

			grids = (cTERMINOS + 256)/256;
			Finalize<<<grids, 256>>>(_devFrec, cTERMINOS);

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));			
			}

			//Recorro el resultado y construyo la lista de adyacente del nodo i		
			for(int j = i - 1; j >= 0; j--)
			{	
				if(result[j] >= beta)
				{
					if(_graph->_nodes[j]._type == CENTER)
						_graph->_nodes[i]._degreeCenter++;
					if(_graph->_nodes[j]._connectedComp < 0)
					{
						connectedComponent[_graph->_nodes[i]._connectedComp]._adjacentList = (int**)realloc(connectedComponent[_graph->_nodes[i]._connectedComp]._adjacentList, sizeof(int*)*
							(connectedComponent[_graph->_nodes[i]._connectedComp]._countVertices + 1));
						_graph->_nodes[j]._pos = connectedComponent[_graph->_nodes[i]._connectedComp]._countVertices;	
						connectedComponent[_graph->_nodes[i]._connectedComp]._vertices = (int*)realloc(connectedComponent[_graph->_nodes[i]._connectedComp]._vertices, sizeof(int)*(connectedComponent[_graph->_nodes[i]._connectedComp]._countVertices + 1));
						connectedComponent[_graph->_nodes[i]._connectedComp]._vertices[connectedComponent[_graph->_nodes[i]._connectedComp]._countVertices] = j;
						_graph->_nodes[j]._connectedComp = _graph->_nodes[i]._connectedComp;
						connectedComponent[_graph->_nodes[i]._connectedComp]._countVertices++;
					}
					else
					{
						if(!connectedComponent[_graph->_nodes[j]._connectedComp]._active)
						{
							queue<int> queueCC;
							queueCC.push(_graph->_nodes[j]._connectedComp);
							connectedComponent[_graph->_nodes[j]._connectedComp]._active = true;
							//Cargar la componente conexa con todos sus hijos recursivamente
							while (queueCC.size() > 0)
							{
								int component = queueCC.front();
								queueCC.pop();

								FILE* file = fopen(connectedComponent[component]._path, "r");
								//Cargar la cantidad de vertices y la cantidad de hijos
								fscanf(file, "%d\n", &(connectedComponent[component]._countVertices));
								//Cargar los vertices que pertenecen a la componente conexa
								connectedComponent[component]._vertices = (int*)malloc(sizeof(int)*connectedComponent[component]._countVertices);
								for (int k = 0; k < connectedComponent[component]._countVertices; k++)
									fscanf(file, "%d\n", &(connectedComponent[component]._vertices[k]));							
								

								//Cargar la lista de adyacencia de la componente conexa
								connectedComponent[component]._adjacentList = (int**)malloc(sizeof(int*)*connectedComponent[component]._countVertices);
								for (int k = 0; k < connectedComponent[component]._countVertices; k++)
								{
									//Construir para cada nodo su lista de adyacentes
									if(_graph->_countAdjPerNode[connectedComponent[component]._vertices[k]] > 0)
									{
										connectedComponent[component]._adjacentList[k] = (int*)malloc(sizeof(int)*_graph->_countAdjPerNode[connectedComponent[component]._vertices[k]]);
										for (int t = 0; t < _graph->_countAdjPerNode[connectedComponent[component]._vertices[k]]; t++)
										{
											fscanf(file, "%d\n", &(connectedComponent[component]._adjacentList[k][t]));
											if(!connectedComponent[_graph->_nodes[connectedComponent[component]._adjacentList[k][t]]._connectedComp]._active)
											{
												connectedComponent[_graph->_nodes[connectedComponent[component]._adjacentList[k][t]]._connectedComp]._active = true;
												queueCC.push(_graph->_nodes[connectedComponent[component]._adjacentList[k][t]]._connectedComp);
											}
										}
									}
								}
								
								fclose(file);								
							}
						}
						
						
						/*if(_graph->_nodes[j]._connectedComp != _graph->_nodes[i]._connectedComp)
						{
							if(connectedComponent[_graph->_nodes[i]._connectedComp]._countChildren == 0)
								connectedComponent[_graph->_nodes[i]._connectedComp]._children = (int*)malloc(sizeof(int));
							else
								connectedComponent[_graph->_nodes[i]._connectedComp]._children = (int*)realloc(connectedComponent[_graph->_nodes[i]._connectedComp]._children, sizeof(int)*(
								connectedComponent[_graph->_nodes[i]._connectedComp]._countChildren + 1));
							connectedComponent[_graph->_nodes[i]._connectedComp]._children[connectedComponent[_graph->_nodes[i]._connectedComp]._countChildren] = _graph->_nodes[j]._connectedComp;
							connectedComponent[_graph->_nodes[i]._connectedComp]._countChildren++;

							if(connectedComponent[_graph->_nodes[j]._connectedComp]._countChildren == 0)
								connectedComponent[_graph->_nodes[j]._connectedComp]._children = (int*)malloc(sizeof(int));
							else
								connectedComponent[_graph->_nodes[j]._connectedComp]._children = (int*)realloc(connectedComponent[_graph->_nodes[j]._connectedComp]._children, sizeof(int)*(
								connectedComponent[_graph->_nodes[j]._connectedComp]._countChildren + 1));
							connectedComponent[_graph->_nodes[j]._connectedComp]._children[connectedComponent[_graph->_nodes[j]._connectedComp]._countChildren] = _graph->_nodes[i]._connectedComp;
							connectedComponent[_graph->_nodes[j]._connectedComp]._countChildren++;												
						}		*/			
					}

					//ACTUALIZAR LAS LISTAS DE ADYACENCIAS DE CADA OBJETO
					if(_graph->_countAdjPerNode[i] == 0)
						connectedComponent[_graph->_nodes[i]._connectedComp]._adjacentList[_graph->_nodes[i]._pos] = (int*)malloc(sizeof(int));
					else
						connectedComponent[_graph->_nodes[i]._connectedComp]._adjacentList[_graph->_nodes[i]._pos] = (int*)realloc(connectedComponent[_graph->_nodes[i]._connectedComp]._adjacentList[_graph->_nodes[i]._pos], sizeof(int)*(_graph->_countAdjPerNode[i] + 1));
					connectedComponent[_graph->_nodes[i]._connectedComp]._adjacentList[_graph->_nodes[i]._pos][_graph->_countAdjPerNode[i]] = j;
					_graph->_countAdjPerNode[i]++;

					if(_graph->_countAdjPerNode[j] == 0)
						connectedComponent[_graph->_nodes[j]._connectedComp]._adjacentList[_graph->_nodes[j]._pos] = (int*)malloc(sizeof(int));
					else
						connectedComponent[_graph->_nodes[j]._connectedComp]._adjacentList[_graph->_nodes[j]._pos] = (int*)realloc(connectedComponent[_graph->_nodes[j]._connectedComp]._adjacentList[_graph->_nodes[j]._pos], sizeof(int)*(_graph->_countAdjPerNode[j] + 1));
					connectedComponent[_graph->_nodes[j]._connectedComp]._adjacentList[_graph->_nodes[j]._pos][_graph->_countAdjPerNode[j]] = i;
					_graph->_countAdjPerNode[j]++;	
					

					_graph->_aproxIntraSim[i] += result[j];
					_graph->_aproxIntraSim[j] += result[j];				
				}
			}
		}
	}	

	cudaFree(_devFrec);			
	free(result);
	cudaFree(_devSimilarityNode);	
	cudaFree(_devDoc);
	cudaFree(_devFrecuency);
	cudaFree(_devWordPosition);
	free(wordPosition);
	free(frecuency);
}

void WriteClusters()
{
	FILE* f1;	
	fopen_s(&f1, pathClusters, "w");
	for (int i = 0; i < _graph->_countNode; i++)
	{
		if(!connectedComponent[_graph->_nodes[i]._connectedComp]._active)
		{
			queue<int> queueCC;
			queueCC.push(_graph->_nodes[i]._connectedComp);
			connectedComponent[_graph->_nodes[i]._connectedComp]._active = true;
			//Cargar la componente conexa con todos sus hijos recursivamente
			while (queueCC.size() > 0)
			{
				int component = queueCC.front();
				queueCC.pop();

				FILE* file = fopen(connectedComponent[component]._path, "r");
				//Cargar la cantidad de vertices y la cantidad de hijos
				fscanf(file, "%d\n", &(connectedComponent[component]._countVertices));
				//Cargar los vertices que pertenecen a la componente conexa
				connectedComponent[component]._vertices = (int*)malloc(sizeof(int)*connectedComponent[component]._countVertices);
				for (int k = 0; k < connectedComponent[component]._countVertices; k++)
					fscanf(file, "%d\n", &(connectedComponent[component]._vertices[k]));						
								

				//Cargar la lista de adyacencia de la componente conexa
				connectedComponent[component]._adjacentList = (int**)malloc(sizeof(int*)*connectedComponent[component]._countVertices);
				for (int k = 0; k < connectedComponent[component]._countVertices; k++)
				{
					//Construir para cada nodo su lista de adyacentes
					if(_graph->_countAdjPerNode[connectedComponent[component]._vertices[k]] > 0)
					{
						connectedComponent[component]._adjacentList[k] = (int*)malloc(sizeof(int)*_graph->_countAdjPerNode[connectedComponent[component]._vertices[k]]);
						for (int t = 0; t < _graph->_countAdjPerNode[connectedComponent[component]._vertices[k]]; t++)
						{
							fscanf(file, "%d\n", &(connectedComponent[component]._adjacentList[k][t]));
							if(!connectedComponent[_graph->_nodes[connectedComponent[component]._adjacentList[k][t]]._connectedComp]._active)
							{
								connectedComponent[_graph->_nodes[connectedComponent[component]._adjacentList[k][t]]._connectedComp]._active = true;
								queueCC.push(_graph->_nodes[connectedComponent[component]._adjacentList[k][t]]._connectedComp);
							}
						}
					}
				}
								
				fclose(file);								
			}
		}
		if (_graph->_nodes[i]._seek)
		{
			_graph->_nodes[i]._analyzed = false;
			fputs(ID[i], f1);
			for (int j = 0; j < _graph->_countAdjPerNode[i]; j++)
			{
				int pos = connectedComponent[_graph->_nodes[i]._connectedComp]._adjacentList[_graph->_nodes[i]._pos][j];
				fputs(", ", f1);
				fputs(ID[pos], f1);
				_graph->_nodes[pos]._analyzed = false;
			}
			for (int j = 0; j < _graph->_nodes[i]._countLinked; j++)
			{
				fputs(", ", f1);
				fputs(ID[_graph->_nodes[i]._linked[j]], f1);
				_graph->_nodes[_graph->_nodes[i]._linked[j]]._analyzed = false;
			}
			fputs("\n", f1);
		}
	}	
	fclose(f1);	
}

int ReadGraph()
{
	if(option == 0)
		return 0;
	FILE* file = fopen(pathGraph, "r");
	char* buffer = 0;
	char* temp = 0;
	char* FIN = 0;

	buffer = (char *)malloc(sizeof(char)*tBuffer);

	temp = (char *)malloc(sizeof(char)* 30);	

	_graph = (Graph*)malloc(sizeof(Graph));
	_graph->_countNode = 0;	
	_graph->_countCennectedComp = 0;

	//Leer la cantidad de nodos del grafo
	fscanf(file, "%i\n", &(_graph->_countNode));

	ID = (char**)malloc(sizeof(char*)*_graph->_countNode);

	//Leer los ID de los objetos
	for (int i = 0; i < _graph->_countNode; i++)
	{
		ID[i] = (char*)malloc(sizeof(char)*30);
		fgets(buffer, tBuffer, file);
		int j = 0;
		while (buffer[j] != '\n')
		{
			ID[i][j] = buffer[j];
			j++;
		}
		ID[i][j] = '\0';
	}

	_graph->_nodes = (Node*)malloc(sizeof(Node)*_graph->_countNode);	

	for (int i = 0; i < _graph->_countNode; i++)
	{		
		_graph->_nodes[i]._active = false;
		_graph->_nodes[i]._type = SATELLITE;			
		_graph->_nodes[i]._weak = false;		
		_graph->_nodes[i]._seek = false;			
		_graph->_nodes[i]._proccessed = false;
		_graph->_nodes[i]._analyzed = false;
		_graph->_nodes[i]._countLinked = 0;
		_graph->_nodes[i]._connectedComp = -1;
		_graph->_nodes[i]._degreeCenter = 0;	

		fgets(buffer, tBuffer, file);
		int j = 0, k = 0;		
		//Leer type del nodo i-esimo		
		while (buffer[j] != ' ')
		{
			temp[k] = buffer[j];
			k++;
			j++;
		}
		temp[k] = '\0';
		_graph->_nodes[i]._type = atoi(temp);
		j++;		

		//Leer el seek del nodo i-esimo
		k = 0;
		while (buffer[j] != ' ')
		{
			temp[k] = buffer[j];
			k++;
			j++;
		}
		temp[k] = '\0';
		_graph->_nodes[i]._seek = bool(atoi(temp));
		j++;

		//Leer el numero de Componente conexa a la que pertenece el nodo i-esimo
		k = 0;
		while (buffer[j] != ' ')
		{
			temp[k] = buffer[j];
			k++;
			j++;
		}
		temp[k] = '\0';
		_graph->_nodes[i]._connectedComp = atoi(temp);
		j++;

		//Leer la posicion del nodo i-esimo dentro de la componente conexa
		k = 0;
		while (buffer[j] != ' ')
		{
			temp[k] = buffer[j];
			k++;
			j++;
		}
		temp[k] = '\0';
		_graph->_nodes[i]._pos = atoi(temp);
		j++;

		//Leer el grado de agrupamiento del nodo i-esimo
		k = 0;
		while (buffer[j] != ' ')
		{
			temp[k] = buffer[j];
			k++;
			j++;
		}
		temp[k] = '\0';
		_graph->_nodes[i]._degreeCenter = atoi(temp);
		j++;

		//Leer la cantidad de linked del nodo i-esimo
		k = 0;
		while (buffer[j] != '\n')
		{
			temp[k] = buffer[j];
			k++;
			j++;
		}
		temp[k] = '\0';
		_graph->_nodes[i]._countLinked = atoi(temp);
				
		//Leer los linkeds del nodo i-esimo
		int t = 0;
		if(_graph->_nodes[i]._countLinked > 0)
		{
			//fgets(buffer, tBuffer, file);
			_graph->_nodes[i]._linked = (int*)malloc(sizeof(int)*_graph->_nodes[i]._countLinked);
			for (j = 0; j < _graph->_nodes[i]._countLinked; j++)
			{
				int tmp = 0;
				fscanf(file, "%i\n", &tmp);
				_graph->_nodes[i]._linked[j] = tmp;			
			}
		}
	}
	
	//LEER LOS VECTORES DE CANTIDAD DE ADYACENTES Y APROX_INTRA_SIM DE CADA VERTICE
	_graph->_countAdjPerNode = (int*)malloc(sizeof(int)*_graph->_countNode);
	_graph->_aproxIntraSim = (float*)malloc(sizeof(float)*_graph->_countNode);
	for (int i = 0; i < _graph->_countNode; i++)
	{
		fscanf(file, "%i\n", &(_graph->_countAdjPerNode[i]));
		fscanf(file, "%f\n", &(_graph->_aproxIntraSim[i]));
	}

	//CONSTRUIR LA LISTA DE COMPONENTES CONEXAS QUE REPRESENTA EL GRAFO
	//Leer la cantidad de componente conexas
	fscanf(file, "%i\n", &(_graph->_countCennectedComp));

	//Leer cada componente conexa del grafo
	connectedComponent = (ConnectedComponent*)malloc(sizeof(ConnectedComponent)*_graph->_countCennectedComp);

	for (int i = 0; i < _graph->_countCennectedComp; i++)
	{
		fgets(buffer, tBuffer, file);
		connectedComponent[i]._path = (char*)malloc(sizeof(char)*100);
		connectedComponent[i]._countVertices = 0;
		connectedComponent[i]._active = false;
		int j = 0;
		while (buffer[j] != '\n')
		{
			connectedComponent[i]._path[j] = buffer[j];
			j++;
		}
		connectedComponent[i]._path[j] = '\0';
	}
	

	fclose(file);
	free(buffer);
	free(temp);
	return 1;
}

void WriteGraph()
{	
	FILE* file = fopen(pathGraph, "w");

	//ALMACENAR LA CANTIDAD DE NODOS DEL GRAFO
	fprintf(file, "%i\n", _graph->_countNode);

	//ALMACENAR LOS ID DE LOS DOCUMENTOS
	for (int i = 0; i < _graph->_countNode; i++)
		fprintf(file, "%s\n", ID[i]);

	//ALMACENAR LOS NODOS DEL GRAFO
	for (int i = 0; i < _graph->_countNode; i++)
	{				
		free(ID[i]);
		//Almacenar el tipo del nodo i-esimo
		fprintf(file, "%i ", _graph->_nodes[i]._type);
		//Almacenar si es semilla el nodo i-esimo
		fprintf(file, "%i ", _graph->_nodes[i]._seek);
		//Almacenar el numero de componente conexa del nodo i-esimo
		fprintf(file, "%i ", _graph->_nodes[i]._connectedComp);
		//Almacenar la posicion del nodo i-esimo en su componente conexa
		fprintf(file, "%i ", _graph->_nodes[i]._pos);
		//Almacenar el grado de agrupamiento del nodo i-esimo
		fprintf(file, "%i ", _graph->_nodes[i]._degreeCenter);
		//Almacenar la cantidad de linked del nodo i-esimo
		fprintf(file, "%i\n", _graph->_nodes[i]._countLinked);
		//Almacenar los linked del nodo i-esimo
		for (int j = 0; j < _graph->_nodes[i]._countLinked; j++)
			fprintf(file, "%i\n", _graph->_nodes[i]._linked[j]);
		if(_graph->_nodes[i]._countLinked > 0)
			free(_graph->_nodes[i]._linked);
	}
	free(ID);
	//Almacenar los restantes vectores que estan en el grafo
	for (int i = 0; i < _graph->_countNode; i++)
	{
		//Almacenar los adyacentes de cada nodo
		fprintf(file, "%i\n", _graph->_countAdjPerNode[i]);
		//Almacenar la aproxIntraSim de cada nodo
		fprintf(file, "%f\n", _graph->_aproxIntraSim[i]);		
	}
	
	//Almacenar el numero de compontente conexa 
	fprintf(file, "%i\n", _graph->_countCennectedComp);

	//Almacenar el camino donde se encuentran las componentes conexas
	for (int i = 0; i < _graph->_countCennectedComp; i++)
	{
		fprintf(file, "%s\n", connectedComponent[i]._path);
		free(connectedComponent[i]._path);
	}
	free(connectedComponent);
	fclose(file);	
	//LIBERAR LOS DATOS DEL GRAFO	
	free(_graph->_aproxIntraSim);
	free(_graph->_countAdjPerNode);
	free(_graph->_nodes);
	free(_graph->_relevance);
	free(_graph);
}

void WriteConnectedComponent()
{
	for (int i = 0; i < _graph->_countCennectedComp; i++)
	{		
		if(connectedComponent[i]._active)
		{
			FILE* file = fopen(connectedComponent[i]._path, "w");
			//Almaceno la cantidad de vertices y la cantidad de hijos que contiene la componente i-esima
			fprintf(file, "%d\n", connectedComponent[i]._countVertices);

			//Almacenar los vertices que representan la componente conexa
			for (int j = 0; j < connectedComponent[i]._countVertices; j++)
				fprintf(file, "%d\n", connectedComponent[i]._vertices[j]);			
			
			//Almacenar la lista de adyacencia que representa la componente conexa
			for (int j = 0; j < connectedComponent[i]._countVertices; j++)
			{
				for (int k = 0; k < _graph->_countAdjPerNode[connectedComponent[i]._vertices[j]]; k++)
					fprintf(file, "%d\n", connectedComponent[i]._adjacentList[j][k]);
				if(_graph->_countAdjPerNode[connectedComponent[i]._vertices[j]] > 0)
					free(connectedComponent[i]._adjacentList[j]);
			}
			free(connectedComponent[i]._vertices);
			free(connectedComponent[i]._adjacentList);
			fclose(file);
		}
		
	}
}

void WriteDocuments()
{
	//ALMACENAR LOS DOCUMENTOS
	FILE* file = fopen(pathDocument, "w");	
	for (int i = 0; i < _graph->_countNode; i++)
	{		
		//Almacenar el DocSize del documento i-esimo
		fprintf(file, "%d ", _document[i].DocSize);
		//Almacenar la Norma del documento i-esimo
		fprintf(file, "%.16f ", _document[i].Norm);
		//Almacenar la Norma del documento i-esimo
		fprintf(file, "%i\n", _document[i]._index);		
	}	

	//Almacenar el size de los vectores
	fprintf(file, "%i\n", _size);

	//Almacenar el vector de postion y frecuencia 
	for (int i = 0; i < _graph->_countNode; i++)
	{
		for (int j = 0; j < _document[i].DocSize; j++)
		{
			fprintf(file, "%i ", _wordPosition[i][j]);

			fprintf(file, "%i\n", (int)_frecuency[i][j]);
		}
		if(_document[i].DocSize > 0)
		{
			free(_wordPosition[i]);
			free(_frecuency[i]);
		}
	}		

	free(_wordPosition);
	free(_frecuency);
	
	//ALMACENAR EL VECTOR DE TERMINOS GENERAL
	//Almacenar la cantidad de terminos generales
	fprintf(file, "%i\n", cTERMINOS);
	//Almacenar los terminos del vector general
	for (int i = 0; i < cTERMINOS; i++)
	{
		fprintf(file, "%s\n", terminos[i]);
		free(terminos[i]);
	}
	free(_document);
	fclose(file);
}

int ReadDocuments()
{
	if(option == 0)
		return 0;
	char* buffer = (char *)malloc(sizeof(char)*tBuffer);
	char* temp = (char*)malloc(sizeof(char)*100);
	FILE* file = fopen(pathDocument, "r");

	_document = (Document*)malloc(sizeof(Document)*_graph->_countNode);

	for (int i = 0; i < _graph->_countNode; i++)
	{		
		fgets(buffer, tBuffer, file);
		int j = 0, k = 0;		
		//Leer el DocSize del documento i-esimo
		k = 0;
		while (buffer[j] != ' ')
		{
			temp[k] = buffer[j];
			k++; j++;
		}
		temp[k] = '\0';
		_document[i].DocSize = atoi(temp);
		j++;

		//Leer la Norma del documento i-esimo
		k = 0;
		while (buffer[j] != ' ')
		{
			temp[k] = buffer[j];
			k++; j++;
		}
		temp[k] = '\0';
		_document[i].Norm = atof(temp);				

		//Leer el _index del documento i-esimo
		k = 0;
		while (buffer[j] != '\n' && buffer[j] != '\r')
		{
			temp[k] = buffer[j];
			k++; j++;
		}
		temp[k] = '\0';
		_document[i]._index = atoi(temp);		
	}
	free(temp);
	//Leer el size de los vectores de las posiciones y de las frecuencias
	fscanf(file, "%i\n", &_size);

	_wordPosition = (int**)malloc(sizeof(int*)*_graph->_countNode);
	_frecuency = (float**)malloc(sizeof(float*)*_graph->_countNode);

	for (int i = 0; i < _graph->_countNode; i++)
	{
		_wordPosition[i] = (int*)malloc(sizeof(int)*_document[i].DocSize);
		_frecuency[i] = (float*)malloc(sizeof(float)*_document[i].DocSize);
		for (int j = 0; j < _document[i].DocSize; j++)
		{
			int tmp = 0;
			fscanf(file, "%i %i\n", &_wordPosition[i][j], &tmp);			
			_frecuency[i][j] = tmp;
		}
	}		

	fscanf(file, "%i\n", &cTERMINOS);	
	
	terminos = (char**)malloc(sizeof(char*)*cTERMINOS);	
	for (int i = 0; i < cTERMINOS; i++)
	{			
		fgets(buffer, tBuffer, file);
		int j = 0;
		//Leer los terminos del documento i-esimo
        terminos[i] = (char*)malloc(sizeof(char)*30);
		while (buffer[j] != '\r' && buffer[j] != '\n')
		{
			terminos[i][j] = buffer[j];
			j++;
		}		
		terminos[i][j] = '\0';
	}	
	fclose(file);
	free(buffer);
}

void DClustR()
{	
	_graph->_relevance = (float*)calloc(_graph->_countNode, sizeof(float));
	bool* mark = (bool*)calloc(_graph->_countNode, sizeof(bool));
	//Actualizo el grafo	
	for (int i = 0; i < M.size(); i++)
	{
		if(!mark[M[i]])
		{
			V_s = vector<int>();
			C_0 = vector<int>();
			vector<int> connectedComp = GetConnectedComponent(M[i], mark);						
			//ccValue++;
			if(connectedComp.size() == 1)
			{
				_graph->_nodes[M[i]]._type = CENTER;
				_graph->_nodes[M[i]]._degreeCenter++;
				_graph->_nodes[M[i]]._seek = true;
			}
			else
			{			
				//CONSTRUIR LA LISTA L'
				//Actualizar para cada vertice en G' el centro con mayor relevancia que lo cubren
				vector<int> L1;
				int ccSize = connectedComp.size();
				for (int j = 0; j < ccSize; j++)
				{
					for (int k = 0; k < _graph->_countAdjPerNode[connectedComp[j]]; k++)
					{
						int v = connectedComponent[_graph->_nodes[connectedComp[j]]._connectedComp]._adjacentList[_graph->_nodes[connectedComp[j]]._pos][k];
						if(_graph->_nodes[v]._type == CENTER && _graph->_relevance[v] > _graph->_nodes[connectedComp[j]]._relGreaterVertex)						
							_graph->_nodes[connectedComp[j]]._relGreaterVertex = _graph->_relevance[v];							
					}
				}
				//Analizar los vertices en V_s
				int Vsize = V_s.size();
				for (int j = 0; j < Vsize; j++)
				{					
					bool NotCovered = false;
					int iVertex = V_s[j];
					if(_graph->_nodes[iVertex]._degreeCenter == 0)
						NotCovered = true;
					for (int k = 0; k < _graph->_countAdjPerNode[iVertex]; k++)
					{
						int v = connectedComponent[_graph->_nodes[iVertex]._connectedComp]._adjacentList[_graph->_nodes[iVertex]._pos][k];							
						if(_graph->_nodes[v]._type == SATELLITE)
						{
							if(_graph->_nodes[v]._degreeCenter == 0)
								NotCovered = true;	
							else 
							{
								if(_graph->_nodes[v]._relGreaterVertex < _graph->_relevance[iVertex])
								{
									_graph->_nodes[v]._active = true;
									NotCovered = true;
								}
							}
						}
					}

					if(NotCovered)
					{
						_graph->_nodes[iVertex]._proccessed = true;
						L1.push_back(iVertex);				
					}
				}
				//Analizar los vertices en C'
				int C0size = C_0.size();
				for (int j = 0; j < C0size; j++)
				{
					bool HasVertexActive = false;
					int iVertex = C_0[j];
					for (int k = 0; k < _graph->_countAdjPerNode[iVertex]; k++)
					{
						int tmp = connectedComponent[_graph->_nodes[iVertex]._connectedComp]._adjacentList[_graph->_nodes[iVertex]._pos][k];
						if(_graph->_relevance[tmp] > _graph->_relevance[iVertex])
						{
							if(!_graph->_nodes[tmp]._proccessed)
							{
								_graph->_nodes[tmp]._proccessed = true;
								L1.push_back(tmp);
							}
							_graph->_nodes[iVertex]._weak = true;
						}
						if(_graph->_nodes[tmp]._active)
							HasVertexActive = true;
					}
							

					if(_graph->_nodes[iVertex]._weak || HasVertexActive)
					{
						_graph->_nodes[iVertex]._type = SATELLITE;
						_graph->_nodes[iVertex]._degreeCenter--;
						_graph->_nodes[iVertex]._weak = false;

						for (int k = 0; k < _graph->_countAdjPerNode[iVertex]; k++)
							_graph->_nodes[connectedComponent[_graph->_nodes[iVertex]._connectedComp]._adjacentList[_graph->_nodes[iVertex]._pos][k]]._degreeCenter--;


						if(_graph->_relevance[iVertex] > 0)
						{
							_graph->_nodes[iVertex]._proccessed = true;
							L1.push_back(iVertex);
						}
					}
				}

				//Ordenar L1 decrecientemente por la relevancia de los vertices
				int L1size = L1.size();
				for (int j = 0 ; j < L1size - 1; j++)
					for (int k = L1size - 1 ; k > j ; k--)
						if (_graph->_relevance[L1[k]] > _graph->_relevance[L1[k - 1]])
						{
							int _TMP = L1[k];
							L1[k] = L1[k-1];
							L1[k-1] = _TMP;
						}	
				
						//SELECCIONANDO LOS VERTICES DEL CUBRIMIENTO DEL GRAFO
				for (int j = 0; j < L1size; j++)
				{
					bool notCoveredAdj = true;
					int iVertex = L1[j];
					//Condicion a)
					if (_graph->_nodes[iVertex]._degreeCenter == 0)
						notCoveredAdj = false;
					//Condicion b)
					else
					{
						for (int k = 0; k < _graph->_countAdjPerNode[iVertex]; k++)
						{
							int adj = connectedComponent[_graph->_nodes[iVertex]._connectedComp]._adjacentList[_graph->_nodes[iVertex]._pos][k];
							if (_graph->_nodes[adj]._type == SATELLITE && _graph->_nodes[adj]._degreeCenter == 0)
							{
								notCoveredAdj = false;
								break;
							}
						}
					}
					//Si no existe un vertice adyacente a L[i] que no ha sido cubierto aplicar la condicion b) 
					if (!notCoveredAdj)
					{
						_graph->_nodes[iVertex]._type = CENTER;
						_graph->_nodes[iVertex]._degreeCenter++;
						for (int k = 0; k < _graph->_countAdjPerNode[iVertex]; k++)
							_graph->_nodes[connectedComponent[_graph->_nodes[iVertex]._connectedComp]._adjacentList[_graph->_nodes[iVertex]._pos][k]]._degreeCenter++;
						//inserto el nodo en C			
						C_0.push_back(iVertex);			
					}
				}
				//Organizo descendientemente de acuerdo al grado de los vertices
				int C1size = C_0.size();
				for (int j = 0 ; j < C1size - 1; j++)
					for (int k = C1size - 1 ; k > j ; k--)
						if (_graph->_countAdjPerNode[C_0[k]] > _graph->_countAdjPerNode[C_0[k - 1]])
						{
							int _TMP = C_0[k];
							C_0[k] = C_0[k-1];
							C_0[k-1] = _TMP;
						}	
	
				//ETAPA DE FILTRADO. ELIMINAR LOS VERTICES NO UTILES						
				for (int j = 0; j < C1size; j++)
				{				
					if (_graph->_nodes[C_0[j]]._type == CENTER)
					{
						_graph->_nodes[C_0[j]]._seek = true;
						for (int k = 0; k < _graph->_countAdjPerNode[C_0[j]]; k++)
						{
							int _u = connectedComponent[_graph->_nodes[C_0[j]]._connectedComp]._adjacentList[_graph->_nodes[C_0[j]]._pos][k];
							Node u = _graph->_nodes[_u];
							if (u._type == CENTER && !u._analyzed)
							{
								int _shared = 0;
								int _posU = connectedComponent[_graph->_nodes[C_0[j]]._connectedComp]._adjacentList[_graph->_nodes[C_0[j]]._pos][k];
								for (int t = 0; t < _graph->_countAdjPerNode[_u]; t++)
								{
									if (_graph->_nodes[connectedComponent[_graph->_nodes[_posU]._connectedComp]._adjacentList[_graph->_nodes[_posU]._pos][t]]._degreeCenter > 1)
										_shared++;
								}

								if (_shared > (_graph->_countAdjPerNode[_u] - _shared))
								{
									_graph->_nodes[_u]._degreeCenter--;
									_graph->_nodes[_u]._type = SATELLITE;
									for (int t = 0; t < _graph->_countAdjPerNode[_u]; t++)
									{
										if (_graph->_nodes[connectedComponent[_graph->_nodes[_posU]._connectedComp]._adjacentList[_graph->_nodes[_posU]._pos][t]]._degreeCenter == 1)
										{											
											if (_graph->_nodes[C_0[j]]._countLinked == 0)
												_graph->_nodes[C_0[j]]._linked = (int*)malloc(sizeof(int));
											else
												_graph->_nodes[C_0[j]]._linked = (int*)realloc(_graph->_nodes[C_0[j]]._linked, sizeof(int)*(_graph->_nodes[C_0[j]]._countLinked + 1));

											//Asigno el vertice a la lista de linked de C[i]
											_graph->_nodes[C_0[j]]._linked[_graph->_nodes[C_0[j]]._countLinked] = connectedComponent[_graph->_nodes[_posU]._connectedComp]._adjacentList[_graph->_nodes[_posU]._pos][t];
											_graph->_nodes[C_0[j]]._countLinked++;
										}
										_graph->_nodes[connectedComponent[_graph->_nodes[_posU]._connectedComp]._adjacentList[_graph->_nodes[_posU]._pos][t]]._degreeCenter--;
									}
								}
								else
									_graph->_nodes[_u]._analyzed = true;
							}
						}
					}
				}
			}
		}
	}	
	free(mark);	
}

void GetConnectedComp(int node, bool* mark)
{
	queue<int> _queue;
	_queue.push(node);	
	while (_queue.size() > 0)
	{
		int temp = _queue.front();		
		_queue.pop();
		for (int i = 0; i < _graph->_countAdjPerNode[temp]; i++)
		{	
			if (!mark[connectedComponent[_graph->_nodes[temp]._connectedComp]._adjacentList[_graph->_nodes[temp]._pos][i]])
			{
				mark[connectedComponent[_graph->_nodes[temp]._connectedComp]._adjacentList[_graph->_nodes[temp]._pos][i]] = true;
				_queue.push(connectedComponent[_graph->_nodes[temp]._connectedComp]._adjacentList[_graph->_nodes[temp]._pos][i]);
			}
		}			
	}
}

/*------------Parametros----------------
argv[1] - Un int = [0,1]. Dice si ya existe un grafo en disco
-argv[2] - Cantidad de documentos a insertar
-argv[3] - Direccion de los documentos a insertar
-argv[4] - umbral
-argv[5] - Direccion del directorio donde se almacenan las componentes conexas
-argv[6] - Direccion de salida de los grupos
-argv[7] - Direccion del grafo
-argv[8] - Direccion de salida de los documentos representados por el grafo
--------------Parametros------------------
*/
int main()
{		
	option = atoi(__argv[1]);
	char* pathAdd = NULL;	
	float threshold = atof(__argv[4]);
	int countAdd = atoi(__argv[2]);	

	if(countAdd > 0)
		pathAdd = __argv[3];	

	pathCC = __argv[5];
	pathClusters = __argv[6];
	pathGraph = __argv[7];
	pathDocument = __argv[8];

	if(option != 0)
	{
		ReadGraph();

		ReadDocuments();
	}	

	Read(pathAdd, threshold, countAdd);	

	
	timeb ini, end;

	int aux = 0;

	ftime(&ini);

	BuildAdjacentList(threshold);	
	
	ftime(&end);
	
	aux += (end.time - ini.time) * 1000 + (end.millitm - ini.millitm);

	WriteDocuments();
	
	ftime(&ini);

	DClustR();

	ftime(&end);
	
	aux += (end.time - ini.time) * 1000 + (end.millitm - ini.millitm);

	printf("%d\n", aux);	

	/*int countCC = 0;
	bool* mark = (bool*)calloc(_graph->_countNode, sizeof(bool));
	for (int i = 0; i < _graph->_countNode; i++)
	{
		if(connectedComponent[_graph->_nodes[i]._connectedComp]._active && !mark[i])
		{
			countCC++;
			mark[i] = true;
			GetConnectedComp(i, mark);
		}
	}
	free(mark);
	printf("%d\n", countCC);*/

	WriteClusters();	
	
	WriteConnectedComponent();

	WriteGraph();		

	return 0;
}