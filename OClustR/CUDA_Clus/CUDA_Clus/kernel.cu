
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <time.h>
#include <sys/timeb.h>
#include <queue>
#include <windows.h>

#define tBuffer 100000
#define threadPerBlocksV 16
#define threadPerBlocksH 16
#define CENTER 1
#define SATELLITE 0

using namespace std;

struct Node
{		
	bool _seek;
	int _type;
	int _degreeCenter;
};

struct Graph
{	
	int _countNode;
	Node* _nodes;
	int** _adjacentList;	
	int* _countAdjPerNode;
	float* _aproxIntraSim;
	float* _relevance;
	int** _linked;
	int* _countLinkedPerNode;
};

Graph *_graph;
bool* _used;
bool* _analyzed;
bool* _covered; 
char** ID;

struct Document
{	
	int DocSize;
	float Norm;	
	int _index;
};

Document* _document;

int* _wordPosition = 0;
float* _frecuency = 0;
int _size = 0;
int _countTerm;

vector<int> C;
vector<int> L;

MEMORYSTATUSEX statexInitial, statexFinal;

//__constant__ float* _devTerm;

__global__ void BuildSimilarityNode(Document* _devDoc, float* _frecD1, float* frec, int* wordPos, int indexDoc, int _countDocs, float* result)
{
	int tid = threadIdx.x + blockDim.x* blockIdx.x;
	__shared__ float _frecPerDoc[threadPerBlocksV][threadPerBlocksH];
	Document d = _devDoc[indexDoc - 1];	
	Document d2;
	if(tid + indexDoc < _countDocs)
	{		
		d2 = _devDoc[tid + indexDoc];
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
	if(threadIdx.y == 0 && tid + indexDoc < _countDocs)
		result[tid + indexDoc] = (d.Norm > 0 && d2.Norm > 0) ? _frecPerDoc[0][threadIdx.x] / (d.Norm*d2.Norm) : 0.0;
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

__global__ void Relevance(int** adjList, float* aproxSim ,int* _countAdjPerNode, int countNode, float* relevance)
{
	int tid = threadIdx.x + blockDim.x* blockIdx.x;
	__shared__ float _relPerVertex[threadPerBlocksV][threadPerBlocksH];
	int _countAdj;
	if(tid < countNode)
	{		
		float result = 0;
		_countAdj = _countAdjPerNode[tid];
		float _aprox = aproxSim[tid];
		int tidY = threadIdx.y;		
		while(tidY < _countAdj)
		{
			int posC = adjList[tid][tidY];
			int _countAdjC = _countAdjPerNode[posC];
			float _aproxC = aproxSim[posC];
			if(_countAdjC <= _countAdj)
				result++;
			if(_aproxC <= _aprox)
				result++;			
			tidY += threadPerBlocksV;
		}
		_relPerVertex[threadIdx.y][threadIdx.x] = result;
	}
	__syncthreads();
	
	int i = threadPerBlocksV/2;
	while(i != 0)
	{
		if(threadIdx.y < i)
			_relPerVertex[threadIdx.y][threadIdx.x] += _relPerVertex[threadIdx.y + i][threadIdx.x];
		__syncthreads();
		i/=2;
	}
	if(threadIdx.y == 0 && tid < countNode)
		relevance[tid] = (_countAdj > 0)? (float)_relPerVertex[threadIdx.y][threadIdx.x] / (float)(2 * _countAdj): 0;
}

void WriteClusters()
{
	//ESCRIBO CADA GRUPO EN DISCO
	FILE* f1;
	bool* mark = (bool*)calloc(_graph->_countNode, sizeof(bool));
	fopen_s(&f1, __argv[3], "w");

	int countNode = 0;
	for (int i = 0; i < C.size(); i++)
	{
		if (_graph->_nodes[C[i]]._seek)
		{
			mark[C[i]] = true;
			fputs(ID[C[i]], f1);
			countNode++;
			for (int j = 0; j < _graph->_countAdjPerNode[C[i]]; j++)
			{
				fputs(", ", f1);
				fputs(ID[_graph->_adjacentList[C[i]][j]], f1);
				mark[_graph->_adjacentList[C[i]][j]] = true;
			}
			for (int j = 0; j < _graph->_countLinkedPerNode[C[i]]; j++)
			{
				fputs(", ", f1);
				fputs(ID[_graph->_linked[C[i]][j]], f1);
				mark[_graph->_linked[C[i]][j]] = true;
			}
			fputs("\n", f1);
		}
	}
	printf("La cantidad de nodos agrupados son: %d\n", countNode);
	fclose(f1);
	printf("Checking if whole nodes are mark.....\n");
	for (int i = 0; i < _graph->_countNode; i++)
	{
		if(!mark[i])
		{
			printf("%d\n", i);
		}
	}
	printf("Finishing if whole nodes are mark.....\n");
	free(mark);
	//LIBERO EL GRAFO
	for (size_t i = 0; i < _graph->_countNode; i++)
	{
		if (_graph->_countAdjPerNode[i] > 0)
			free(_graph->_adjacentList[i]);
		if (_graph->_countLinkedPerNode[i] > 0)
			free(_graph->_linked[i]);
		free(ID[i]);
	}
	free(_graph->_countAdjPerNode);
	free(_graph->_countLinkedPerNode);
	free(ID);
	free(_graph->_linked);
	free(_graph->_adjacentList);
	free(_graph->_nodes);	
	free(_graph);
}

void Read(char* path)
{	
	char** terminos = 0;
	int cTERMINOS = 0;
	int cObjetos = 0;	
	_size = 0;
	FILE* f;
	fopen_s(&f, path, "r");

	char* buffer = 0;
	buffer = (char *)malloc(sizeof(char)*tBuffer);

	char* temp = 0;
	temp = (char *)malloc(sizeof(char)* 30);

	int i = 0, j = 0;

	char* FIN = fgets(buffer, tBuffer, f);

	while (FIN != 0)
	{
		if (cObjetos == 0)
		{
			_document = (Document*)malloc(sizeof(Document));
			ID = (char**)malloc(sizeof(char*));
		}
		else
		{
			_document = (Document*)realloc(_document, sizeof(Document)*(cObjetos + 1));
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

			if(_wordPosition == 0)
			{
				_wordPosition = (int*)malloc(sizeof(int));
				_frecuency = (float*)malloc(sizeof(float));
			}
			else
			{
				_wordPosition = (int*)realloc(_wordPosition, sizeof(int)*(_size + 1));
				_frecuency = (float*)realloc(_frecuency, sizeof(float)*(_size + 1));
			}

			_wordPosition[_size] = index;
			_frecuency[_size] = atof(temp);

			_document[cObjetos].Norm += powf(_frecuency[_size], 2);
			if(_document[cObjetos].DocSize == 0)
				_document[cObjetos]._index = _size;
			_size++;			
			_document[cObjetos].DocSize++;
		}

		_document[cObjetos].Norm = sqrtf(_document[cObjetos].Norm);

		//Creo los nodos del grafo
		if (cObjetos == 0)
		{
			_graph = (Graph*)malloc(sizeof(Graph));			
			_graph->_countNode = 0;
			_graph->_aproxIntraSim = 0;
			_graph->_countAdjPerNode = 0;
			_graph->_relevance = 0;
			_graph->_countLinkedPerNode = 0;
			_graph->_linked = 0;
			_graph->_nodes = (Node*)malloc(sizeof(Node));
		}
		else
			_graph->_nodes = (Node*)realloc(_graph->_nodes, sizeof(Node)*(_graph->_countNode + 1));					
		
		_graph->_nodes[_graph->_countNode]._seek = false;
		_graph->_nodes[_graph->_countNode]._type = SATELLITE;
		_graph->_nodes[_graph->_countNode]._degreeCenter = 0;				
		_graph->_countNode++;

		cObjetos++;

		FIN = fgets(buffer, tBuffer, f);
	}

	_countTerm = cTERMINOS;

	fclose(f);
	f = 0;

	free(buffer);
	buffer = 0;

	free(temp);
	temp = 0;

	for (int i = 0; i < cTERMINOS; i++)
	{
		free(terminos[i]); terminos[i] = 0;
	}
	free(terminos);
	terminos = 0;
}
size_t freeM, totalM, finalFreeM;
void BuildGraph(float beta)
{	
	_graph->_adjacentList = (int**)malloc(sizeof(int*)*_graph->_countNode);
	_graph->_aproxIntraSim = (float*)calloc(_graph->_countNode, sizeof(float));
	_graph->_countAdjPerNode = (int*)calloc(_graph->_countNode, sizeof(int));

	int* _devWordPosition;
	float* _devFrecuency;
	float* _devSimilarityNode;
	Document* _devDoc;

	cudaError_t cudaStatus = cudaMemGetInfo(&freeM, &totalM);	
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
	cudaStatus = cudaMemcpy(_devWordPosition, _wordPosition, _size*sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(_devFrecuency, _frecuency, _size*sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(_devDoc, _document, _graph->_countNode *sizeof(Document), cudaMemcpyHostToDevice);
	float* _devFrec;
	//asigno memoria para el array de terminos 
	float* _frecD1 = (float*)calloc(_countTerm, sizeof(float));
	cudaStatus = cudaMalloc((void**)&_devFrec, _countTerm*sizeof(float));			
	//copio la frecuencia del nodo i para la memoria del dispositivo
	cudaStatus = cudaMemcpy(_devFrec, _frecD1, _countTerm*sizeof(float), cudaMemcpyHostToDevice);
	free(_frecD1);

	cudaMemGetInfo(&finalFreeM, &totalM);
	finalFreeM = freeM - finalFreeM;
	int countAdj = 0;
	for(int i = 0; i < _graph->_countNode; i++)
	{
		if(i == 7498 || i == 30270)
			printf("Janier...");
		if(i < _graph->_countNode - 1)
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
			grids = ((_graph->_countNode - (i + 1))* 2 + (threadPerBlocksH - 1))/threadPerBlocksH;
			BuildSimilarityNode<<<grids, blocks>>>(_devDoc, _devFrec, _devFrecuency, _devWordPosition, i + 1, _graph->_countNode, _devSimilarityNode);

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));			
			}
			//Copio el resultado para la memoria del CPU
			cudaStatus = cudaMemcpy(result, _devSimilarityNode, _graph->_countNode*sizeof(float), cudaMemcpyDeviceToHost);									

			grids = (_countTerm + 256)/256;
			Finalize<<<grids, 256>>>(_devFrec, _countTerm);

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));			
			}

			//Recorro el resultado y construyo la lista de adyacente del nodo i		
			for(int j = i + 1; j < _graph->_countNode; ++j)
			{			
				if(result[j] >= beta)
				{
					countAdj++;
					//Agrego la arista a la lista de adyacencia del nodo i
					if (_graph->_countAdjPerNode[i] == 0)
						_graph->_adjacentList[i] = (int*)malloc(sizeof(int));
					else
						_graph->_adjacentList[i] = (int*)realloc(_graph->_adjacentList[i], sizeof(int)*(_graph->_countAdjPerNode[i] + 1));				

					_graph->_adjacentList[i][_graph->_countAdjPerNode[i]] = j;
					_graph->_countAdjPerNode[i]++;

					//Agrego la arista a la lista de adyacencia del nodo j
					if (_graph->_countAdjPerNode[j] == 0)
						_graph->_adjacentList[j] = (int*)malloc(sizeof(int));
					else
						_graph->_adjacentList[j] = (int*)realloc(_graph->_adjacentList[j], sizeof(int)*(_graph->_countAdjPerNode[j] + 1));				

					_graph->_adjacentList[j][_graph->_countAdjPerNode[j]] = i;
					_graph->_countAdjPerNode[j]++;
					//Aumento la similitud del nodo i y el nodo j
					_graph->_aproxIntraSim[i] += result[j];
					_graph->_aproxIntraSim[j] += result[j];				
				}
			}
			
		}
		_graph->_aproxIntraSim[i] = (_graph->_countAdjPerNode[i] > 0) ? _graph->_aproxIntraSim[i] / _graph->_countAdjPerNode[i] : 0;			
	}
	printf("La cantidad de adyacentes son %d\n", countAdj);
	cudaFree(_devFrec);
	free(result);
	cudaFree(_devSimilarityNode);	
	free(_wordPosition);
	free(_frecuency);
	free(_document);
	_wordPosition = 0;
	_frecuency = 0;
	_document = 0;
	cudaFree(_devDoc);
	cudaFree(_devFrecuency);
	cudaFree(_devWordPosition);
}

void OClustR()
{		
	//ETAPA II: CALCULO DE LA RELEVANCIA
	//int** _devAdjList;
	//int* _devCountAdj;
	//float* _devAproxSim;
	//float* _devRelevance;
	////int* _devTruthPositon;
	//
	//cudaError_t cudaStatus = cudaMemGetInfo(&freeM, &totalM);

	//_graph->_relevance = (float*)malloc(_graph->_countNode*sizeof(float));	
	//
	//int** h_adjList = (int**)malloc(_graph->_countNode*sizeof(int*));

	//for(int i = 0; i < _graph->_countNode; i++)
	//{
	//	cudaStatus = cudaMalloc((void**)&h_adjList[i], _graph->_countAdjPerNode[i]*sizeof(int));
	//	cudaStatus = cudaMemcpy(h_adjList[i], _graph->_adjacentList[i], _graph->_countAdjPerNode[i]*sizeof(int), cudaMemcpyHostToDevice);
	//}

	//cudaStatus = cudaMalloc((void***)&_devAdjList, _graph->_countNode*sizeof(int*));
	//cudaStatus = cudaMemcpy(_devAdjList, h_adjList, _graph->_countNode*sizeof(int*), cudaMemcpyHostToDevice);

	////asigno memoria para cada arrays en el GPU
	//cudaStatus = cudaMalloc((void**)&_devCountAdj, _graph->_countNode*sizeof(int));	
	//cudaStatus = cudaMalloc((void**)&_devAproxSim, _graph->_countNode*sizeof(float));	
	//cudaStatus = cudaMalloc((void**)&_devRelevance, _graph->_countNode*sizeof(float));		

	////copio los respectivo valores de Node y compactAdjList para los arrays en el GPU
	//cudaStatus = cudaMemcpy(_devCountAdj, _graph->_countAdjPerNode, _graph->_countNode*sizeof(int), cudaMemcpyHostToDevice);
	//cudaStatus = cudaMemcpy(_devAproxSim, _graph->_aproxIntraSim, _graph->_countNode*sizeof(float), cudaMemcpyHostToDevice);
	//

	////ejecuto el kernel de la relevancia
	//dim3 blocks(threadPerBlocksH, threadPerBlocksV);
	//int grids = (_graph->_countNode* 2 + (threadPerBlocksH - 1))/threadPerBlocksH;
	//Relevance<<<grids, blocks>>>(_devAdjList, _devAproxSim, _devCountAdj, _graph->_countNode, _devRelevance);
	//cudaStatus = cudaGetLastError();

	//size_t auxFreeM;
	//cudaMemGetInfo(&auxFreeM, &totalM);

	//if(freeM - auxFreeM > finalFreeM)
	//	finalFreeM = freeM - auxFreeM;

	//printf("%d\n", finalFreeM/1024/1024);

	////copio los nodos de vuelta al CPU
	//cudaStatus = cudaMemcpy(_graph->_relevance, _devRelevance, _graph->_countNode*sizeof(float), cudaMemcpyDeviceToHost);	
	//for(int i =0; i < _graph->_countNode; ++i)	
	//	cudaFree(h_adjList[i]);

	//cudaFree(_devCountAdj);
	//cudaFree(_devAdjList);
	//cudaFree(_devAproxSim);
	//cudaFree(_devRelevance);
	//free(h_adjList);
	//for(int i = 0; i < _graph->_countNode; ++i)
	//{	
	//	if(i == 7498)
	//		printf("Relevance");
	//	if(_graph->_countAdjPerNode[i] > 0)
	//	{
	//		if (_graph->_relevance[i] > 0)
	//			L.push_back(i);			
	//	}		
	//	else 
	//	{
	//		//Si el vertice es aislado lo inserto directamente en C
	//		C.push_back(i);
	//		_graph->_nodes[i]._seek = true;
	//		_graph->_nodes[i]._type = CENTER;
	//		_graph->_nodes[i]._degreeCenter++;
	//	}
	//}
	
	//HALLO LA RELEVANCIA DE CADA VERTICE
	_graph->_relevance = (float*)malloc(_graph->_countNode*sizeof(float));	
	for (int i = 0; i < _graph->_countNode; i++)
	{		
		for (int j = 0; j < _graph->_countAdjPerNode[i]; j++)
		{
			int pos = _graph->_adjacentList[i][j];
			if (_graph->_countAdjPerNode[pos] <= _graph->_countAdjPerNode[i])
				_graph->_relevance[i]++;
			if ((_graph->_aproxIntraSim[pos]*1.0)/_graph->_countAdjPerNode[pos] <= (_graph->_aproxIntraSim[i]*1.0)/_graph->_countAdjPerNode[i])
				_graph->_relevance[i]++;
		}			
		_graph->_relevance[i] = (_graph->_countAdjPerNode[i] > 0)? _graph->_relevance[i]/(2.0*_graph->_countAdjPerNode[i]): 0.0;
		if (_graph->_countAdjPerNode[i] > 0)
		{
			//Si la relevancia del nodo i es mayor que cero la inserto en L
			if (_graph->_relevance[i] > 0)
				L.push_back(i);
		}
		else
		{
			//Si el vertice es aislado lo inserto directamente en C
			C.push_back(i);
			_graph->_nodes[i]._degreeCenter++;
			_graph->_nodes[i]._type = CENTER;
			_graph->_nodes[i]._seek = true;
		}		
	}	

	printf("La cantidad de objetos en L inicialmente %d\n", L.size());
	printf("La cantidad de objetos en C inicialmente %d\n", C.size());

	free(_graph->_aproxIntraSim);	
	int lSize = L.size();
	for (int i = 0 ; i < lSize - 1; i++)
		for (int j = lSize - 1 ; j > i ; j--)
			if (_graph->_relevance[L[j]] > _graph->_relevance[L[j - 1]])
			{
				int _TMP = L[j];
				L[j] = L[j-1];
				L[j-1] = _TMP;
			}		
	free(_graph->_relevance);	
	//SELECCIONANDO LOS VERTICES DEL CUBRIMIENTO DEL GRAFO
	for (int i = 0; i < lSize; i++)
	{
		bool notCoveredAdj = true;
		if(L[i] == 7498)
			printf("Relevance");
		//Condicion a)
		if (_graph->_nodes[L[i]]._degreeCenter == 0)
			notCoveredAdj = false;
		//Condicion b)
		else
		{
			for (int j = 0; j < _graph->_countAdjPerNode[L[i]]; j++)
			{
				if (_graph->_nodes[_graph->_adjacentList[L[i]][j]]._degreeCenter == 0)
				{
					notCoveredAdj = false;
					break;
				}
			}
		}
		//Si no existe un vertice adyacente a L[i] que no ha sido cubierto aplicar la condicion b) 
		if (!notCoveredAdj)
		{
			_graph->_nodes[L[i]]._type = CENTER;
			_graph->_nodes[L[i]]._degreeCenter++;
			for (int j = 0; j < _graph->_countAdjPerNode[L[i]]; j++)
				_graph->_nodes[_graph->_adjacentList[L[i]][j]]._degreeCenter++;
			//inserto el nodo en C						
			C.push_back(L[i]);			
		}
	}	

	printf("La cantidad de objetos en C %d\n", C.size());

	int cSize = C.size();
	for (int i = 0 ; i < cSize - 1; i++)
		for (int j = cSize - 1 ; j > i ; j--)
			if (_graph->_countAdjPerNode[C[j]] > _graph->_countAdjPerNode[C[j - 1]])
			{
				int _TMP = C[j];
				C[j] = C[j-1];
				C[j-1] = _TMP;
			}	
	
	//ETAPA DE FILTRADO. ELIMINAR LOS VERTICES NO UTILES
	_analyzed = (bool*)calloc(_graph->_countNode, sizeof(bool));
	_graph->_countLinkedPerNode = (int*)calloc(_graph->_countNode, sizeof(int));
	_graph->_linked = (int**)malloc(_graph->_countNode*sizeof(int*));
	for (size_t i = 0; i < cSize; i++)
	{
		if(C[i] == 7498)
			printf("Relevance");
		if (_graph->_nodes[C[i]]._type == CENTER)
		{
			_graph->_nodes[C[i]]._seek = true;
			for (size_t j = 0; j < _graph->_countAdjPerNode[C[i]]; j++)
			{
				int _posU = _graph->_adjacentList[C[i]][j];
				Node* u = &(_graph->_nodes[_graph->_adjacentList[C[i]][j]]);
				if (u->_type == CENTER && !_analyzed[_posU])
				{
					int _shared = 0;					
					for (int k = 0; k < _graph->_countAdjPerNode[_posU]; k++)
					{
						if (_graph->_nodes[_graph->_adjacentList[_posU][k]]._degreeCenter > 1)
							_shared++;
					}

					if (_shared > (_graph->_countAdjPerNode[_posU] - _shared))
					{
						u->_degreeCenter--;
						u->_type = SATELLITE;
						for (int k = 0; k < _graph->_countAdjPerNode[_posU]; k++)
						{
							if (_graph->_nodes[_graph->_adjacentList[_posU][k]]._degreeCenter == 1)
							{
								if (_graph->_countLinkedPerNode[C[i]] == 0)
									_graph->_linked[C[i]] = (int*)malloc(sizeof(int));
								else
									_graph->_linked[C[i]] = (int*)realloc(_graph->_linked[C[i]], sizeof(int)*(_graph->_countLinkedPerNode[C[i]] + 1));

								//Asigno el vertice a la lista de linked de C[i]
								_graph->_linked[C[i]][_graph->_countLinkedPerNode[C[i]]] = _graph->_adjacentList[_posU][k];
								_graph->_countLinkedPerNode[C[i]]++;
							}
							_graph->_nodes[_graph->_adjacentList[_posU][k]]._degreeCenter--;
						}
					}
					else
						_analyzed[_posU] = true;
				}
			}
		}
	}
	free(_analyzed);
	/*ftime(&end);
	cout << (end.time - ini.time) * 1000 + (end.millitm - ini.millitm);
	cout << "\n";*/

	statexFinal.dwLength = sizeof(statexFinal);
 
	GlobalMemoryStatusEx(&statexFinal);
	
}

int main()
{
	statexInitial.dwLength = sizeof(statexInitial);
 
	GlobalMemoryStatusEx(&statexInitial);

	Read(__argv[1]);	

	//timeb ini, end;

	//ftime(&ini);	

	BuildGraph(atof(__argv[2]));	

	OClustR();

	//ftime(&end);
	
	//printf("%d\n", (end.time - ini.time) * 1000 + (end.millitm - ini.millitm));	

	WriteClusters();

	printf("%d\n", (statexInitial.ullAvailPhys - statexFinal.ullAvailPhys)/1024/1024);
		
	return 0;
}



 