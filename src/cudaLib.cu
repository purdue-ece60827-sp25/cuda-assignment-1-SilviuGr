
#include "cudaLib.cuh"


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {

	//	Insert GPU SAXPY kernel code here

	int i = blockIdx.x*blockDim.x +threadIdx.x;

    //size is the size of the array
	if(i<size){

		y[i] = scale*x[i] + y[i];
		//**********please work baby! :) 
	}


}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";


    //memorysizeforthe float vector
	uint64_t vectorBytes = vectorSize*sizeof(float);

	//Allocate host memory
	

	float *h_A = new float[vectorSize];
	float *h_B = new float[vectorSize];

	float *h_C = new float[vectorSize];

	//std::vector<float> h_C(vectorSize);
	//std::vector<float> h_C(vectorSize);

    //fill vectors with random values
	vectorInit(h_A, vectorSize);
	vectorInit(h_B,vectorSize);
	//vectorInit()

    //Allocate device memory
	float *d_A, *d_B;

           ////error catching malloc device memory 
	cudaError_t err = cudaMalloc((void**)&d_A,vectorBytes);

	if(err != cudaSuccess){
		std::cerr << "failed cuda malloc on d_A" << cudaGetErrorString(err) <<std::endl;

		return -1;
	}

	        //error catching malloc device memory 
	err = cudaMalloc((void**)&d_B,vectorBytes);

	if(err!=cudaSuccess){
		std::cerr << "failed cuda malloc on d_B"<< cudaGetErrorString(err)<<std::endl;
		return -1;
	}

	//cudaMalloc((void**)&d_C,vectorBytes);

	cudaMemcpy(d_A,h_A,vectorBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,h_B,vectorBytes,cudaMemcpyHostToDevice);

	float scale = (float)(rand()%100);

	int numBlocks = (vectorSize+255)/256;

	
	saxpy_gpu<<<numBlocks, 256>>>(d_B,d_A,scale, vectorSize); ///////////////////////////////kernel

	cudaDeviceSynchronize();

	err = cudaGetLastError();

	if(err!=cudaSuccess){
		std::cerr<<"kernel launch failed"<< cudaGetErrorString(err)<<std::endl;
		return -1;
	}

	err = cudaMemcpy(h_C,d_A, vectorBytes,cudaMemcpyDeviceToHost);

	if(err!= cudaSuccess){
		std::cerr <<"failed cudaMemcpy on h_C"<< cudaGetErrorString(err)<<std::endl;
		return -1;
	}

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" a = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", h_A[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", h_B[i]);
		}
		printf(" ... }\n");
	#endif

	#ifndef DEBUG_PRINT_DISABLE 
		printf(" c = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", h_C[i]);
		}
		printf(" ... }\n");
	#endif




	int errorCount = verifyVector(h_B,h_A, h_C, scale, vectorSize);

	cudaFree(d_A);
	cudaFree(d_B);
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	std::cout<<"Error Count: ";
	std::cout<<errorCount<<std::endl;

	//	Insert code here

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code 

	int i = blockIdx.x*blockDim.x +threadIdx.x;

	float y;
	float x;

	curandState_t rng;

	curand_init(clock64(),i,0,&rng);

	uint64_t count = 0;

	if(i<pSumSize){

		

		for(uint64_t a =0; a<sampleSize; a++){

			y = curand_uniform(&rng);
			x = curand_uniform(&rng);
			
			if( int(x*x+y*y)==0){
				count++;
			}
		}

		pSums[i] = count;
	}

}
//void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t reduceSize) {

	//	Insert code here
	__shared__ uint64_t sharedMem[256];

    int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x +threadIdx.x;

	uint64_t hitSum = 0;

	uint64_t sizeofPsums = gridDim.x*blockDim.x*reduceSize;

	
    //summing up reduce size elements
	for(uint64_t k = 0; k<reduceSize; k++){

		uint64_t b = k+reduceSize*i;

		if(b< sizeofPsums){

			//to sum reduceSize points per thread
			hitSum += pSums[b];
		}
	}

	

	sharedMem[tid] = hitSum;
	__syncthreads();
    //sum it down until I have one value per
	for(uint64_t s = blockDim.x/2; s>0; s/=2){

		if(tid<s){

			sharedMem[tid] += sharedMem[tid+s];
		}

		__syncthreads();
	}
    

	if(tid==0){

		totals[blockIdx.x] = sharedMem[0];
	}

}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	if(reduceSize>(generateThreadCount*sampleSize)){
		std::cerr<<"reduceSize is greater than generateThreadCount, don't be silly"<<std::endl;
		return -1;
	}

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	std::cout<<"\n";
	std::cout<<"number of points calculated: "<<(generateThreadCount*sampleSize)<<std::endl;

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	int numBlocks = (generateThreadCount+255)/256;

    //make the size of the input to generatePoints
	//equal to the number of threads
	//each thread makes sampleSize points, sums
	//up the number of hits and stores it in the index
	//of the input vector corresponding to the thread
	uint64_t vectorSize1 = generateThreadCount;

	//this is the size of the reduced hit vector
	//this input is to the reduceCounts kernel
	//the indices of this vector hold the reduced 
	//sums of the hit points from the original vector
	uint64_t vectorSize2 = (reduceThreadCount+255)/256;

    //this is the bytesize of the hit vector for the first summing kernel
	uint64_t vectorBytes1 = vectorSize1*sizeof(uint64_t);

	//this is the bytesize of the reduced hit vector kernel
	//(second Kernel)
	//uint64_t vectorBytes2 = vectorSize2*sizeof(uint64_t);


	uint64_t *h_A = new uint64_t[vectorSize1];
    //to receive the output from kernel 2
	uint64_t *h_B = new uint64_t[vectorSize2];

    //first vector input for kernel1
	uint64_t *d_A;

    //second vector input for kernel2
	uint64_t *d_B;


	
    //allocating memory on device for d_A
	cudaError_t err = cudaMalloc((void**)&d_A,vectorBytes1);

	if(err != cudaSuccess){
		std::cerr << "failed cuda malloc on d_A" << cudaGetErrorString(err) <<std::endl;

		return -1;
	}
    
	//cudaMemcpy(d_A,h_A,vectorBytes1,cudaMemcpyHostToDevice);


    //(uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize)
	generatePoints<<<numBlocks, 256>>>(d_A,vectorSize1,sampleSize);

	cudaDeviceSynchronize();
/*
	cudaMemcpy(h_A,d_A,vectorBytes1,cudaMemcpyDeviceToHost);

	uint64_t checkCount = 0;

	for(uint64_t a = 0; a<vectorSize1; a++){

		checkCount += h_A[a];
	}

	float mooCount = 4.0f*( (double)checkCount)/(  (double)(generateThreadCount*sampleSize)  );
	std::cout<<mooCount<<" =>pi before reduction"<<std::endl;
*/
	

	 //changing numBlocks to reflect the new thread requirement
	//for the second kernel
	numBlocks = (reduceThreadCount+255)/256;

   err = cudaMalloc((void**)&d_B,numBlocks*sizeof(uint64_t));

	if(err != cudaSuccess){
		std::cerr << "failed cuda malloc on d_B" << cudaGetErrorString(err) <<std::endl;

		return -1;
	}


	reduceCounts<<<numBlocks, 256>>>(d_A,d_B,reduceSize);

    //not sure if needed but wanted to be safe
	cudaDeviceSynchronize();

	cudaMemcpy(h_B,d_B,numBlocks*sizeof(uint64_t),cudaMemcpyDeviceToHost);

	uint64_t hitSum = 0;

	for(uint64_t k =0; k<numBlocks; k++){

		hitSum += h_B[k];

	}

	approxPi = 4.0f*( (double)hitSum)/(  (double)(generateThreadCount*sampleSize)  );


    cudaFree(d_A);
	cudaFree(d_B);
	delete[] h_B;


	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}
