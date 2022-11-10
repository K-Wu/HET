// code from
// http://www.shodor.org/media/content//petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
//@@ Example of grid and block configuration
//	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
// dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
// MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
// C = A * B (all are row-major matrices)

// Get a matrix element
__device__ __forceinline__ float& GetElement(float* A, int num_cols, int row,
                                             int col) {
  return A[row * num_cols + col];
}

template <int BLOCK_SIZE, bool ACornerTurnLoadFlag>
__global__ void MatMulKernel(float* A, float* B, float* C, int num_A_rows,
                             int num_A_cols, int num_B_cols) {
  // Block row and column
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  // Each thread block computes one sub-matrix Csub of C
  float* Csub = &C[blockRow * BLOCK_SIZE * num_B_cols + blockCol * BLOCK_SIZE];
  // Each thread computes one element of Csub
  // by accumulating results into Cvalue
  float Cvalue = 0.0f;
  // Thread row and column within Csub
  int row = threadIdx.y;
  int col = threadIdx.x;
  // Loop over all the sub-matrices of A and B that are
  // required to compute Csub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  for (int m = 0; m < (num_A_cols / BLOCK_SIZE); ++m) {
    // Get sub-matrix Asub of A
    float* Asub = &A[blockRow * num_A_cols * BLOCK_SIZE + m * BLOCK_SIZE];
    // Get sub-matrix Bsub of B
    float* Bsub = &B[m * num_B_cols * BLOCK_SIZE + blockCol * BLOCK_SIZE];
    // Shared memory used to store Asub and Bsub respectively
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    // Load Asub and Bsub from device memory to shared memory
    // Each thread loads one element of each sub-matrix
    if (ACornerTurnLoadFlag) {
      // The loading region and loaded layout in the shared memory is still the
      // same. It is just reassign tasks among threads to make sure the load is
      // coalesced in some cases
      As[col][row] = (row + blockRow * BLOCK_SIZE < num_A_rows &&
                      m * BLOCK_SIZE + col < num_A_cols)
                         ? GetElement(Asub, col, row)
                         : 0.0f;
    } else {
      As[row][col] = (row + blockRow * BLOCK_SIZE < num_A_rows &&
                      m * BLOCK_SIZE + col < num_A_cols)
                         ? GetElement(Asub, row, col)
                         : 0.0f;
    }

    Bs[row][col] = (m * BLOCK_SIZE + row < num_A_cols &&
                    blockCol * BLOCK_SIZE + col < num_B_cols)
                       ? GetElement(Bsub, row, col)
                       : 0.0f;
    // Synchronize to make sure the sub-matrices are loaded
    // before starting the computation
    __syncthreads();
    // Multiply Asub and Bsub together
    for (int e = 0; e < BLOCK_SIZE; ++e) Cvalue += As[row][e] * Bs[e][col];
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }
  // Write Csub to device memory
  // Each thread writes one element
  if (blockRow * BLOCK_SIZE + row < num_A_rows &&
      blockCol * BLOCK_SIZE + col < num_B_cols) {
    GetElement(Csub, row, col) = Cvalue;
  }
}