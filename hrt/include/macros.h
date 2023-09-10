#pragma once

#if CUDA_ARCHS == 86  // RTX 3090
// TODO: use the reg tiling flag instead of equal_1 checking
#define MY_SGEMM_LAUNCH_BOUNDS                       \
  __launch_bounds__(RIGHT_REG_TILED_FLAG ? 64 : 256, \
                    RIGHT_REG_TILED_FLAG ? 6 : 3)
#define NO_SCATTER_GATHER_LAUNCH_BOUNDS              \
  __launch_bounds__(RIGHT_REG_TILED_FLAG ? 64 : 256, \
                    RIGHT_REG_TILED_FLAG ? 5 : 3)

#define MY_SGEMM_GRID_CONFIG(SUFFIX)                             \
  constexpr int WORK_BLOCK_SIZE##SUFFIX##_X =                    \
      REG_TILING_FLAG##SUFFIX ? 32 : 32;                         \
  constexpr int WORK_BLOCK_SIZE##SUFFIX##_Y =                    \
      REG_TILING_FLAG##SUFFIX ? 32 : 32;                         \
  constexpr int WORK_BLOCK_SIZE##SUFFIX##_K =                    \
      REG_TILING_FLAG##SUFFIX ? 8 : 32;                          \
  constexpr int THREADING_BLOCK_SIZE##SUFFIX##_X =               \
      REG_TILING_FLAG##SUFFIX ? WORK_BLOCK_SIZE##SUFFIX##_X      \
                              : WORK_BLOCK_SIZE##SUFFIX##_X / 2; \
  constexpr int THREADING_BLOCK_SIZE##SUFFIX##_Y =               \
      REG_TILING_FLAG##SUFFIX ? 2 : WORK_BLOCK_SIZE##SUFFIX##_Y / 2;
// TODO: specify thread factor larger than 1

#elif CUDA_ARCHS == 89  // L4
// TODO: use the reg tiling flag instead of equal_1 checking
#define MY_SGEMM_LAUNCH_BOUNDS                       \
  __launch_bounds__(RIGHT_REG_TILED_FLAG ? 64 : 256, \
                    RIGHT_REG_TILED_FLAG ? 6 : 3)
#define NO_SCATTER_GATHER_LAUNCH_BOUNDS              \
  __launch_bounds__(RIGHT_REG_TILED_FLAG ? 64 : 256, \
                    RIGHT_REG_TILED_FLAG ? 5 : 3)

#define MY_SGEMM_GRID_CONFIG(SUFFIX)                             \
  constexpr int WORK_BLOCK_SIZE##SUFFIX##_X =                    \
      REG_TILING_FLAG##SUFFIX ? 32 : 32;                         \
  constexpr int WORK_BLOCK_SIZE##SUFFIX##_Y =                    \
      REG_TILING_FLAG##SUFFIX ? 32 : 32;                         \
  constexpr int WORK_BLOCK_SIZE##SUFFIX##_K =                    \
      REG_TILING_FLAG##SUFFIX ? 8 : 32;                          \
  constexpr int THREADING_BLOCK_SIZE##SUFFIX##_X =               \
      REG_TILING_FLAG##SUFFIX ? WORK_BLOCK_SIZE##SUFFIX##_X      \
                              : WORK_BLOCK_SIZE##SUFFIX##_X / 2; \
  constexpr int THREADING_BLOCK_SIZE##SUFFIX##_Y =               \
      REG_TILING_FLAG##SUFFIX ? 2 : WORK_BLOCK_SIZE##SUFFIX##_Y / 2;
// TODO: specify thread factor larger than 1

#elif CUDA_ARCHS == 80  // A100
// TODO: use the reg tiling flag instead of equal_1 checking

#define MY_SGEMM_LAUNCH_BOUNDS                       \
  __launch_bounds__(RIGHT_REG_TILED_FLAG ? 64 : 256, \
                    RIGHT_REG_TILED_FLAG ? 8 : 3)
#define NO_SCATTER_GATHER_LAUNCH_BOUNDS              \
  __launch_bounds__(RIGHT_REG_TILED_FLAG ? 64 : 256, \
                    RIGHT_REG_TILED_FLAG ? 8 : 3)

#define MY_SGEMM_GRID_CONFIG(SUFFIX)                             \
  constexpr int WORK_BLOCK_SIZE##SUFFIX##_X =                    \
      REG_TILING_FLAG##SUFFIX ? 32 : 32;                         \
  constexpr int WORK_BLOCK_SIZE##SUFFIX##_Y =                    \
      REG_TILING_FLAG##SUFFIX ? 32 : 32;                         \
  constexpr int WORK_BLOCK_SIZE##SUFFIX##_K =                    \
      REG_TILING_FLAG##SUFFIX ? 8 : 32;                          \
  constexpr int THREADING_BLOCK_SIZE##SUFFIX##_X =               \
      REG_TILING_FLAG##SUFFIX ? WORK_BLOCK_SIZE##SUFFIX##_X      \
                              : WORK_BLOCK_SIZE##SUFFIX##_X / 2; \
  constexpr int THREADING_BLOCK_SIZE##SUFFIX##_Y =               \
      REG_TILING_FLAG##SUFFIX ? 2 : WORK_BLOCK_SIZE##SUFFIX##_Y / 2;
// TODO: specify thread factor larger than 1

#endif
