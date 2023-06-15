#pragma once

#if CUDA_ARCHS == 86  // RTX 3090
#define MY_SGEMM_LAUNCH_BOUNDS                        \
  __launch_bounds__(RIGHT_REG_TILED_FLAG ? 128 : 256, \
                    RIGHT_REG_TILED_FLAG ? 6 : 3)

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
      REG_TILING_FLAG##SUFFIX ? 4 : WORK_BLOCK_SIZE##SUFFIX##_Y / 2;

#elif CUDA_ARCHS == 80  // A100
#define MY_SGEMM_LAUNCH_BOUNDS                        \
  __launch_bounds__(RIGHT_REG_TILED_FLAG ? 128 : 256, \
                    RIGHT_REG_TILED_FLAG ? 6 : 3)

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
      REG_TILING_FLAG##SUFFIX ? 4 : WORK_BLOCK_SIZE##SUFFIX##_Y / 2;

#endif
