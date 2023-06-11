#pragma once

#if CUDA_ARCHS == 86  // RTX 3090

#elif CUDA_ARCHS == 80  // A100

#endif

#define MY_SGEMM_LAUNCH_BOUNDS                              \
  __launch_bounds__(THREADING_BLOCK_SIZE_Y == 1 ? 64 : 256, \
                    THREADING_BLOCK_SIZE_Y == 1 ? 12 : 3)

#define MY_SGEMM_GRID_CONFIG(SUFFIX)                             \
  constexpr int WORK_BLOCK_SIZE##SUFFIX##_X =                    \
      REG_TILING_FLAG##SUFFIX ? 64 : 32;                         \
  constexpr int WORK_BLOCK_SIZE##SUFFIX##_Y =                    \
      REG_TILING_FLAG##SUFFIX ? 16 : 32;                         \
  constexpr int WORK_BLOCK_SIZE##SUFFIX##_K =                    \
      REG_TILING_FLAG##SUFFIX ? 8 : 32;                          \
  constexpr int THREADING_BLOCK_SIZE##SUFFIX##_X =               \
      REG_TILING_FLAG##SUFFIX ? WORK_BLOCK_SIZE##SUFFIX##_X      \
                              : WORK_BLOCK_SIZE##SUFFIX##_X / 2; \
  constexpr int THREADING_BLOCK_SIZE##SUFFIX##_Y =               \
      REG_TILING_FLAG##SUFFIX ? 1 : WORK_BLOCK_SIZE##SUFFIX##_Y / 2;
