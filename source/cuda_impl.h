#pragma once

void furthest_point_sampling_kernel_wrapper( int b, int n, int m, const float* dataset, float* temp, int64_t* idxs,
                                             cudaStream_t stream );
