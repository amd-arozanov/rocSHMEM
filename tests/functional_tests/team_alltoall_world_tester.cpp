/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

// This file is included by team_alltoall_world_tester.hpp
// The kernel TeamAlltoallWorldTest is defined in team_alltoall_tester.cpp
// and will be available through linking

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
template <typename T1>
TeamAlltoallWorldTester<T1>::TeamAlltoallWorldTester(TesterArguments args)
    : TeamAlltoallTester<T1>(args) {
  // Allocate device memory for ROCSHMEM_TEAM_WORLD
  // We need to copy it to device memory since it's a host variable
  CHECK_HIP(hipMalloc(&team_world_device, sizeof(rocshmem_team_t)));

  // ROCSHMEM_TEAM_WORLD contains a device pointer, so we copy the pointer value
  // to device memory so kernel can access it
  // hipMemcpy with hipMemcpyHostToDevice is synchronous, so no need for sync
  CHECK_HIP(hipMemcpy(team_world_device, &ROCSHMEM_TEAM_WORLD,
                      sizeof(rocshmem_team_t), hipMemcpyHostToDevice));
  
  // Warn if n_pes is less than 256, as the bug may not manifest
  if (this->n_pes < 256) {
    std::cerr << "Warning: Testing with ROCSHMEM_TEAM_WORLD directly. "
              << "This test is designed to catch a bug that manifests "
              << "when n_pes >= 256. Current n_pes = " << this->n_pes << std::endl;
  }
}

template <typename T1>
TeamAlltoallWorldTester<T1>::~TeamAlltoallWorldTester() {
  if (team_world_device != nullptr) {
    CHECK_HIP(hipFree(team_world_device));
  }
}

template <typename T1>
void TeamAlltoallWorldTester<T1>::preLaunchKernel() {
  // No need to create split teams - we use ROCSHMEM_TEAM_WORLD directly
  this->bw_factor = this->n_pes;

  // ROCSHMEM_TEAM_WORLD was copied in constructor, no need to copy again
  // The value in device memory should remain valid
}

template <typename T1>
void TeamAlltoallWorldTester<T1>::launchKernel(dim3 gridSize, dim3 blockSize,
                                               int loop, size_t size) {
  size_t shared_bytes = 0;
  int num_elems = size / sizeof(T1);

  // Use ROCSHMEM_TEAM_WORLD directly - this tests the bug where
  // alltoall_pSync_pool is allocated with wrong size
  // Pass team_world_device pointer - kernel will dereference it to get team
  // value
  hipLaunchKernelGGL(TeamAlltoallWorldTest<T1>, gridSize, blockSize,
                     shared_bytes, this->stream, loop, this->args.skip,
                     this->start_time, this->end_time, this->source_buf,
                     this->dest_buf, num_elems, this->_shmem_context,
                     team_world_device);

  this->num_msgs = (loop + this->args.skip) * gridSize.x;
  this->num_timed_msgs = loop * gridSize.x;
}

template <typename T1>
void TeamAlltoallWorldTester<T1>::postLaunchKernel() {
  // No teams to destroy - we use ROCSHMEM_TEAM_WORLD directly
}

// Explicit template instantiations
template class TeamAlltoallWorldTester<int>;
template class TeamAlltoallWorldTester<long>;
template class TeamAlltoallWorldTester<long long>;

