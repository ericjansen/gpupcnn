//////////////////////////////////////////////
// GPU-BASED PULSE-COUPLED NEURAL NETWORK FOR
// CLASSIFYING RETINOPATHY AND MACULAR EDEMA
// developer : ERIC JANSEN
// e-mail : eric[at]jansen[dot]net
// http://www.ericjansen.net
// ONLY WORKING UNDER LINUX
//////////////////////////////////////////////
/*
Copyright (c) 2012, Computer Engineering and Telematics,
Dept. of Electrical Engineering, Institut Teknologi Sepuluh Nopember (ITS)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software
   must display the following acknowledgement:
   This product includes software developed by Computer Engineering and
   Telematics, Dept. of Electrical Engineering, Institut Teknologi Sepuluh
   Nopember.
4. Neither the name of Institut Teknologi Sepuluh Nopember (ITS) nor the
   names of its contributors may be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY ITS ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL [ERIC JANSEN] BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _GPUPCNNKERNEL_CUH_
#define _GPUPCNNKERNEL_CUH_

#include <opencv2/gpu/devmem2d.hpp>
#include <cmath>

void callGPUProcessS2(cv::gpu::DevMem2D_<float> S,
	cv::gpu::DevMem2D_<float> S2);

void callGPUProcessE(cv::gpu::DevMem2D_<float> E);

void callGPUKernel(const cv::gpu::DevMem2D_<float>& F,
	const cv::gpu::DevMem2D_<float>& L,
	const cv::gpu::DevMem2D_<float>& E,
	const cv::gpu::DevMem2D_<float>& S2,
	const cv::gpu::DevMem2D_<unsigned char>& Sum1,
	cv::gpu::PtrStep Y,
	const float& dAF,const float& dAL,const float& dAE,
	const float& dVF,const float& dVE,const float& dB,
  const int& R,const int& C);

#endif
