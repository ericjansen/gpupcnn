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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <sys/types.h>
#include <dirent.h>
#include "gpupcnnkernel.cuh"

/////////////////////////////////////////////////////////
// [USAGE]
// ./gpupcnn [PATH] [iteration] [dAF] [dAL] [FileNo] [-c]
///////////////////////////////////////////////////////// 

using namespace std;

#define N 6

int main(int argc,char** argv)
{
    std::stringstream strPath,strCropped;

    if (strcmp(argv[1],"") != 0)
      strPath << argv[1];
    else strPath << "./";

    DIR *dp;
    struct dirent *ep;
    dp = opendir(strPath.str().c_str());

    int iN,iPar;
    float dAF,dAL;

    if (strcmp(argv[2],"") == 0) iN = 15;
    else iN = atoi(argv[2]);

    if (strcmp(argv[3],"") == 0)
      dAF = 0.75;
    else dAF = (float)atof(argv[3]);

    if (strcmp(argv[4],"") == 0)
      dAL = 0.75;
    else dAL = (float)atof(argv[4]);

    if (strcmp(argv[5],"") == 0)
      iPar = 0;
    else iPar = atoi(argv[5]);

    if (strcmp(argv[6],"") != 0)
      strCropped << argv[6];


    if (dp != NULL) {

      while (ep = readdir(dp)) {

        if (ep->d_type == DT_REG) {

          float k[3][3] = { 0.5,1.0,0.5,
                            1.0,1.0,1.0,
                            0.5,1.0,0.5};
          float dAE = 0.069;
          float dVF = 0.01;
          float dVL = 1;
          float dVE = 0.2;
          float dB = 0.2;

          std::ofstream fmoto;

          std::stringstream strFileName;
          strFileName << strPath.str() << ep->d_name;

          cv::Mat img = cv::imread(strFileName.str());
          cv::Mat gray = cv::Mat(img.size(),CV_8UC1);
          cv::cvtColor(img,gray,CV_BGR2GRAY);

          int nRow = img.rows;
          int nCol = img.cols;

          cv::gpu::GpuMat gGray;

          gGray.upload(gray);

          cv::gpu::GpuMat gSum1(gGray.size(),CV_8UC1);

          cv::Mat km = cv::Mat(3,3,CV_32F,k);

          cv::gpu::GpuMat gS(gGray.size(),CV_8UC1);
          cv::gpu::GpuMat gS2(gGray.size(),CV_32FC1);
          cv::gpu::GpuMat gF(gGray.size(),CV_32FC1);
          cv::gpu::GpuMat gL(gGray.size(),CV_32FC1);
          cv::gpu::GpuMat gE(gGray.size(),CV_32FC1);

          cv::Mat Y = cv::Mat::zeros(img.size(),CV_8UC1);

          cv::gpu::GpuMat gY(gGray.size(),CV_8U);
          gY.upload(Y);

          cv::Mat Sum1 = cv::Mat(gGray.size(),CV_8UC1);
          cv::Mat S = cv::Mat(gGray.size(),CV_8UC1);
          cv::Mat S2 = cv::Mat(gGray.size(),CV_32FC1);
          cv::Mat F = cv::Mat(gGray.size(),CV_32FC1);
          cv::Mat L = cv::Mat(gGray.size(),CV_32FC1);
          cv::Mat E = cv::Mat(gGray.size(),CV_32FC1);
          gray.copyTo(S);

          for (int y=0; y<S2.rows; ++y)
            for (int x=0; x<S2.cols; ++x)
              S2.at<float>(y,x) = S.at<uchar>(y,x)/255.0;

          for (int y=0; y<E.rows; ++y)
            for (int x=0; x<E.cols; ++x)
              E.at<float>(y,x) = 2.0;
  

          gF.upload(F);
          gL.upload(L);
          gS2.upload(S2);
          gE.upload(E);
          gSum1.upload(Sum1);
          gY.upload(Y);

          gS2.copyTo(gF);
          gS2.copyTo(gL);

          clock_t start = clock();

          for (int n=0; n<iN; ++n) {
            cv::gpu::filter2D(gY,gSum1,gSum1.depth(),km);
            callGPUKernel(gF,gL,gE,gS2,gSum1,gY,dAF,dAL,dAE,dVF,dVE,dB,nRow,nCol);
          }

          clock_t end = clock();

          float time = ((float)end - (float)start)/CLOCKS_PER_SEC;
          float flp = float(2*((img.rows*img.cols*img.rows)/time)/1000000);

          std::cout << "Total time = " << time << "Seconds\n";
          std::cout << "Total MFLOPS = " << flp << "MFLOPS\n";

          fmoto.open(strPath.str().c_str(),std::ios::out | std::ios::app |
              std::ios::in);

          fmoto << "Out_" << iPar << "_" << ep->d_name << "," << 
            img.cols << "x" << img.rows << "," << dAF << "," << dAL << 
            "," << time << "s," << flp << "MFLOPS,N " << iN << "\n";

          gY.download(Y);

          stringstream strOut;
          strOut << strPath.str() << "Res/" << ep->d_name << "_it" << 
            iN << "_" << iPar << ".pgm";

          cv::imwrite(strOut.str(),Y);
          fmoto.close();

          if (strCropped.str() == std::string("-c")) {
            cv::Mat cimg = cv::imread(strOut.str());
            cv::cvtColor(cimg,cimg,CV_BGR2GRAY);
            cv::Rect cropped(255,40,910,910);
            cv::Mat res(cimg,cropped);

            cv::Mat out = res.clone();

            std::stringstream strCOut;
            strCOut << strPath.str() << "cropped/" << ep->d_name <<
              "it" << iN << "_" << 
              iPar << ".pgm";
            cv::imwrite(strCOut.str(),out);
          }
        }        
      }
      (void) closedir(dp);
    } else
      perror("Couldn't open the directory");

    return 0;
}
