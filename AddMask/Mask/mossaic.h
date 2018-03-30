#ifndef _MOSSAIC_H_
#define _MOSSAIC_H_

#include<opencv2/opencv.hpp>


extern "C"
{
void add_mossaic(const char* org, const char* heat, const char* out, float ratio);
}


cv::Mat gen_mask(cv::Mat&, float ratio);
cv::Mat merge_mask(cv::Mat&, cv::Mat&);


#endif
