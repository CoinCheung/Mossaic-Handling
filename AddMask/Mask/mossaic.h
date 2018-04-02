#ifndef _MOSSAIC_H_
#define _MOSSAIC_H_

#include<opencv2/opencv.hpp>


extern "C"
{
void add_mossaic(const char* org, const char* heat, const char* out, float ratio);
void generate_AB_dataset(const char*, float);
void gen_AB_image(const char* org, const char* heat, const char* out, float ratio);
}



#endif
