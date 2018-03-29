#include<iostream>
#include<opencv2/opencv.hpp>


void add_mask();


int main()
{
    add_mask();
    return 0;
}


void add_mask()
{
    using namespace std;
    using namespace cv;

    Mat heatmap = imread("heatmap.jpg", -1);
}

