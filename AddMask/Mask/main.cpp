#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>
#include<memory>
#include"topk.h"
#include"point.h"
#include"logging.h"



void add_mask(float ratio);
cv::Mat gen_mask(cv::Mat&, float ratio);
cv::Mat merge_mask(cv::Mat&, cv::Mat&);

void try_something();

int main()
{
    add_mask(0.45);
    // try_something();
    return 0;
}


void add_mask(float ratio)
{
    using namespace std;
    using namespace cv;

    Mat org_pic = imread("./pics/batch0-46_org.jpg", -1);
    Mat heatmap = imread("./pics/batch0-46_hm.jpg", -1);
    CHECK(!org_pic.empty()) << "image orginal does not exist!" << endl;
    CHECK(!heatmap.empty()) << "image heatmap does not exist!" << endl;
    Mat mask = gen_mask(heatmap, ratio);
    imwrite("./pics/batch0-46_mask.jpg", mask);
    // Mat merge = merge_mask(org_pic, heatmap);
    // imwrite("batch0-46_merge.jpg", merge);
}


cv::Mat gen_mask(cv::Mat& heatmap, float ratio)
{
    using namespace cv;
    using namespace std;

    CHECK(heatmap.channels() == 1) << "heatmap channel number must be 1" << endl;

    int rows{heatmap.rows};
    int cols{heatmap.cols};
    Mat res(rows, cols, CV_8UC1, Scalar::all(0));
    unsigned char *hp{nullptr};
    long mask_num{static_cast<long>(ratio*rows*cols)};
    TopkHeap th(mask_num);
    int i{0};
    int j{0};
    int count{0};
    point pt;


    // fill true with initial points
    for (; i < rows; i++)
    {
        j = 0;
        hp = heatmap.ptr<unsigned char>(i);
        for (; j < cols; j++, count++)
        {
            if (count == mask_num)
                break;
            th.add(point(i,j,hp[j]));
        }    

        if (count == mask_num)
            break;
    }

    // construct heap
    th.adjust();

    // traverse the remaining pixels
    unsigned char root_val;
    for (; i < rows; i++)
    {
        hp = heatmap.ptr<unsigned char>(i);
        for (; j < cols; j++)
        {
            root_val = th.get_root_val();
            if (hp[j] > root_val)
            {
                th.set_root(point(i, j, hp[j]));
                th.sink_root();
            }
        }
        j = 0;
   }

    // assign values to mask matrix
    count = 0;
    for (point& p:th.heap)
    {
        res.ptr<unsigned char>(p.rpos)[p.cpos] = 255;
    }


    return res;
}


cv::Mat merge_mask(cv::Mat&);



void try_something()
{
    TopkHeap th(10);
    for (int i{9}; i >= 0; i--)
    // for (int i{0}; i < 10; i++)
        th.add(point(0,0,i));

    th.print();
    th.adjust();
    th.print();

}
