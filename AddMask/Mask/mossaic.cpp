#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>
#include"topk.h"
#include"point.h"
#include"logging.h"
#include"mossaic.h"




// int main()
// {
//     const char* org_pic = "./pics/batch1-42_org.jpg";
//     const char* heatmap = "./pics/batch1-42_hm.jpg";
//     const char* masked_pic = "./pics/batch1-42_merge.jpg";
//     add_mossaic(org_pic, heatmap, masked_pic, 0.45);
//     // try_something();
//     return 0;
// }


void add_mossaic(const char* org, const char* heat, const char* out, float ratio)
{
    using namespace std;
    using namespace cv;

    Mat org_pic = imread(org, -1);
    Mat heatmap = imread(heat, -1);
    CHECK(!org_pic.empty()) << "image orginal does not exist!" << endl;
    CHECK(!heatmap.empty()) << "image heatmap does not exist!" << endl;

    Mat mask = gen_mask(heatmap, ratio);
    Mat merge = merge_mask(org_pic, mask);
    imwrite(out, merge);
}


/* generate mask Mat for mossaic location */
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


cv::Mat merge_mask(cv::Mat& org_img, cv::Mat& mask)
{
    using namespace std;
    using namespace cv;

    int rows{org_img.rows};
    int cols{org_img.cols};
    int channels{org_img.channels()};

    CHECK((rows == mask.rows) && (cols == mask.cols))
        << "original image and mask should have same size" << endl;

    auto type{org_img.type()};
    unsigned char *sp{nullptr};
    unsigned char *dp{nullptr};
    Rect rect;
    Mat grid;
    Scalar color;

    int msize{10};
    int cstep{channels * msize};


    for (int i{0}; i < rows - msize; i += msize)
    {
        sp = org_img.ptr<unsigned char>(i);
        dp = mask.ptr<unsigned char>(i);

        for (int js{0}, jd{0}; jd < cols - msize; js+=cstep, jd+=msize)
        {
            if (dp[jd] == 255)
            {
                color = Scalar(sp[js], sp[js+1], sp[js+2]);
                rect = Rect(jd, i, msize, msize);

                grid = org_img(rect);
                Mat(rect.size(), type, color).copyTo(grid);
            }
        }
    }

    return org_img;
}


