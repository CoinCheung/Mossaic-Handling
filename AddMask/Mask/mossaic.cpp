#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>
#include<thread>
#include<mutex>
#include<fstream>
#include<string>
#include"topk.h"
#include"point.h"
#include"logging.h"
#include"mossaic.h"


/* global variables */
std::mutex file_mut;
std::mutex cout_mut;


/* definitions */
cv::Mat gen_mask(cv::Mat&, float ratio);
cv::Mat merge_mask(cv::Mat&, cv::Mat&);
cv::Mat concat_horizontal(cv::Mat left, cv::Mat right);
void gen_AB_thread(std::ifstream&, float);

void try_something();

#ifdef MAIN
int main()
{
    const char* org_pic = "./pics/batch1-42_org.jpg";
    const char* heatmap = "./pics/batch1-42_hm.jpg";
    const char* masked_pic = "./pics/batch1-42_merge.jpg";
    const char* concated_pic = "./pics/batch1-42_concat.jpg";
    // add_mossaic(org_pic, heatmap, masked_pic, 0.45);
    gen_AB_image(org_pic, heatmap, concated_pic, 0.45);
    generate_AB_dataset("./pics/filenames.txt", 0.45);
    // try_something();
    return 0;
}
#endif



/* methods to be exported as lib interfaces */
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



void gen_AB_image(const char* org, const char* heat, const char* out, float ratio)
{
    using namespace std;
    using namespace cv;

    Mat org_pic = imread(org, -1);
    Mat heatmap = imread(heat, -1);
    CHECK(!org_pic.empty()) << "image orginal does not exist!" << endl;
    CHECK(!heatmap.empty()) << "image heatmap does not exist!" << endl;

    Mat mask = gen_mask(heatmap, ratio);
    Mat merge = merge_mask(org_pic, mask);
    Mat concat = concat_horizontal(org_pic, merge);
    imwrite(out, concat);

}


void generate_AB_dataset(const char* filename, float ratio)
{
    using namespace std;

    ifstream fin(filename);

    CHECK(fin) << "Cannot find index file" << endl;
 
    thread t1(gen_AB_thread, std::ref(fin), ratio);
    thread t2(gen_AB_thread, std::ref(fin), ratio);
    thread t3(gen_AB_thread, std::ref(fin), ratio);
    thread t4(gen_AB_thread, std::ref(fin), ratio);

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    fin.close();
}



/* inner methods */
void gen_AB_thread(std::ifstream &fin, float ratio)
{
    using namespace std;

    string str_org;
    string str_hm;
    string str_out;


    while (true)
    {

        {
            lock_guard<mutex> locker(file_mut);
            if (!(getline(fin, str_org) && getline(fin, str_hm)))
                break;
        }

        str_out.assign(str_org);

        string::size_type pos{str_out.find("_org")};
        if (pos != string::npos)
        {
            str_out.replace(pos, 4, "_concat");
        }
        else 
        {
            pos = str_out.find("_hm"); 
            CHECK(pos != string::npos) << "image names error, no image name with _org or _hm exists" << endl;
            str_out.replace(pos, 3, "_concat");
        }
        
        gen_AB_image(str_org.c_str(), str_hm.c_str(), str_out.c_str(), ratio);
    }
}


cv::Mat concat_horizontal(cv::Mat left, cv::Mat right)
{
    using namespace cv;
    using namespace std;

    int rows{left.rows};
    int cols{left.cols};
    int channels{left.channels()};

    CHECK(rows == right.rows) << "two images should have same height" << endl;
    // CHECK(cols == right.cols) << "two images should have same width" << endl;
    CHECK(channels == right.channels()) << "two images should have same channels" << endl;

    Mat res(rows, cols<<1, CV_8UC3);
    Mat roi;

    roi = res(Rect(0,0,cols,rows));
    left.copyTo(roi);
    roi = res(Rect(cols,0,cols,rows));
    right.copyTo(roi);

    return res;
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
    Mat org{org_img.clone()};
    Mat grid;
    Rect rect;
    Scalar color;

    int msize{10};
    int cstep{channels * msize};


    for (int i{0}; i < rows - msize; i += msize)
    {
        sp = org.ptr<unsigned char>(i);
        dp = mask.ptr<unsigned char>(i);

        for (int js{0}, jd{0}; jd < cols - msize; js+=cstep, jd+=msize)
        {
            if (dp[jd] == 255)
            {
                color = Scalar(sp[js], sp[js+1], sp[js+2]);
                rect = Rect(jd, i, msize, msize);

                grid = org(rect);
                Mat(rect.size(), type, color).copyTo(grid);
            }
        }
    }

    return org;
}


void try_something()
{
    using namespace std;

    int a[10];
    int i{0};
    for (auto& el:a)
        el = i++;

    // std::cout << std::end(a) << endl;

    std::string ab("fdadf,fdagac:");
    
    //
    vector<string> v;
    string token("da"); 
    string::size_type pos1, pos2;

    pos1 = 0;
    pos2 = ab.find(token);
    while (pos2 != string::npos)
    {
        v.push_back(ab.substr(pos1, pos2-pos1));
        pos1 = pos2 + token.size();
        pos2 = ab.find(token, pos1);
    }
    if (pos1 != ab.length())
        v.push_back(ab.substr(pos1));

    for (auto& el:v)
        cout << el << endl;
}
