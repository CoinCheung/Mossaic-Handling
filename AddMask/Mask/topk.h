#ifndef _TOPK_H_
#define _TOPK_H_

#include<vector>
#include"point.h"


class TopkHeap
{
    public:
        std::vector<point> heap;
        long len;

        TopkHeap(long);
        void add(point);
        void adjust();
        void sink_node(long);
        point get_root();
        unsigned char get_root_val();
        void set_root(point);
        void sink_root();

        void print();

};



#endif
