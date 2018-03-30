#include<vector>
#include"point.h"
#include"topk.h"
#include"logging.h"
#include<iostream>





TopkHeap::TopkHeap(long num)
{
    heap.reserve(num);
    len = 0;
}



void TopkHeap::add(point p)
{
    heap.push_back(p);
    len ++;
}



void TopkHeap::adjust()
{
    for (long i{(len-1)>>1}; i >= 0; i--)
        sink_node(i);
}


void TopkHeap::sink_node(long ind)
{
    using namespace std;
    long child1;
    long child2;
    long min;

    while (true)
    {
        // get minimal child of current node
        child1 = (ind<<1)+1;
        child2 = (ind<<1)+2;

        if (child2 < len)
            min = heap[child1].val < heap[child2].val ? child1 : child2;
        else if (child1 < len)
            min = child1;
        else
            break;

        if (heap[min].val < heap[ind].val)
        {
            swap(heap[min], heap[ind]);
            ind = min;
        }
        else
            break;
    }
}


point TopkHeap::get_root()
{
    CHECK(len > 0) << "the length of Topk algorithm heap should be greater than 0" 
        << std::endl;
    return heap[0];
}


unsigned char TopkHeap::get_root_val()
{
    CHECK(len > 0) << "the length of Topk algorithm heap should be greater than 0" 
        << std::endl;
    return heap[0].val;
}


void TopkHeap::set_root(point p)
{
    if (len == 0)
        heap.push_back(p);
    else
        heap[0] = p;
}


void TopkHeap::sink_root()
{
    sink_node(0);
}


void TopkHeap::print()
{
    for (auto& p:heap)
        std::cout << (int)p.val << ", ";
    std::cout << std::endl;
}

