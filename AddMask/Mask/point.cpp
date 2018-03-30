#include"point.h"
#include<iostream>


point::point(): rpos(0), cpos(0), val(0)
{
}


point::point(int r, int c, unsigned char v): rpos(r), cpos(c), val(v)
{
}


point::point(point &p)
{
    rpos = p.rpos;
    cpos = p.cpos;
    val = p.val;
}

point::point(point const &p)
{
    rpos = p.rpos;
    cpos = p.cpos;
    val = p.val;
}


point::point(point &&p)
{
    rpos = p.rpos;
    cpos = p.cpos;
    val = p.val;
}


point& point::operator=(point p)
{
    rpos = p.rpos;
    cpos = p.cpos;
    val = p.val;
    return *this;
}

