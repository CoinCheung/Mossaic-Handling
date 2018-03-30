#ifndef _POINT_H_
#define _POINT_H_

class point
{
    public:
        int rpos;
        int cpos;
        unsigned char val;

        point();
        point(int r, int c, unsigned char v);
        point( point &);
        point(const point &);
        point(point &&);
        point& operator=(point p);
};


#endif
