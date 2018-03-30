#ifndef _LOGGING_H_
#define _LOGGING_H_

#include<iostream>
#include<sstream>


#define CHECK(x)  \
    if (!(x))       \
    log::logging() << __FILE__ << ": " << __LINE__ << ": Check failed: \"" #x \
    << "\": "


class log
{
    public:

        static log logging();
        log operator<<(std::string);
        log operator<<(int);
        log operator<<(const char*);
        log operator<<(std::ostream& (*op)(std::ostream&));

        ~log();
};

#endif
