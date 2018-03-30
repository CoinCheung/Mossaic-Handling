#include"logging.h"
#include<string>
#include<iostream>


log log::logging()
{
    log obj;
    return obj;
}


log log::operator<<(std::string str)
{
    std::cout << str;
    return *this;
}


log log::operator<<(int num)
{
    std::cout << num;
    return *this;
}


log log::operator<<(const char* message)
{
    std::cout << message;
    return *this;
}


log log::operator<<(std::ostream& (*op)(std::ostream&))
{
    op(std::cout);
    return *this;
}

log::~log()
{
    std::cout << std::endl;
    abort();
}
