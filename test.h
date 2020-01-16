#ifndef TEST_H
#define TEST_H
#include <string>

class Test
{
public:
    Test();
    static void testG8M(std::string picPath);
    static void testPerspective(std::string picPath);
    static void test();
};

#endif // TEST_H
