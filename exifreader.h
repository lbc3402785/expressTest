#ifndef EXIFREADER_H
#define EXIFREADER_H
#include <exiv2/exiv2.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
class ExifReader
{
public:

    static ExifReader& getInstance(){
        static ExifReader instance;
        Exiv2::XmpParser::initialize();
        ::atexit(Exiv2::XmpParser::terminate);
        return  instance;
    }
    bool getFocalLength(std::string picPath,cv::Mat& image, double &result);
private:
     ExifReader();
};

#endif // EXIFREADER_H
