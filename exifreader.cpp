#include "exifreader.h"
#include <exception>
const double diag24x36 = std::sqrt(36.0 * 36.0 + 24.0 * 24.0);
double getMetadataFocalLength(Exiv2::ExifData& exifData){
    Exiv2::Exifdatum& FocalLengthDatum=exifData["Exif.Photo.FocalLength"];
    if(FocalLengthDatum.size()>0){
        return std::stod(FocalLengthDatum.toString());
    }
    return -1;
}
bool hasDigitMetadata(Exiv2::ExifData& exifData,std::string key){
    Exiv2::Exifdatum& FocalLengthIn35mmFilmDatum=exifData[key];
    return FocalLengthIn35mmFilmDatum.size()>0;
}
std::string getMetadata(Exiv2::ExifData& exifData,std::string key){
    return exifData[key].toString();
}
bool ExifReader::getFocalLength(std::string picPath,cv::Mat& out, double &result)
{
    out=cv::imread(picPath);
    if(!out.empty()){
        Exiv2::Image::UniquePtr image=0;
        try {
           image = Exiv2::ImageFactory::open(picPath);
        } catch (std::exception& e) {
            std::cout<<e.what()<<std::endl<<std::flush;
        }
        if(image.get() != 0){
            image->readMetadata();
            Exiv2::ExifData &exifData = image->exifData();
            if (!exifData.empty()) {

                double focalLengthmm = getMetadataFocalLength(exifData);

                bool hasFocalIn35mmMetadata=hasDigitMetadata(exifData,"Exif.Photo.FocalLengthIn35mmFilm");
                const double focalIn35mm = hasFocalIn35mmMetadata ? std::stod(getMetadata(exifData,"Exif.Photo.FocalLengthIn35mmFilm")) : -1.0;
                const double invRatio =  static_cast<double>(out.rows) / static_cast<double>(out.cols);
                double sensorWidth=-1.0;
                if(hasFocalIn35mmMetadata)
                {
                    std::cout<<"33"<<std::endl<<std::flush;
                    if(focalLengthmm > 0.0)
                    {
                        // no sensorWidth but valid focalLength and valid focalLengthIn35mm, so deduce sensorWith approximation
                        const double sensorDiag = (focalLengthmm * diag24x36) / focalIn35mm; // 43.3 is the diagonal of 35mm film
                        sensorWidth = sensorDiag * std::sqrt(1.0 / (1.0 + invRatio * invRatio));
                    }
                    else
                    {
                        // no sensorWidth and no focalLength but valid focalLengthIn35mm, so consider sensorWith as 35mm
                        sensorWidth = diag24x36 * std::sqrt(1.0 / (1.0 + invRatio * invRatio));
                        focalLengthmm = sensorWidth * (focalIn35mm ) / 36.0;
                    }
                    result=focalLengthmm*static_cast<double>(out.cols)/sensorWidth;
                }else{
                    double focalLengthmm=4.5f;
                    double sensorWidth=2.76;
                    result=focalLengthmm*static_cast<double>(out.cols)/sensorWidth;
                }
            }else{
                double focalLengthmm=4.5f;
                double sensorWidth=2.76;
                result=focalLengthmm*static_cast<double>(out.cols)/sensorWidth;
            }
        }else{
            std::cout<<"11"<<std::endl<<std::flush;
            double focalLengthmm=4.5f;
            double sensorWidth=2.76;
            result=focalLengthmm*static_cast<double>(out.cols)/sensorWidth;
        }
        return true;
    }
    return false;
}

ExifReader::ExifReader()
{

}
