#ifndef CERESNONLINEAR_HPP
#define CERESNONLINEAR_HPP
#include <map>
#include <Eigen/Dense>
namespace fitting {

struct Key2DPointCost{
    Key2DPointCost(float  fx,Eigen::Vector3f srcKeyPoint,Eigen::Vector3f dstKeyPoint):fx(fx),srcKeyPoint(srcKeyPoint),dstKeyPoint(dstKeyPoint){

    }
    template<typename T>
    bool operator()(const T* const quat, const T* const translation,T* residual)const;

    //bool operator()(const double* const quat, const double* const translation,double* residual)const;
private:
    Eigen::Vector3f srcKeyPoint;
    Eigen::Vector3f dstKeyPoint;

    float fx;
};
template<typename T>
bool Key2DPointCost::operator()(const T* const quat, const T* const translation, T* residual)const{
    Eigen::Quaternion<T> q(quat[0],quat[1],quat[2],quat[3]);
    Eigen::Matrix<T,3,3> Rotation=q.toRotationMatrix();
    Eigen::Matrix<T, 3, 1> t(translation[0],translation[1],translation[2]);

    Eigen::Matrix<T,3,1> tmpSrcKeyPoint;
    tmpSrcKeyPoint(0,0)=T(srcKeyPoint(0,0));
    tmpSrcKeyPoint(1,0)=T(srcKeyPoint(1,0));
    tmpSrcKeyPoint(2,0)=T(srcKeyPoint(2,0));
    tmpSrcKeyPoint=Rotation*tmpSrcKeyPoint+t;

    T dis=tmpSrcKeyPoint(2,0);
    Eigen::Matrix<T,3,1> tmpDstKeyPoint;
    tmpDstKeyPoint(0,0)=T(dstKeyPoint(0,0));
    tmpDstKeyPoint(1,0)=T(dstKeyPoint(1,0));
    tmpDstKeyPoint(2,0)=T(dstKeyPoint(2,0));
    Eigen::Matrix<T,3,1> diff=dis*tmpDstKeyPoint-T(fx)*tmpSrcKeyPoint;
    residual[0]=T(diff(0,0));
    residual[1]=T(diff(1,0));
    //residual[2]=double(tmpDstKeyPoint(2,0)/dis-tmpDstKeyPoint(2,0));
    return true;
}

//bool Key2DPointCost::operator()(const double* const quat, const double* const translation, double* residual)const{
//    Eigen::Quaternion<double> q(quat[0],quat[1],quat[2],quat[3]);
//    Eigen::Matrix<double,3,3> Rotation=q.toRotationMatrix();
//    Eigen::Matrix<double, 3, 1> t(translation[0],translation[1],translation[2]);

//    Eigen::Matrix<double,3,1> tmpSrcKeyPoint;
//    tmpSrcKeyPoint(0,0)=double(srcKeyPoint(0,0));
//    tmpSrcKeyPoint(1,0)=double(srcKeyPoint(1,0));
//    tmpSrcKeyPoint(2,0)=double(srcKeyPoint(2,0));
//    tmpSrcKeyPoint=Rotation*tmpSrcKeyPoint+t;

//    double dis=tmpSrcKeyPoint(2,0);
//    Eigen::Matrix<double,3,1> tmpDstKeyPoint;
//    tmpDstKeyPoint(0,0)=double(dstKeyPoint(0,0));
//    tmpDstKeyPoint(1,0)=double(dstKeyPoint(1,0));
//    tmpDstKeyPoint(2,0)=double(dstKeyPoint(2,0));
//    Eigen::Matrix<double,3,1> diff=dis*tmpDstKeyPoint-fx*tmpSrcKeyPoint;
//    residual[0]=double(diff(0,0));
//    residual[1]=double(diff(1,0));
//    //residual[2]=double(tmpDstKeyPoint(2,0)/dis-tmpDstKeyPoint(2,0));
//    return true;
//}

}
#endif // CERESNONLINEAR_HPP
