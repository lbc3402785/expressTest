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

struct LandmarkCost{
    LandmarkCost(FaceModel keyModel,float  fx,int vertexId,Eigen::Vector3f dstKeyPoint):keyModel(keyModel),fx(fx),vertexId(vertexId),dstKeyPoint(dstKeyPoint){

    }
    template<typename T>
    bool operator()(const T* const quat, const T* const translation,const T* const shape_coeffs, const T* const blendshape_coeffs,T* residual)const;

    //bool operator()(const double* const quat, const double* const translation,double* residual)const;
private:
    FaceModel keyModel;
    int vertexId;
    Eigen::Vector3f dstKeyPoint;

    float fx;
};
template<typename T>
std::array<T, 3> get_shape_point(FaceModel keyModel,int vertId,const T* const shape_coeffs, const T* const blendshape_coeffs){
    int num_coeffs_fitting=keyModel.SB.cols();
    int num_blendshapes=keyModel.EB.cols();
    auto mean=keyModel.Face.row(vertId);//1x3
    auto basis=keyModel.SB.block(3*vertId,0,3,num_coeffs_fitting);
    auto blendShapes=keyModel.EB.block(3*vertId,0,3,num_blendshapes);
    std::array<T, 3> point{T(mean(0,0)), T(mean(0,1)), T(mean(0,2))};

    for (int i = 0; i < num_coeffs_fitting; ++i)
    {
        point[0] += T(basis.row(0).col(i)(0)) * shape_coeffs[i]; // it seems to be ~15% faster when these are
                                                                 // static_cast<double>() instead of T()?
    }
    for (int i = 0; i < num_coeffs_fitting; ++i)
    {
        point[1] += T(basis.row(1).col(i)(0)) * shape_coeffs[i];
    }
    for (int i = 0; i < num_coeffs_fitting; ++i)
    {
        point[2] += T(basis.row(2).col(i)(0)) * shape_coeffs[i];
    }


    for (int i = 0; i < num_blendshapes; ++i)
    {
        point[0] += T(blendShapes.row(0).col(i)(0)) * blendshape_coeffs[i]; // it seems to be ~15% faster when these are
                                                                 // static_cast<double>() instead of T()?
    }
    for (int i = 0; i < num_blendshapes; ++i)
    {
        point[1] += T(blendShapes.row(1).col(i)(0)) * blendshape_coeffs[i];
    }
    for (int i = 0; i < num_blendshapes; ++i)
    {
        point[2] += T(blendShapes.row(2).col(i)(0)) * blendshape_coeffs[i];
    }
    return point;
}
template<typename T>
bool LandmarkCost::operator()(const T* const quat, const T* const translation,const T* const shape_coeffs, const T* const blendshape_coeffs, T* residual)const{
    Eigen::Quaternion<T> q(quat[0],quat[1],quat[2],quat[3]);
    Eigen::Matrix<T,3,3> Rotation=q.toRotationMatrix();
    Eigen::Matrix<T, 3, 1> t(translation[0],translation[1],translation[2]);

    const auto point_arr=get_shape_point<T>(keyModel, vertexId, shape_coeffs, blendshape_coeffs);
    Eigen::Matrix<T,3,1> tmpSrcKeyPoint;
    tmpSrcKeyPoint(0,0)=point_arr[0];
    tmpSrcKeyPoint(1,0)=point_arr[1];
    tmpSrcKeyPoint(2,0)=point_arr[2];
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
/**
 * Cost function for a prior on the parameters.
 *
 * Prior towards zero (0, 0...) for the parameters.
 * Note: The weight is inside the norm, so may not correspond to the "usual"
 * formulas. However I think it's equivalent up to a scaling factor, but it
 * should be checked.
 */
struct PriorCost
{

    /**
     * Creates a new prior object with set number of variables and a weight.
     *
     * @param[in] num_variables Number of variables that the parameter vector contains.
     * @param[in] weight A weight that the parameters are multiplied with.
     */
    PriorCost(int numVariables, double weight = 1.0) : numVariables(numVariables), weight(weight){};

    /**
     * Cost function implementation.
     *
     * @param[in] x An array of parameters.
     * @param[in] residual An array of the resulting residuals.
     * @return Returns true. The ceres documentation is not clear about that I think.
     */
    template <typename T>
    bool operator()(const T* const x, T* residual) const
    {
        for (int i = 0; i < numVariables; ++i)
        {
            residual[i] = weight * x[i];
        }
        return true;
    };


public:
    double PriorCost::getWeight() const
    {
        return weight;
    }

    void PriorCost::setWeight(double value)
    {
        weight = value;
    }

private:
    int numVariables;
    double weight;
};
}
#endif // CERESNONLINEAR_HPP
