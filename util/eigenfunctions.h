#ifndef EIGENFUNCTIONS_H
#define EIGENFUNCTIONS_H
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
template <typename Type>
class EigenFunctions
{
public:
    static void xRotation180(Eigen::Matrix<Type, 3, Eigen::Dynamic>& points);
    static void yRotation180(Eigen::Matrix<Type, 3, Eigen::Dynamic>& points);
    static void zRotation180(Eigen::Matrix<Type, 3, Eigen::Dynamic>& points);
    static Eigen::Matrix<Type, 3, 3> computeCovariance(Eigen::Matrix<Type, 3, Eigen::Dynamic> points,Eigen::Matrix<Type, 3, 1> center);
    static void svdDecomposition(Eigen::Matrix<Type, 3,3> &A,Eigen::Matrix<Type, 3, 3> &U,Eigen::Matrix<Type, 3, 3> &S,Eigen::Matrix<Type, 3, 3> &V);
    static void saveEigenPoints(Eigen::Matrix<Type,3,Eigen::Dynamic>&points,Eigen::Matrix3Xi faces,Eigen::Matrix4Xi colors,std::string savePath);
    static void saveEigenPoints(Eigen::Matrix<Type,3,Eigen::Dynamic>&points,Eigen::Matrix4Xi polygons,Eigen::Matrix4Xi colors,std::string savePath);
    static Type computeScale(Eigen::Matrix<Type,3,Eigen::Dynamic> points);
    static void center(Eigen::Matrix<Type,3,Eigen::Dynamic>& points);
    static Eigen::Matrix3Xf computeFaceNormals( Eigen::Matrix3Xf points, Eigen::Matrix3Xi faces);
    static Eigen::SparseMatrix<float> computeFaceNormalsWeight(int64_t vnum, Eigen::Matrix3Xi faces);
    static Eigen::Matrix3Xf computeVertNormals( Eigen::Matrix3Xf points,Eigen::Matrix3Xi faces);
};


template<typename Type>
void EigenFunctions<Type>::xRotation180(Eigen::Matrix<Type, 3, Eigen::Dynamic> &points)
{
    Eigen::Matrix<Type, 3, 3> R=-Eigen::Matrix<float, 3, 3>::Identity();
    R(0,0)=1;
    points=R*points;
}
template<typename Type>
void EigenFunctions<Type>::yRotation180(Eigen::Matrix<Type, 3, Eigen::Dynamic> &points)
{
    Eigen::Matrix<Type, 3, 3> R=-Eigen::Matrix<float, 3, 3>::Identity();
    R(1,1)=1;
    points=R*points;
}
template<typename Type>
void EigenFunctions<Type>::zRotation180(Eigen::Matrix<Type, 3, Eigen::Dynamic> &points)
{
    Eigen::Matrix<Type, 3, 3> R=-Eigen::Matrix<float, 3, 3>::Identity();
    R(2,2)=1;
    points=R*points;
}
template<typename Type>
Eigen::Matrix<Type, 3, 3> EigenFunctions<Type>::computeCovariance(Eigen::Matrix<Type, 3, Eigen::Dynamic> points, Eigen::Matrix<Type, 3, 1> center)
{
    points.colwise()-=center;
    return std::move(points*points.transpose());
}

template<typename Type>
void EigenFunctions<Type>::svdDecomposition(Eigen::Matrix<Type, 3, 3> &A, Eigen::Matrix<Type, 3, 3> &U,Eigen::Matrix<Type, 3, 3> &S, Eigen::Matrix<Type, 3, 3> &V)
{
    using namespace Eigen;
    using namespace Eigen::internal;
    using namespace Eigen::Architecture;
    JacobiSVD<Eigen::Matrix<Type, Dynamic, Dynamic> > svd(A, ComputeThinU | ComputeThinV );
    V = svd.matrixV(), U = svd.matrixU();
    S = U.transpose() * A * V; // S = U^-1 * A * VT * -1
}

template<typename Type>
void EigenFunctions<Type>::saveEigenPoints(Eigen::Matrix<Type, 3, Eigen::Dynamic>& points,Eigen::Matrix3Xi faces,Eigen::Matrix4Xi colors, std::string savePath)
{
    std::ofstream out(savePath);
    if(!out){
        std::cerr<<"write file:"<<savePath<<" fail!"<<std::endl<<std::flush;
        return;
    }
    if(points.size()==0){
        std::cerr<<"points is empty!"<<std::endl<<std::flush;
        return;
    }

    bool hasColor=colors.size()>0;
    for (int i = 0; i < points.cols(); i++) {
        out << "v " << points(0,i)<<" " << points(1,i) << " " << points(2,i) << " ";
        if(hasColor){
            out << colors(0,i)<<" " << colors(1,i) << " " << colors(2,i) << " "<< colors(3,i);
        }
        out<< std::endl;
    }
    for(int i=0;i<faces.cols();i++){
        out << "f " << faces(0,i)+1<<" " << faces(1,i)+1 << " " << faces(2,i)+1 << std::endl;;
    }
    out.close();
}

template<typename Type>
void EigenFunctions<Type>::saveEigenPoints(Eigen::Matrix<Type, 3, Eigen::Dynamic> &points, Eigen::Matrix4Xi polygons, Eigen::Matrix4Xi colors, std::string savePath)
{
    std::ofstream out(savePath);
    if(!out){
        std::cerr<<"write file:"<<savePath<<" fail!"<<std::endl<<std::flush;
        return;
    }
    if(points.size()==0){
        std::cerr<<"points is empty!"<<std::endl<<std::flush;
        return;
    }

    bool hasColor=colors.size()>0;
    for (int i = 0; i < points.cols(); i++) {
        //std::cout<<i<<std::endl<<std::flush;
        out << "v " << points(0,i)<<" " << points(1,i) << " " << points(2,i) << " ";
        if(hasColor){
            out << colors(0,i)<<" " << colors(1,i) << " " << colors(2,i) << " "<< colors(3,i);
        }
        out<< std::endl;
    }
    std::cout<<"------------"<<std::endl<<std::flush;
    for(int i=0;i<polygons.cols();i++){
        out << "f " << polygons(0,i)+1<<" " << polygons(1,i)+1 << " " << polygons(2,i)+1 <<" "<<polygons(3,i)+1 << std::endl;;
    }
    out.close();
}

template<typename Type>
Type EigenFunctions<Type>::computeScale(Eigen::Matrix<Type, 3, Eigen::Dynamic> points)
{
    int vnum=points.cols();
    Type maxX=1e-5;
    Type maxY=1e-5;
    Type maxZ=1e-5;
    Type minX=1e5;
    Type minY=1e5;
    Type minZ=1e5;
    for (int i=0;i<vnum;i++) {
        Type x = points(0,i),y = points(1,i),z = points(2,i);
        maxX = std::fmax(x ,maxX ); maxY = std::fmax(y ,maxY ) ; maxZ = std::fmax(z ,maxZ );
        minX = std::fmin(x ,minX ); minY = std::fmin(y ,minY ) ; minZ = std::fmin(z ,minZ );
    }
    Type xlen,ylen,zlen;
    xlen = fabs(maxX-minX);
    ylen = fabs(maxY-minY);
    zlen = fabs(maxZ-minZ);
    Eigen::Vector3f meshScale(xlen,ylen,zlen);
    return meshScale.maxCoeff();
}

template<typename Type>
void EigenFunctions<Type>::center(Eigen::Matrix<Type, 3, Eigen::Dynamic> &points)
{
    Eigen::Matrix<Type, Eigen::Dynamic, 1> c=points.rowwise().mean();
    points.colwise()-=c;
}

template<typename Type>
Eigen::Matrix3Xf EigenFunctions<Type>::computeFaceNormals(Eigen::Matrix3Xf points, Eigen::Matrix3Xi faces)
{
    Eigen::Matrix3Xf faceNormals;
    faceNormals.resize(3,faces.cols());
    for(int i=0;i<faces.cols();i++){
        int i0=faces(0,i);
        int i1=faces(1,i);
        int i2=faces(2,i);
        Eigen::Vector3f v1Subv0=points.col(i1)-points.col(i0);
        Eigen::Vector3f v2Subv0=points.col(i2)-points.col(i0);
        Eigen::Vector3f normal=v1Subv0.cross(v2Subv0);
        faceNormals.col(i)=normal;
    }
    return faceNormals;
}

template<typename Type>
Eigen::SparseMatrix<float> EigenFunctions<Type>::computeFaceNormalsWeight(int64_t vnum, Eigen::Matrix3Xi faces)
{
    int64_t fnum=faces.cols();
    Eigen::SparseMatrix<float> weights;
    weights.resize(vnum,fnum);
    std::vector<Eigen::Triplet<float>> trips;
    trips.resize(fnum*3);
    for(int64_t i=0;i<fnum;i++){
        int r0=faces(0,i);
        int r1=faces(1,i);
        int r2=faces(2,i);
        trips[3*i]=Eigen::Triplet<float>(r0,i,1.0);
        trips[3*i+1]=Eigen::Triplet<float>(r1,i,1.0);
        trips[3*i+2]=Eigen::Triplet<float>(r2,i,1.0);
    }
    weights.setFromTriplets(trips.begin(),trips.end());//vnumxfnum
    return std::move(weights);
}

template<typename Type>
Eigen::Matrix3Xf EigenFunctions<Type>::computeVertNormals(Eigen::Matrix3Xf points, Eigen::Matrix3Xi faces)
{
    int64_t vnum=points.cols();
    Eigen::SparseMatrix<float> weights=computeFaceNormalsWeight(vnum,faces);//vnumxfnum
    Eigen::Matrix3Xf fn=computeFaceNormals(points,faces);//3xfnum
    Eigen::MatrixX3f vn=weights*fn.transpose();//vnumxfnum * fnumx3=vnumx3
    return  vn.transpose();
}


#endif // EIGENFUNCTIONS_H
