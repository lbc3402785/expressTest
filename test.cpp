#include "test.h"
#include "src/MMSolver.h"
#include "src/Dlib.h"
#include <QSettings>
#include "exifreader.h"
#include "util/eigenfunctions.h"
Test::Test()
{

}
bool KeypointDetectgion(Mat image, MatF &KP)
{
    vector<vector<Point>> keypoints;
    double S = 0.5;
    Mat simage;
    cv::resize(image, simage, Size(), S, S);

    vector<Rect> rectangles;
    DlibInit("data\\shape_predictor_68_face_landmarks.dat");
    DlibFace(simage, rectangles, keypoints);

    if (keypoints.size() <= 0)
    {
        errorcout << "NO POINTS" << endl;
        return false;
    }

    KP = ToEigen(keypoints[0]) * (1.0 / S);

    return true;
}
void loadKeyIndex(std::string keyPath,std::vector<int>& keys){
    QSettings settings(QString::fromStdString(keyPath),QSettings::IniFormat);
    keys.clear();
    for(int i=1;i<=68;i++){
        QString key=QString::number(i);
        int value=settings.value(key,-1).value<int>();
        if(value==-1){
            std::cerr<<"invalid index!"<<std::endl<<std::flush;
        }else{
            //std::cerr<<"value:"<<value<<std::endl<<std::flush;
            keys.push_back(value-1);
        }
    }

}
void extractKeyShape(const MatF &SB, std::vector<int> &indexes, MatF &keySB)
{
    int n=indexes.size();
    int L=SB.cols();
    keySB.resize(n*3,L);
    for(int i=0;i<n;i++){
        int index=indexes[i];
        keySB.block(i*3,0,3,L)=SB.block(index*3,0,3,L);
    }
}

void extractKeyFace(const MatF Face, std::vector<int> &indexes, MatF &keyFace)
{
    keyFace.resize(indexes.size(),3);
    for(int i=0;i<keyFace.rows();i++){
        keyFace.row(i)=Face.row(indexes[i]);
    }
}
void extractG8MKeyModel(const FaceModel &BFMModel, std::vector<int> &indexes, FaceModel &keyModel)
{
    extractKeyShape(BFMModel.EB,indexes,keyModel.EB);
    extractKeyFace(BFMModel.Face,indexes,keyModel.Face);
    keyModel.SB=keyModel.EB;
}
void extractBFMKeyModel(const FaceModel &BFMModel, std::vector<int> &indexes, FaceModel &keyModel)
{
    extractKeyShape(BFMModel.SB,indexes,keyModel.SB);
    extractKeyShape(BFMModel.EB,indexes,keyModel.EB);
    extractKeyFace(BFMModel.Face,indexes,keyModel.Face);
}
void Test::testG8M(std::string picPath)
{

    cv::Mat image= imread(picPath);
    MatF KP;

    if(KeypointDetectgion(image, KP)){

        Eigen::Matrix4Xi colors;
        //        FaceModel bfmShape;
        //        bfmShape.Initialize("E:\\model\\BFMUV.obj.npz",true);
        //        Eigen::Matrix3Xf BFMpoints=bfmShape.Face.transpose();
        //        Eigen::Matrix3Xi BFMFaces=bfmShape.TRI.transpose();
        //        EigenFunctions<float>::saveEigenPoints(BFMpoints,BFMFaces,colors,"BFMBASE.obj");
        //        std::vector<int> BFMkeys;
        //        fitting::ModelFitting::loadKeyIndex("E:\\model\\BFM.ini",BFMkeys);
        //        FaceModel bfmKeyShape;
        //        fitting::ModelFitting::extractBFMKeyModel(bfmShape,BFMkeys,bfmKeyShape);
        //-----------------
        FaceModel shape;
        shape.InitializeG8M("data\\G8M_BlendShapes1.npz",false);
        Eigen::Matrix3Xf points=shape.Face.transpose();

        FaceModel keyShape;
        std::vector<int> g8Mkeys;
        loadKeyIndex("data\\g8MExpress.ini",g8Mkeys);
        extractG8MKeyModel(shape,g8Mkeys,keyShape);

        MMSolver g8mSolver;
        g8mSolver.FM=keyShape;
        g8mSolver.FMFull=shape;
        g8mSolver.FixShape=true;
        g8mSolver.SX0.resize(g8mSolver.FMFull.SB.cols(),1);
        g8mSolver.SX0.setZero();
        g8mSolver.SX=g8mSolver.SX0;
        g8mSolver.Solve(KP);
        std::cout<<"g8mSolver.EX:"<<g8mSolver.EX<<std::endl<<std::flush;
        MatF face=g8mSolver.FMFull.Generate(g8mSolver.SX,g8mSolver.EX);
        points=face.transpose();
        Mat res = MMSDraw(image, g8mSolver,KP);
        imshow("RES", res);
        cv::waitKey();

    }

}

void Test::testPerspective(string picPath)
{
    //    cv::Mat image= imread(picPath);
    bool center=true;
    MatF KP;
    ExifReader& reader=ExifReader::getInstance();
    double fx;
    cv::Mat image;
    if(reader.getFocalLength(picPath,image,fx)){
        std::cout<<"fx:"<<fx<<std::endl<<std::flush;
        if(KeypointDetectgion(image, KP)){
            if(center){
                KP.col(0).array()-=image.cols/2;
                KP.col(1).array()-=image.rows/2;
            }
            FaceModel bfmShape;
            bfmShape.Initialize("E:\\model\\BFMUVWM.obj.npz",true);
            Eigen::Matrix3Xf BFMpoints=bfmShape.Face.transpose();
            Eigen::Matrix3Xi BFMFaces=bfmShape.TRI.transpose();
            Eigen::Matrix4Xi colors;
            EigenFunctions<float>::saveEigenPoints(BFMpoints,BFMFaces,colors,"BFMBASE.obj");
            std::vector<int> BFMkeys;
            loadKeyIndex("E:\\model\\BFM.ini",BFMkeys);
            FaceModel bfmKeyShape;
            extractBFMKeyModel(bfmShape,BFMkeys,bfmKeyShape);
            MMSolver solver;
            solver.fx=fx;
            solver.FMFull=bfmShape;
            solver.FM=bfmKeyShape;
//            KP/=solver.fx;
            solver.Solve(KP);
            solver.SolvePerspective2(KP);
            MMSObj(image, solver, "E:\\coding\\git-code\\expressTest\\output\\", "test",true);
            Mat res = MMSDraw(image, solver,KP,true);
            Mat simage;
            cv::resize(res, simage, Size(), 0.25, 0.25);
            imshow("RES", simage);
            cv::waitKey();
        }
    }

}

void Test::test()
{
    MatF A=MatF::Ones(2,3);
    A(0,0)=5;
    A(0,2)=6;
    A(1,2)=3;
    A(2,1)=4;
    std::cout<<"A:"<<A<<std::endl<<std::flush;
    MatF zs=A.col(2);
    MatF fs=MatF::Ones(zs.rows(),zs.cols())*2;
    MatF rs=fs.array()/zs.array();
    std::cout<<"rs:"<<rs<<std::endl<<std::flush;
    MatF xs=A.col(0);
    MatF ys=A.col(1);
    xs=rs.array()*xs.array();
    ys=rs.array()*ys.array();
    MatF pro=MatF::Zero(A.rows(),2);
    pro.col(0)=xs;
    pro.col(1)=ys;
//    rotated.col(0)=rs.array()*rotated.col(0).array();
//    rotated.col(1)=rs.array()*rotated.col(1).array();
//    MatF pro=rotated.block(0,0,rotated.rows(),2);
    std::cout<<"pro:"<<pro<<std::endl<<std::flush;
}
