#pragma once
#define TODO
#define errorcout cout
#define debugcout cout
#define CV_AA -1
#include "cnpy.h"
#include "NumpyUtil.h"
#include "ceresnonlinear.hpp"
#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/rotation.h"
#include "ceresnonlinear.hpp"
using namespace cv;
inline cv::Point2f ImageCoordinate(MatF face, int i)
{
	float s = 2.0;
	float x = face(3 * i + 0) * s;
	float y = face(3 * i + 1) * s;

	Point2f p(x, -y);
	Point2f offset(250, 250);

	return p + offset;
}

inline Point Shape(MatF A)
{
	return Point(A.rows(), A.cols());
}

inline Point Shape(MatI A)
{
	return Point(A.rows(), A.cols());
}

inline Eigen::Matrix3f Orthogonalize(MatF R)
{
	// Set R to the closest orthonormal matrix to the estimated affine transform:
	Eigen::JacobiSVD<Eigen::Matrix3f, Eigen::NoQRPreconditioner> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3f U = svd.matrixU();
	const Eigen::Matrix3f V = svd.matrixV();
	Eigen::Matrix3f R_ortho = U * V.transpose();
	// The determinant of R must be 1 for it to be a valid rotation matrix
	if (R_ortho.determinant() < 0)
	{
		U.block<1, 3>(2, 0) = -U.block<1, 3>(2, 0); // not tested
		R_ortho = U * V.transpose();
	}

	return R_ortho;
}


inline Eigen::VectorXf SolveLinear(MatF A, MatF B, float lambda)
{
	//lambda = 1.0
	// Solve d[(Ax-b)^2 + lambda * x^2 ]/dx = 0 
		// https://math.stackexchange.com/questions/725185/minimize-a-x-b
	Eigen::MatrixXf Diagonal = Eigen::MatrixXf::Identity(A.cols(), A.cols()) * lambda;
	auto AA = A.transpose() * A + Diagonal;
	Eigen::VectorXf X = AA.colPivHouseholderQr().solve(A.transpose() * B);
	return X;
}

inline Eigen::VectorXf SolveLinear(MatF A, MatF B)
{
	Eigen::VectorXf X = A.colPivHouseholderQr().solve(B);
	return X;
}




class ProjectionParameters
{
public:
	Eigen::Matrix3f R; ///< 3x3 rotation matrix
    float tx, ty,tz; ///< x and y translation
	float s;      ///< Scaling

	//Need Transpose
	Mat GenerateCVProj()
	{
		Matrix3f Rt = R.transpose() * s;

		Mat P = Mat::eye(4, 4, CV_32F);

		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				P.at<float>(i, j) = Rt(i, j);
			}
		}

		P.at<float>(0, 3) = tx * s;
		P.at<float>(1, 3) = ty * s;

		return P;
	}
};


inline MatF Projection(ProjectionParameters p, MatF model_points)
{
	MatF R = p.R.transpose() * p.s;
	R = R.block(0, 0, 3, 2);

	MatF rotated = model_points * R;

	auto sTx = p.tx * p.s;
	auto sTy = p.ty * p.s;

	int N = model_points.rows();
	rotated.col(0).array() += sTx;
	rotated.col(1).array() += sTy;

	return rotated;// .block(0, 0, N, 2);
}

inline MatF Rotation(ProjectionParameters p, MatF model_points)
{
	MatF R = p.R;
	R = R.block(0, 0, 3, 3);
	MatF rotated = model_points * R;
	return rotated;// .block(0, 0, N, 2);
}


inline MatF ToEigen(std::vector<Point> image_points)
{
	int N = image_points.size();

	MatF b(N, 2);
	for (int i = 0; i < N; ++i)
	{
		Eigen::Vector2f p = Eigen::Vector2f(image_points[i].x, image_points[i].y);
		b.block<1, 2>(i, 0) = p;
	}

	return b;
}

inline Matrix<float, 1, Eigen::Dynamic> GetMean(MatF &input)
{
	return input.colwise().mean();
}

inline void SubtractMean(MatF &input)
{
	input.rowwise() -= GetMean(input);
}





class FaceModel
{
public:
	MatF SM;
	MatF SB;
	MatF EM;
	MatF EB;

	MatI TRI;
	MatI TRIUV;
	MatI Ef;
	MatI Ev;

	MatF UV;

	MatF Face;
	MatF GeneratedFace;
	Matrix<float, 1, Eigen::Dynamic> Mean;

	FaceModel()
	{

	}

	void Initialize(string file, bool LoadEdge)
	{
		cnpy::npz_t npz = cnpy::npz_load(file);

		SM = ToEigen(npz["SM"]);
		SB = ToEigen(npz["SB"]);

		EM = ToEigen(npz["EM"]);
		EB = ToEigen(npz["EB"]);

		MatF FaceFlat = SM + EM; // 204 * 1 
		Face = Reshape(FaceFlat, 3);

		/*Mean = GetMean(Face);
		Face.rowwise() -= Mean; */

		if (LoadEdge)
		{
			TRI = ToEigenInt(npz["TRI"]);

			try
			{
				TRIUV = ToEigenInt(npz["TRIUV"]);
			}
			catch (...)
			{
				TRIUV = TRI;
			}
			Ef = ToEigenInt(npz["Ef"]);
			Ev = ToEigenInt(npz["Ev"]);

			UV = ToEigen(npz["UV"]);
		}
	}

    void InitializeG8M(string file, bool LoadEdge){
        cnpy::npz_t npz = cnpy::npz_load(file);

        SM = ToEigen(npz["SM"]);
        //SB = ToEigen(npz["SB"]);

        //EM = ToEigen(npz["EM"]);
        EB = ToEigen(npz["EB"]);

        MatF FaceFlat = SM /*+ EM*/; // 204 * 1
        Face = Reshape(FaceFlat, 3);
        SB=EB;
    }
	MatF Generate(MatF SX, MatF EX)
	{
		MatF FaceS = SB * SX;
		MatF S = Reshape(FaceS, 3);

		MatF FaceE = EB * EX;
		MatF E = Reshape(FaceE, 3);

		GeneratedFace =  Face + S + E;
		return GeneratedFace;
	}
};




class MMSolver
{
public:
	bool USEWEIGHT = true;
	float WEIGHT = 1.0;
	//vector<int> SkipList = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  11, 12, 13, 14, 15, 16 };
	//vector<int> SkipList = { 0, 1, 2, 3, 4,   5, 6, 7,    9,10,  11, 12,  13, 14, 15, 16 };
	vector<int> SkipList = {8, };

    float fx=1.0;
	FaceModel FM;
	FaceModel FMFull;
    bool FIXFIRSTSHAPE = true;

	void Initialize(string file, string file2)
	{
		FM.Initialize(file, false);
		//FMFull.Initialize(file2, FM.Mean, true);
		FMFull.Initialize(file2, true);


		if (FIXFIRSTSHAPE)
		{
			FM.SB.col(0) *= 0;
			FMFull.SB.col(0) *= 0;
		}
	}
    MatF PerspectiveProjection(ProjectionParameters p, MatF model_points)
    {
        MatF rotated=model_points*p.R.transpose();
        rotated.col(0).array() += p.tx;
        rotated.col(1).array() += p.ty;
        rotated.col(2).array() += p.tz;
        MatF zs=rotated.col(2);
        MatF fs=MatF::Ones(zs.rows(),zs.cols())*fx;
        MatF rs=fs.array()/zs.array();
        MatF xs=rotated.col(0);
        MatF ys=rotated.col(1);
        xs=rs.array()*xs.array();
        ys=rs.array()*ys.array();
        MatF pro=MatF::Zero(rotated.rows(),2);
        pro.col(0)=xs;
        pro.col(1)=ys;
        return pro;
    }
    MatF Transform(ProjectionParameters p, MatF model_points)
    {
        MatF rotated=model_points*p.R.transpose();
        rotated.col(0).array() += p.tx;
        rotated.col(1).array() += p.ty;
        rotated.col(2).array() += p.tz;
        return rotated;
    }
    MatF SolveShapePerspective(ProjectionParameters p, MatF image_points, MatF M, MatF SB, float lambda)
    {
        //cout << Shape(SB) << endl;

        MatF rotated = Transform(p, M);//Nx3
        MatF zs=rotated.col(2);
        MatF xs=rotated.col(0);
        xs*=fx;
        MatF ys=rotated.col(1);
        ys*=fx;
        MatF imageXs=image_points.col(0);
        imageXs=imageXs.array()*zs.array();
        MatF imageYs=image_points.col(1);
        imageYs=imageYs.array()*zs.array();
        MatF errXs=imageXs-xs;
        MatF errYs=imageYs-ys;
        MatF error = MatF::Zero(image_points.rows(),image_points.cols());
        error.col(0)=errXs;
        error.col(1)=errYs;
        error = Reshape(error, 1);

        int N = M.rows();
        int N2 = SB.rows();
        int L = SB.cols();

        assert(N2 == N * 3);
        auto sTx = p.tx * p.s;
        auto sTy = p.ty * p.s;
        MatF SBX(N * 2, L);
        for (size_t i = 0; i < N; i++)
        {
            MatF SBRotation= p.R * SB.block(i * 3, 0, 3, L);//3xL
            MatF xs=SBRotation.row(0);//1xL
            MatF ys=SBRotation.row(1);//1xL
            float zc=rotated(i,2);
            xs*=fx;
            ys*=fx;
            SBX.block(i * 2, 0, 1, L)=xs;
            SBX.block(i * 2+1, 0, 1, L)=ys;
            /*SBX.row(i * 2) .array()= sTx;
            SBX.row(i * 2+1).array() = sTy;*/
        }

        if (USEWEIGHT)
        {
            Matrix<float, Eigen::Dynamic, 1> W = Matrix<float, Eigen::Dynamic, 1>::Ones(2 * N, 1);
            for (size_t i = 0; i < SkipList.size(); i++)
            {
                W(2 * SkipList[i] + 0, 0) = WEIGHT;
                W(2 * SkipList[i] + 1, 0) = WEIGHT;
            }
            SBX = W.asDiagonal() * SBX;
            error = W.asDiagonal() * error;
        }

        auto X = SolveLinear(SBX/fx, error/fx, lambda);

        cout << "error:"<<error  << endl;

        return X;
        //MatF rotated = (model_points + Ax) * R;
    }
	MatF SolveShape(ProjectionParameters p, MatF image_points, MatF M, MatF SB, float lambda)
	{
		//cout << Shape(SB) << endl;

		MatF R = p.R.transpose()  * p.s;
		R = R.block(0, 0, 3, 2);

		MatF rotated = Projection(p, M);

		MatF error = image_points - rotated;
		error = Reshape(error, 1);

		int N = M.rows();
		int N2 = SB.rows();
		int L = SB.cols();

		assert(N2 == N * 3);
		auto sTx = p.tx * p.s;
		auto sTy = p.ty * p.s;
		MatF SBX(N * 2, L);
		MatF Rt = R.transpose();
		for (size_t i = 0; i < N; i++)
		{
			SBX.block(i * 2, 0, 2, L) = Rt * SB.block(i * 3, 0, 3, L);
			/*SBX.row(i * 2) .array()= sTx;
			SBX.row(i * 2+1).array() = sTy;*/
		}

		if (USEWEIGHT)
		{
			Matrix<float, Eigen::Dynamic, 1> W = Matrix<float, Eigen::Dynamic, 1>::Ones(2 * N, 1);
			for (size_t i = 0; i < SkipList.size(); i++)
			{
				W(2 * SkipList[i] + 0, 0) = WEIGHT;
				W(2 * SkipList[i] + 1, 0) = WEIGHT;
			}
			SBX = W.asDiagonal() * SBX;
			error = W.asDiagonal() * error;
		}

		auto X = SolveLinear(SBX, error, lambda);

		//cout << (error - SBX * X).norm() << endl;

		return X;
		//MatF rotated = (model_points + Ax) * R;
	}
    ProjectionParameters SolveProjectionNonlinear(MatF image_points, MatF model_points){
        using namespace fitting;
        int N = image_points.rows();
        Eigen::Quaternionf q(params.R);
        std::vector<double> cameraRotation(4,0.0);
        cameraRotation[0] = q.w();
        cameraRotation[1] = q.x();
        cameraRotation[2] = q.y();
        cameraRotation[3] = q.z();
        std::vector<double> translation(3,100.0);
        translation[0]=params.tx;
        translation[0]=params.ty;
        ceres::Problem problem;
        for(int i=0;i<N;i++){
           Eigen::Vector3f srcKeyPoint= model_points.row(i);
           Eigen::Vector3f dstKeyPoint(image_points(i,0),image_points(i,1),1);
           fitting::Key2DPointCost* cost=new fitting::Key2DPointCost(fx,srcKeyPoint,dstKeyPoint);
           ceres::CostFunction* costFunction=new ceres::AutoDiffCostFunction<fitting::Key2DPointCost,2,4,3>(cost);
//           ceres::CostFunction* costFunction =
//                new ceres::NumericDiffCostFunction<fitting::Key2DPointCost, ceres::CENTRAL,2, 4, 3> (cost);
           problem.AddResidualBlock(costFunction,/*new ceres::CauchyLoss(0.5)*/NULL,&cameraRotation[0],&translation[0]);
        }
        ceres::QuaternionParameterization* cameraFitQuaternionParameterisation = new ceres::QuaternionParameterization();
        problem.SetParameterization(&cameraRotation[0], cameraFitQuaternionParameterisation);
        ceres::Solver::Options solverOptions;
        solverOptions.linear_solver_type = ceres::SPARSE_SCHUR;
        solverOptions.num_threads = 1;
        //solverOptions.max_num_iterations=500;
        solverOptions.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary solverSummary;
        ceres::Solve(solverOptions, &problem, &solverSummary);
        std::cout << solverSummary.BriefReport() << "\n";
        Eigen::Quaternion<double> qd(cameraRotation[0],cameraRotation[1],cameraRotation[2],cameraRotation[3]);
        Eigen::Matrix<double, 3, 1> t(translation[0],translation[1],translation[2]);
        Eigen::Matrix<double,3,3> Rotation=qd.toRotationMatrix();
        Eigen::Matrix<float,3,3> Rf=Rotation.cast<float>();
        Eigen::Matrix<float,3,1> tf=t.cast<float>();
        std::cout << "Rf:" << Rf << std::endl;
        std::cout << "tf:" << tf << std::endl;
        return {Rf,tf(0,0),tf(1,0),tf(2,0),1.0f};
    }
    ProjectionParameters MMSolver::SolveProjection(MatF image_points, MatF model_points)
    {
      //########## Mean should be subtracted from model_points ############

      Matrix<float, 1, Eigen::Dynamic> Mean = GetMean(model_points);
      model_points.rowwise() -= Mean;


      using Eigen::Matrix;
      int N = image_points.rows();

      assert(image_points.rows() == model_points.rows());
      assert(2 == image_points.cols());
      assert(3 == model_points.cols());

      model_points.conservativeResize(N, 4);
      model_points.col(3).setOnes();

      Matrix<float, Eigen::Dynamic, 8> A = Matrix<float, Eigen::Dynamic, 8>::Zero(2 * N, 8);
      for (int i = 0; i < N; ++i)
      {
        Eigen::Vector4f P = model_points.row(i);// .transpose();//Eigen::Vector4f();
        A.block<1, 4>(2 * i, 0) = P;       // even row - copy to left side (first row is row 0)
        A.block<1, 4>((2 * i) + 1, 4) = P; // odd row - copy to right side
      } // 4th coord (homogeneous) is already 1

      //Matrix<float, 1, Eigen::Dynamic> MeanX = image_points.colwise().mean();
      //image_points.rowwise() -= MeanX;

      MatF b = Reshape(image_points, 1);

      if (USEWEIGHT)
      {
        Matrix<float, Eigen::Dynamic, 1> W = Matrix<float, Eigen::Dynamic, 1>::Ones(2 * N, 1);
        for (size_t i = 0; i < SkipList.size(); i++)
        {
          W(2 * SkipList[i] + 0, 0) = WEIGHT;
          W(2 * SkipList[i] + 1, 0) = WEIGHT;
        }
        A = W.asDiagonal() * A;
        b = W.asDiagonal() * b;
      }

      const Matrix<float, 8, 1> k = SolveLinear(A, b); // resulting affine matrix (8x1)
      // Extract all values from the estimated affine parameters k:
      const Eigen::Vector3f R1 = k.segment<3>(0);
      const Eigen::Vector3f R2 = k.segment<3>(4);
      Eigen::Matrix3f R;
      Eigen::Vector3f r1 = R1.normalized(); // Not sure why R1.normalize() (in-place) produces a compiler error.
      Eigen::Vector3f r2 = R2.normalized();
      R.block<1, 3>(0, 0) = r1;
      R.block<1, 3>(1, 0) = r2;
      R.block<1, 3>(2, 0) = r1.cross(r2);
      float sTx = k(3);
      float sTy = k(7);

      //sTx += Mean(0);
      //sTy += Mean(1);

      const auto s = (R1.norm() + R2.norm()) / 2.0f;


      Eigen::Matrix3f R_ortho = Orthogonalize(R);

      std::cout << "R:" << R << std::endl;
      std::cout << "R_ortho:" << R_ortho << std::endl;
      MatF T = Mean * R_ortho.transpose();
      // Remove the scale from the translations:
      auto t1 = sTx / s - T(0);
      auto t2 = sTy / s - T(1);

      auto error = (A*k - b).norm();

      std::cout << "TTS:" << t1 << " " << t2 << " " << s << std::endl;
      std::cout << "error:" << error << std::endl;

      return ProjectionParameters{ R_ortho, t1, t2,0, s };
    }
	MatF SX;
	MatF EX;
	ProjectionParameters params;

	bool FixShape = false;
	MatF SX0;
	

	

	void Solve(MatF KP)
	{
		MatF Face = FM.Face;
		MatF S = Face * 0;
		MatF E = Face * 0;


        float Lambdas[7] = { 100.0, 30.0, 10.0, 5.0,4.0,3.0,2.0};


        for (size_t i = 0; i < 4; i++)
		{
            params = SolveProjection(KP, Face);

			if (FixShape)
			{
				SX = SX0;
			}
			else
			{
                SX = SolveShape(params, KP, FM.Face + E, FM.SB, Lambdas[i]*5);
				if (FIXFIRSTSHAPE)
				{
					SX(0, 0) = 0;
				}
			}
			MatF FaceS = FM.SB * SX;
			S = Reshape(FaceS, 3);

            EX = SolveShape(params, KP, FM.Face + S, FM.EB, Lambdas[i]*1);
			
			MatF FaceE = FM.EB * EX;
			E = Reshape(FaceE, 3);

			Face = FM.Face + S + E;

			
		}

	}
    void SolvePerspective(MatF KP)
    {
        MatF Face = FM.Face;
        MatF S = Face * 0;
        MatF E = Face * 0;


        float Lambdas[7] = { 100.0, 30.0, 10.0, 5.0,4.0,3.0,2.0};


        for (size_t i = 0; i < 4; i++)
        {
            params = SolveProjectionNonlinear(KP, Face);

            if (FixShape)
            {
                SX = SX0;
            }
            else
            {
                SX = SolveShapePerspective(params, KP, FM.Face + E, FM.SB, Lambdas[i]*5);
                if (FIXFIRSTSHAPE)
                {
                    SX(0, 0) = 0;
                }
            }
            MatF FaceS = FM.SB * SX;
            S = Reshape(FaceS, 3);

            EX = SolveShapePerspective(params, KP, FM.Face + S, FM.EB, Lambdas[i]*1);

            MatF FaceE = FM.EB * EX;
            E = Reshape(FaceE, 3);

            Face = FM.Face + S + E;


        }

    }
};





inline Mat MMSDraw(Mat orig, MMSolver &MMS, MatF &KP,bool center=false)
{

	auto params = MMS.params;
	auto Face2 = MMS.FMFull.Generate(MMS.SX, MMS.EX);
//	MatF projected = Projection(params, Face2);
    MatF projected = MMS.PerspectiveProjection(params, Face2);
    if(center){
        projected.col(0).array()+=orig.cols/2;
        projected.col(1).array()+=orig.rows/2;
    }
	auto image = orig.clone();
	auto image2 = orig.clone();

	auto Ev = MMS.FMFull.Ev;

	for (size_t i = 0; i < Ev.rows(); i++)
	{
		int i1 = Ev(i, 0);
		int i2 = Ev(i, 1);

		auto x = projected(i1, 0);
		auto y = projected(i1, 1);

		auto x2 = projected(i2, 0);
		auto y2 = projected(i2, 1);

		line(image, Point(x, y), Point(x2, y2), Scalar(0, 0, 255, 255), 1);
		//image.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
		//circle(image, Point(x, y), 1, Scalar(0, 0, 255), -1);
	}


	auto Face = MMS.FM.Generate(MMS.SX, MMS.EX);
//	projected = Projection(params, Face);
    projected = MMS.PerspectiveProjection(params, Face);
    if(center){
        projected.col(0).array()+=orig.cols/2;
        projected.col(1).array()+=orig.rows/2;
    }
	for (size_t i = 0; i < projected.rows(); i++)
	{
		auto x = projected(i, 0);
		auto y = projected(i, 1);
        circle(image, Point(x, y), 8, Scalar(255, 0, 0, 255), -1, CV_AA);
	}

	if (MMS.USEWEIGHT)
	{
		for (size_t i = 0; i < MMS.SkipList.size(); i++)
		{
			int i2 = MMS.SkipList[i];
			auto x = projected(i2, 0);
			auto y = projected(i2, 1);
            circle(image, Point(x, y), 24, Scalar(255, 0, 0, 255), 1, CV_AA);
		}
	}
    if(center){
        KP.col(0).array()+=orig.cols/2;
        KP.col(1).array()+=orig.rows/2;
    }
	for (size_t i = 0; i < KP.rows(); i++)
	{
		auto x = KP(i, 0);
		auto y = KP(i, 1);

        circle(image, Point(x, y), 8, Scalar(0, 255, 0, 255), -1, CV_AA);
	}

	//imshow("IMG", image / 2 + image2 / 2);
	return image / 2 + image2 / 2;
}





// Apply affine transform calculated using srcTri and dstTri to src
inline void applyAffineTransform(Mat &warpImage, Mat &src, vector<Point2f> &srcTri, vector<Point2f> &dstTri)
{
	// Given a pair of triangles, find the affine transform.
	Mat warpMat = getAffineTransform(srcTri, dstTri);

	// Apply the Affine Transform just found to the src image
	warpAffine(src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}


// Warps and alpha blends triangular regions from img1 and img2 to img
inline void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> &t1, vector<Point2f> &t2)
{
	TODO // Need to make sure rect is in Mat.
	try
	{
		Rect r1 = boundingRect(t1);
		Rect r2 = boundingRect(t2);

		//cout << r1 << r2 << endl;
		// Offset points by left top corner of the respective rectangles
		vector<Point2f> t1Rect, t2Rect;
		vector<Point> t2RectInt;
		for (int i = 0; i < 3; i++)
		{

			t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
			t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
			t2RectInt.push_back(Point(t2[i].x - r2.x, t2[i].y - r2.y)); // for fillConvexPoly

		}

		// Get mask by filling triangle
		Mat mask = Mat::zeros(r2.height, r2.width, CV_8UC3);
		fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 8, 0);

		// Apply warpImage to small rectangular patches
		Mat img1Rect;
		img1(r1).copyTo(img1Rect);

		Mat img2Rect = Mat::zeros(r2.height, r2.width, img1Rect.type());

		applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);

		multiply(img2Rect, mask, img2Rect);
		multiply(img2(r2), Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
		img2(r2) = img2(r2) + img2Rect;
	}
	catch (const std::exception& e) { 
		std::cout << e.what(); 
		//fillConvexPoly(img2, t2, Scalar(0, 0, 255), 16, 0);
	}
	


}


inline Mat MMSTexture(Mat orig, MMSolver &MMS, int W, int H,bool center=false, bool doubleface = true)
{
	auto image = orig.clone();
	Mat texture = Mat::zeros(H, W, CV_8UC3);

	auto params = MMS.params;
	auto Face2 = MMS.FMFull.Generate(MMS.SX, MMS.EX);
//	MatF projected = Projection(params, Face2);
    MatF projected = MMS.PerspectiveProjection(params, Face2);
    if(center){
        projected.col(0).array()+=orig.cols/2;
        projected.col(1).array()+=orig.rows/2;
    }

	auto TRI = MMS.FMFull.TRIUV;
	auto UV = MMS.FMFull.UV;

	for (size_t t = 0; t < TRI.rows(); t++)
	{

		vector<Point2f> t1;
		vector<Point2f> t2;
		for (size_t i = 0; i < 3; i++)
		{
			int j = TRI(t, i);
			auto x = projected(j, 0);
            if(x<0)x=0;
            if(x>=orig.cols)x=orig.cols-1;
			auto y = projected(j, 1);
            if(y<0)y=0;
            if(y>=orig.rows)y=orig.rows-1;
			auto u = (UV(j, 0)) * (W - 1);
			auto v = (1 - UV(j, 1)) * (H - 1);
			t1.push_back(Point2f(x, y));
			t2.push_back(Point2f(u, v));

			//cout << Point2f(x, y) << Point2f(u, v) << endl;
		}

		auto c = (t1[2] - t1[0]).cross(t1[1] - t1[0]);
		
		if (doubleface || c > 0)
		{
			warpTriangle(image, texture, t1, t2);
		}
	}

	return texture;

}


inline Mat MMSNormal(Mat orig, MMSolver &MMS, int W, int H)
{
	auto image = orig.clone();
	Mat texture = Mat::zeros(H, W, CV_8UC3);

	auto params = MMS.params;
	auto Face2 = MMS.FMFull.Generate(MMS.SX, MMS.EX);
	Face2 = Rotation(params, Face2);
	auto TRI = MMS.FMFull.TRIUV;
	auto UV = MMS.FMFull.UV;

	for (size_t t = 0; t < TRI.rows(); t++)
	{

		vector<Point3f> t1;
		vector<Point> t2;
		for (size_t i = 0; i < 3; i++)
		{
			int j = TRI(t, i);
			auto x = Face2(j, 0);
			auto y = Face2(j, 1);
			auto z = Face2(j, 2);

			auto u = (UV(j, 0)) * (W - 1);
			auto v = (1.0 - UV(j, 1)) * (H - 1);
			t1.push_back(Point3f(x, y, z));
			t2.push_back(Point(u, v));

			//cout << Point2f(x, y) << Point2f(u, v) << endl;
		}

		Point3f c = (t1[2] - t1[0]).cross(t1[1] - t1[0]);
		auto normal = c / norm(c);

		/*if (c.z > 0)
		{
			float n = ((c.z / norm(c))) * 255;

			fillConvexPoly(texture, t2, Scalar(n,n,n), 8, 0);
		}*/

		{
			fillConvexPoly(texture, t2, Scalar(normal.x + 1.0, normal.y + 1.0, normal.z + 1.0) * 128, 8, 0);
		}

	}

	return texture;

}



#include <sstream>
#include <fstream>  
#include "cnpy.h"


inline void FMObj(Mat texture, FaceModel &FM, string folder, string filename0)
{
	string filename = folder + filename0;
	//Mat texture = MMSTexture(orig, MMS, 1024, 1024);
	imwrite(filename + ".png", texture);

	//cout << Shape(MMS.SX) << Shape(MMS.EX) << endl;
	auto Face = FM.GeneratedFace;
	auto TRI = FM.TRI;
	auto TRIUV = FM.TRIUV;
	auto UV = FM.UV;

	//string numpyfile = filename + ".npz";


	//cnpy::npz_save(numpyfile, "SX", MMS.SX.data(), { (unsigned long long)MMS.SX.rows(), (unsigned long long)MMS.SX.cols() }, "w"); //"w" overwrites any existing file
	//cnpy::npz_save(numpyfile, "EX", MMS.EX.data(), { (unsigned long long)MMS.EX.rows(), (unsigned long long)MMS.EX.cols() }, "a"); //"a" appends to the file we created above

	{
		std::stringstream ss;


		ss << "mtllib " << filename0 << ".mtl" << endl;
		ss << "o FaceObject" << endl;

		int N = Face.rows();
		for (size_t i = 0; i < N; i++)
		{
			ss << "v " << Face(i, 0) << " " << Face(i, 1) << " " << Face(i, 2) << endl;
		}

		N = UV.rows();
		for (size_t i = 0; i < N; i++)
		{
			ss << "vt " << UV(i, 0) << " " << UV(i, 1) << endl;
		}

		ss << "usemtl material_0" << endl;
		ss << "s 1" << endl;

		N = TRI.rows();
		for (size_t i = 0; i < N; i++)
		{
			ss << "f " << TRI(i, 0) + 1 << "/" << TRIUV(i, 0) + 1 << " "
				<< TRI(i, 1) + 1 << "/" << TRIUV(i, 1) + 1 << " "
				<< TRI(i, 2) + 1 << "/" << TRIUV(i, 2) + 1 << " "
				<< endl;
		}


		std::string input = ss.str();

		std::ofstream out(filename + ".obj", std::ofstream::out);
		out << input;
		out.close();
	}

	{
		std::stringstream ss;


		ss << "mtllib " << filename0 << ".mtl" << endl;
		ss << "o FaceObject" << endl;

		ss << "newmtl material_0" << endl;
		ss << "	Ns 0.000000" << endl;
		ss << "Ka 0.200000 0.200000 0.200000" << endl;
		ss << "Kd 0.639216 0.639216 0.639216" << endl;
		ss << "Ks 1.000000 1.000000 1.000000" << endl;
		ss << "Ke 0.000000 0.000000 0.000000" << endl;
		ss << "Ni 1.000000" << endl;
		ss << "d 1.000000" << endl;
		ss << "illum 2" << endl;
		ss << "map_Kd " << filename0 + ".png" << endl;



		std::string input = ss.str();

		std::ofstream out(filename + ".mtl", std::ofstream::out);
		out << input;
		out.close();
	}


}


inline void MMSObj(Mat orig, MMSolver &MMS, string folder, string filename0,bool center=false)
{
	string filename = folder + filename0;
    Mat texture = MMSTexture(orig, MMS, 1024,1024,center);
	imwrite(filename + ".png", texture);

	auto Face = MMS.FMFull.Generate(MMS.SX, MMS.EX);
	auto TRI = MMS.FMFull.TRI;
	auto TRIUV = MMS.FMFull.TRIUV;
	auto UV = MMS.FMFull.UV;

	string numpyfile = filename + ".npz";
	

	cnpy::npz_save(numpyfile, "SX", MMS.SX.data(), { (unsigned long long)MMS.SX.rows(), (unsigned long long)MMS.SX.cols() }, "w"); //"w" overwrites any existing file
	cnpy::npz_save(numpyfile, "EX", MMS.EX.data(), { (unsigned long long)MMS.EX.rows(), (unsigned long long)MMS.EX.cols() }, "a"); //"a" appends to the file we created above

	{
		std::stringstream ss;


		ss << "mtllib " << filename0 << ".mtl" << endl;
		ss << "o FaceObject" << endl;

		int N = Face.rows();
		for (size_t i = 0; i < N; i++)
		{
			ss << "v " << Face(i, 0) << " " << Face(i, 1) << " " << Face(i, 2) << endl;
			ss << "vt " << UV(i, 0) << " " << UV(i, 1) << endl;
		}

		ss << "usemtl material_0" << endl;
		ss << "s 1" << endl;

		N = TRI.rows();
		for (size_t i = 0; i < N; i++)
		{
			ss << "f " << TRI(i, 0) + 1 << "/" << TRIUV(i, 0) + 1 << " "
				<< TRI(i, 1) + 1 << "/" << TRIUV(i, 1) + 1 << " "
				<< TRI(i, 2) + 1 << "/" << TRIUV(i, 2) + 1 << " "
				<< endl;
		}


		std::string input = ss.str();

		std::ofstream out(filename + ".obj", std::ofstream::out);
		out << input;
		out.close();
	}

	{
		std::stringstream ss;


		ss << "mtllib " << filename0 << ".mtl" << endl;
		ss << "o FaceObject" << endl;

		ss << "newmtl material_0" << endl;
		ss << "	Ns 0.000000" << endl;
		ss << "Ka 0.200000 0.200000 0.200000" << endl;
		ss << "Kd 0.639216 0.639216 0.639216" << endl;
		ss << "Ks 1.000000 1.000000 1.000000" << endl;
		ss << "Ke 0.000000 0.000000 0.000000" << endl;
		ss << "Ni 1.000000" << endl;
		ss << "d 1.000000" << endl;
		ss << "illum 2" << endl;
		ss << "map_Kd " << filename0 + ".png" << endl;



		std::string input = ss.str();

		std::ofstream out(filename + ".mtl", std::ofstream::out);
		out << input;
		out.close();
	}


}


