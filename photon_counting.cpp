#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <random>
#include <iomanip>
#include <string>
#include <map>

using namespace std;
using namespace cv;


//주석 ctrl+k+c 해제 ctrl+k+u
int main()
{
	// mat grayscale_img;
	//Mat color_img = imread("c:/users/jinung/desktop/num2.2_re.jpg");
	Mat color_img = imread("/home/ubics/lena_dark.jpg");
	int rows = color_img.rows;
	int cols = color_img.cols;


	double minval, maxval;
	Point minloc, maxloc;
	// mat imgnormal;
	Mat grayscale_img(rows, cols, CV_64FC1);
	//Mat grayscale_img(rows, cols, CV_8UC1);
	Mat imgnormal(rows, cols, CV_64FC1);
	//Mat imgnormal(rows, cols, CV_8UC1);
	Mat an(rows, cols, CV_64FC1);
	Mat lambda(rows, cols, CV_64FC1);
	//Mat ck(rows, cols, cv_64fc1);
	//Mat nlm(rows, cols, cv_64fc1);

	
	cvtColor(color_img, grayscale_img, COLOR_RGB2GRAY);
	 grayscale_img.convertTo(grayscale_img, CV_64FC1, 1.f/255); //소수점으로 정규화 하기위해 cv_32f사용 flout32를 의미함
	 normalize(grayscale_img, imgnormal, 0.0, 1.0, NORM_MINMAX, -1, noArray()); //정규화 함수 0.0~1.0으로 소수점으로 이미지를 정규화함

	// normalize(grayscale_img, imgnormal, 0.0, 1.0, norm_minmax,-1, noarray()); //정규화 함수 0.0~1.0으로 소수점으로 이미지를 정규화함
	//normalize(grayscale_img, imgnormal, 1, 255, NORM_MINMAX, -1, noArray()); //정규화 함수 0.0~1.0으로 소수점으로 이미지를 정규화함
	

	

	int k = 0.1;
	//double np = 10000;
    double np = 1000000;
	int n = 10;


	// imgnormal = imgnormal/1000;
	double img_sum = sum(grayscale_img)[0]; //sum(a(:)) 행렬의 모든 값을 더한 값
	an = grayscale_img / img_sum; //an = a./sum(a(:)) 그레이 이미지행렬에 행렬의 모든 함을 나누었다
	
	//double lambda = 1000*grayscale_img;
	 lambda = an * np; //lambda = np.*an;
	 
	//double lambda_mean = mean(lambda)[0]; //mu = mean2(lambda) 평균을 구함
	//void meanstd = meanstddev(lambda,lambda_mean,lambda_std,inputarray mask=noarray());

	Scalar mean, stddev;
	meanStdDev(lambda, mean, stddev); //lambda에 대해서 평균과 표준편차를 반환한다

	stddev = stddev * stddev; // s = std2(lambda)^2;
	
	double lambda_mean = mean[0]; //lambda에 대한 평균의 배열에서 첫번째 것을 가져온다
	double lambda_stddev = stddev[0]; //lambda에 대한 표준편차의 배열에서 첫번째 것을 가져온다
	double alpha = (lambda_mean * lambda_mean) / lambda_stddev; //alpha = (mu^2)/s
	double beta = lambda_mean / lambda_stddev;

	
	//---------------------------푸아송
	
	//std::mt19937 gen(1701);		//일정한 랜덤 시드를 사용함 값이 일정
	std::random_device gen;			//랜덤시드를 사용함 그래서 값이 계속 바뀐다

	Mat temp(rows, cols, CV_64FC1);
	Mat poisson_mat(rows, cols, CV_64FC1);
	for(int k =0; k<n; k++)
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
			double p = lambda.at<double>(i,j);
			poisson_distribution<> distr(p);
			poisson_mat.at<double>(i, j) = distr(gen);
			}
		}
		
		temp = temp+poisson_mat;
	}
	cout << temp;
	
	
	
	
	//----------------축적하는부분 해결하면 된다.6월19일------------
	//cout << distr(gen) << "==poisson value" << endl;
	//float a = test(p_dist, samples);
	 //Mat a = Mat::zeros(rows, cols, CV_64FC1);
	 //Mat temp(rows, cols, CV_64FC1);
	 //Mat ck = Mat::zeros(rows, cols, CV_64FC1);
	 //Mat temp(rows, cols, CV_32SC1);
	//  for(int i = 0; i < 2; i++)
	//  {
	// 	 for (int j = 0; j < n; i++)
	// 	 {
	// 		 temp = temp +ck;
			 
	// 	 }
	//  }

    //  for(int i =0; i<n; i++)
    //  {
    //      temp = temp+poisson_mat;
		 
    //  }
        


	 Mat MLE(rows, cols, CV_64FC1);
	 MLE = temp / (n * np);
	


		//minMaxLoc(an, &minval, &maxval, &minloc, &maxloc);
		//Mat an_re = an / maxval; //an_re = an / max(an(:))



	minMaxLoc(MLE, &minval, &maxval, &minloc, &maxloc);
	Mat MLE_re = MLE / maxval;
	
	 cout << "maxval ==== " << maxval << endl;


	// imgnormal = (255 / maxval) * imgsrc;
	// minmaxloc(imgsrc,&minval,&maxval,&minloc,&maxloc);

	//fastnlmeansdenoising(an_re, nlm, 30.0, 7, 21);

	//Mat denoise;
	//fastNlMeansDenoising(MLE_re, denoise, 10.0, 7, 21);

	//randu(imgnormal, Scalar::all(0), scalar::all(255));

	//gaussianblur(an_re, denoise , size(7, 7), 0);
	//cout << denoise << endl;

	//imshow("an_img", an_re);
	imshow("color_img", MLE_re);
	//imshow("grayscale_img", grayscale_img);
	imshow("normalization_img", temp);
	//imshow("denoise", denoise);

	waitKey(0);
	return 0;
}
[출처] poisson final code|작성자 as8121

