/*
 *  ASML - Skin Lesion Image Segmentation Using Delaunay Triangulation for
 *  Melanoma Detection
 *
 *
 *  Written by Andrea Pennisi
 *
 *  Please, report suggestions/comments/bugs to
 *  andrea.pennisi@gmail.com
 *
 */

#include "skindetection.h"

#include "kernels/kernel.cuh"

#include <opencv2/cudaimgproc.hpp> //for opencv cuda version of function

//#define USE_SERIAL
//#define USE_NAIVE
#define USE_OPTIMIZED

SkinDetection::SkinDetection(const cv::Mat &img)
{
    this->img = img;

    Y_MIN  = 0;
    Y_MAX  = 255;
    Cr_MIN = 133;
    Cr_MAX = 145;
    Cb_MIN = 77;
    Cb_MAX = 127;

    Hmin = 0;
    Hmax = 160;
    smin = 100;
    smax = 255;
    vmin = 0; //60
    vmax = 15;
}


void SkinDetection::setParameters(const int &Y, const int &Cr, const int &Cb, const int &H, const int &s, const int &v) {
    Y_MIN = Y;
    Cr_MIN = Cr;
    Cb_MIN = Cb;

    Hmax = H;
    smin = s;
    vmax = v;
}

void SkinDetection::compute() {
	logExecTimes.logStart("SkinDetection::compute cuda");

    cv::Mat img_skin = img.clone();

    std::vector<cv::Mat> channels;
    cv::Mat img_hist_equalized;
	//DLP 20230310 updated to use current constant naming convention
    
    //cv::cvtColor(img, img_hist_equalized, cv::COLOR_BGR2YCrCb); //change the color image from BGR to YCrCb format
	
	//Get BGR image height, width, channels
    int heightBGR = img.rows;
    int widthBGR = img.cols;
    int imageChannelsBGR = img.channels();
    //declare working mats
    cv::Mat inBGR, inYCrCb, outYCrCb, outBGR;
    //creat host array pointers
    unsigned char *inImageDataBGR, *inImageDataYCrCb, *outImageDataBGR, *outImageDataYCrCb;

    //cout << "img - channels: " << imageChannelsBGR << "width: " << widthBGR <<  "height: " << heightBGR << endl;
	
    //Convert input mat to unsigned chars and set pointer to it
    img.copyTo(inBGR);
    inImageDataBGR = inBGR.ptr<unsigned char>();
    
    //Convert output mat to unsigned chars and set pointer to it
    outYCrCb.create(heightBGR, widthBGR, CV_8UC3);
    outImageDataYCrCb = outYCrCb.ptr<unsigned char>();
    
    //call kernel that converts bgr to ycrcb
    BGR2YCrCb_kernel_wrapper(inImageDataBGR, outImageDataYCrCb, widthBGR, heightBGR, imageChannelsBGR);
    
    //Converting output array back into Mat
    cv::Mat outYCrCb_temp(heightBGR, widthBGR, CV_8UC3, outImageDataYCrCb);
    outYCrCb_temp.copyTo(img_hist_equalized);
    //cout << "img_hist_equalized - channels: " << img_hist_equalized.channels()  << "width: " << img_hist_equalized.cols <<  "height: " << img_hist_equalized.rows << endl;
 
    //********************************
    // end substitution for BGR to YCrCb
    //********************************
	
    cv::split(img_hist_equalized,channels); //split the image into channels

    cv::equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)

    cv::merge(channels,img_hist_equalized); //merge 3 channels including the modified 1st channel into one image

	//DLP 20230310 updated to use current constant naming convention
    //cv::cvtColor(img_hist_equalized, img_hist_equalized, CV_YCrCb2BGR);
    
    //cv::cvtColor(img_hist_equalized, img_hist_equalized, cv::COLOR_YCrCb2BGR);
    
    //********************************
    //Substitution for YCrCb to BGR
    //********************************

    //Get YCrCb image height, width, channels
    int heightYCrCb = img_hist_equalized.rows;
    int widthYCrCb = img_hist_equalized.cols;
    int imageChannelsYCrCb = img_hist_equalized.channels();
	
    //Convert input mat to unsigned chars and set pointer to it
    img_hist_equalized.copyTo(inYCrCb);
    inImageDataYCrCb = inYCrCb.ptr<unsigned char>();
    
    //Convert output mat to unsigned chars and set pointer to it
    outBGR.create(heightYCrCb, widthYCrCb, CV_8UC3);
    outImageDataBGR = outBGR.ptr<unsigned char>();
    
    //call kernel that converts ycrcb to bgr
    YCrCb2BGR_kernel_wrapper(inImageDataYCrCb, outImageDataBGR, widthYCrCb, heightYCrCb, imageChannelsYCrCb);
    
    //Converting output array back into Mat
    cv::Mat outBGR_temp(heightYCrCb, widthYCrCb, CV_8UC3, outImageDataBGR);
    outBGR_temp.copyTo(img_hist_equalized);
    //cout << "img_hist_equalized,3 - channels: " << img_hist_equalized.channels()  << "width: " << img_hist_equalized.cols <<  "height: " << img_hist_equalized.rows << endl;

    //********************************
    // end substitution for YCrCb to BGR
    //********************************

    img = img_hist_equalized.clone();

    //Hair Removal
    cv::Mat sel = cv::getStructuringElement(cv::MORPH_RECT , cv::Size(11,11));
    cv::morphologyEx(img, img, cv::MORPH_CLOSE,  sel);

	//DLP 20230310 updated to use current constant naming convention
    //cv::cvtColor(img, ycrcb, CV_BGR2YCrCb);
    cv::cvtColor(img, ycrcb, cv::COLOR_BGR2YCrCb);
    cv::inRange(ycrcb,cv::Scalar(Y_MIN,Cr_MIN,Cb_MIN),cv::Scalar(Y_MAX,Cr_MAX,Cb_MAX),bw);

    img_skin = cv::Scalar(0, 0, 0);

    for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
            if((int)bw.at<uchar>(i, j) == 0)
                img_skin.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(i, j);
        }
    }
	
#ifdef USE_SERIAL
	cv::cvtColor(img_skin, hsv, cv::COLOR_BGR2HSV);
#endif

#ifdef USE_OPTIMIZED
	//****************************************
	//Substitution for bgr to hsv - optimized
	//****************************************
	cv::cuda::GpuMat gpu_img_skin; gpu_img_skin.upload(img_skin);
	
	int height = img_skin.rows;
    int width = img_skin.cols;

	hsv.create(height, width, CV_8U);
	cv::cuda::GpuMat gpu_hsv; gpu_hsv.upload(hsv);

	cv::cuda::cvtColor(gpu_img_skin, gpu_hsv, cv::COLOR_BGR2HSV);
	gpu_hsv.download(hsv);
	//****************************************
	//end substitution for bgr to hsv - optimized
	//****************************************
#endif

#ifdef USE_NAIVE
    //********************************
    //Substitution for bgr to hsv - naive
    //********************************

    //Getting height, width, channels
    int height = img_skin.rows;
    int width = img_skin.cols;
    int imageChannels = img_skin.channels();

    //allocate memory on device
    unsigned char *hostBGRImageData;
	unsigned char *hostHSVImageData;
	
	//Converting Mat to float
    hostBGRImageData = img_skin.ptr<unsigned char>();
    
    hsv.create(height, width, CV_8U);
    hostHSVImageData = hsv.ptr<unsigned char>();
    
    //call kernel that converts bgr to hsv
    bgr_to_hsv_kernel_v1_wrapper(hostBGRImageData, hostHSVImageData, width, height, imageChannels);
    
    //Converting output array back into Mat
    cv::Mat temp(height, width, CV_8U, hostHSVImageData);
	temp.copyTo(hsv);

	//********************************
	// end substitution for bgr to hsv - naive 
	//********************************
#endif

    for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j< img.cols; j++) {
            int valueH = hsv.at<cv::Vec3b>(i,j)[2];
            int valueS = hsv.at<cv::Vec3b>(i,j)[1];
            int valueV = hsv.at<cv::Vec3b>(i,j)[0];

            float normH = static_cast<float>(valueH) / static_cast<float>(valueH + valueS + valueV);
            float normS = static_cast<float>(valueS) / static_cast<float>(valueH + valueS + valueV);
            float normV = static_cast<float>(valueV) / static_cast<float>(valueH + valueS + valueV);

            hsv.at<cv::Vec3b>(i, j) = cv::Vec3f( normH*255, normS*255, normV*255 );
        }
    }

    cv::convertScaleAbs(hsv, hsv);

    bw = cv::Scalar(0);

    inRange(hsv, cv::Scalar(Hmin, smin, vmin), cv::Scalar(Hmax, smax, vmax), bw);
    cv::dilate(bw, bw, cv::Mat(), cv::Point(-1, -1), 6);

    cv::erode(bw, bw, cv::Mat(), cv::Point(-1, -1), 2);

    cv::Mat canny_output;

    cv::Canny( bw, canny_output, 50, 200, 3 );
    std::vector < std::vector<cv::Point> > contours;

    cv::findContours(bw, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    bw = cv::Scalar(0);
    cv::Moments moms = cv::moments(cv::Mat(contours[0]));
    double maxArea = moms.m00;
    int idx = 0;

    for (size_t contourIdx = 0; contourIdx < contours.size(); ++contourIdx)
    {
        moms = cv::moments(cv::Mat(contours[contourIdx]));
        if(moms.m00 > maxArea) {
            maxArea = moms.m00;
            idx = contourIdx;
        }
    }
	//DLP 20230310 updated to use current constant naming convention
    //cv::drawContours( bw, contours, idx, cv::Scalar(255), CV_FILLED );
    cv::drawContours( bw, contours, idx, cv::Scalar(255), cv::FILLED );

    img.copyTo(neo, bw);

    /*int countBw = 0;
    for(int i = 0; i < bw.rows; i++) {
        for(int j = 0; j < bw.cols; j++) {
            if((int)bw.at<uchar>(i, j) == 255) {
                countBw++;
            }
        }
    }

    perc = double(100 * countBw) / double(bw.rows * bw.cols);
    std::cout << "Perc: "<< perc << std::endl;*/
	logExecTimes.logStop("SkinDetection::compute cuda");
}
