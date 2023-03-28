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
#ifndef QTCVTRANSFORM_H
#define QTCVTRANSFORM_H

//DLP 20230310 include chrono and timinglogger for capturing timings
#include <chrono>
#include "timinglogger.h"

#include <QApplication>
#include <QMainWindow>
#include "opencv2/opencv.hpp"
#include <string>
#include <dirent.h>
#include <iostream>
#include <vector>
#include <sstream>


cv::Mat QImage2Mat(const QImage *qimg);
QImage*  Mat2QImage(const cv::Mat img);

#endif // QTCVTRANSFORM_H
