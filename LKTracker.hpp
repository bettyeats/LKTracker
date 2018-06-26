//
//  LKTracker.hpp
//  CaffeClassifyServer
//
//  Created by betty on 2018/4/7.
//  Copyright © 2018年 RemarkMedia. All rights reserved.
//

#ifndef LKTracker_hpp
#define LKTracker_hpp

#include <stdio.h>
#include "utils.hpp"
#include <opencv2/opencv.hpp>

//Bounding Boxes
struct BoundingBox : public cv::Rect {
    BoundingBox(){}
    BoundingBox(cv::Rect r): cv::Rect(r){}
public:
    float overlap;        //Overlap with current Bounding Box
    int sidx;             //scale index
};

class LKTracker{
private:
    std::vector<cv::Point2f> pointsFB;
    cv::Size window_size;
    int level;
    std::vector<uchar> status;
    std::vector<uchar> FB_status;
    std::vector<float> similarity;
    std::vector<float> FB_error;
    float simmed;
    float fbmed;
    cv::TermCriteria term_criteria;
    float lambda;
    void normCrossCorrelation(const cv::Mat& img1,const cv::Mat& img2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);
    bool filterPts(std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2);
public:
    LKTracker();
    bool trackf2f(const cv::Mat& img1, const cv::Mat& img2,
                  std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);
    float getFB(){return fbmed;}
    void bbPredict(const std::vector<cv::Point2f>& points1,const std::vector<cv::Point2f>& points2,
              const BoundingBox& bb1,BoundingBox& bb2);
    void bbPoints(std::vector<cv::Point2f>& points,const BoundingBox& bb);
};


#endif /* LKTracker_hpp */
