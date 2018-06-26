//
//  LKTracker.cpp
//  CaffeClassifyServer
//
//  Created by betty on 2018/4/7.
//  Copyright © 2018年 RemarkMedia. All rights reserved.
//

#include "LKTracker.hpp"
using namespace cv;

LKTracker::LKTracker(){
    term_criteria = TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 20, 0.03);
    window_size = Size(4,4);
    level = 5;
    lambda = 0.5;
}


bool LKTracker::trackf2f(const Mat& img1, const Mat& img2,vector<Point2f> &points1, vector<cv::Point2f> &points2){
    if (points1.size()<1){
        printf("Points not available\n");
        return false;
    }
    //TODO!:implement c function cvCalcOpticalFlowPyrLK() or Faster tracking function
    //Forward-Backward tracking
    double begin = clock();
    calcOpticalFlowPyrLK( img1,img2, points1, points2, status,similarity, window_size, level, term_criteria, lambda, 0);
    double end1 = clock();
    std::cout << "1111 time : " << (end1 - begin)/CLOCKS_PER_SEC << "ms\n" << std::endl;
    calcOpticalFlowPyrLK( img2,img1, points2, pointsFB, FB_status,FB_error, window_size, level, term_criteria, lambda, 0);
    double end2 = clock();
    std::cout << "2222 time : " << (end2 - end1)/CLOCKS_PER_SEC << "ms\n" << std::endl;
    //Compute the real FB-error
    for( int i= 0; i<points1.size(); ++i ){
        FB_error[i] = norm(pointsFB[i]-points1[i]);
    }
    //Filter out points with FB_error[i] > median(FB_error) && points with sim_error[i] > median(sim_error)
    normCrossCorrelation(img1,img2,points1,points2);
    return filterPts(points1,points2);
}

void LKTracker::normCrossCorrelation(const Mat& img1,const Mat& img2, vector<Point2f>& points1, vector<Point2f>& points2) {
    Mat rec0(10,10,CV_8U);
    Mat rec1(10,10,CV_8U);
    Mat res(1,1,CV_32F);
    
    for (int i = 0; i < points1.size(); i++) {
        if (status[i] == 1) {
            getRectSubPix( img1, Size(10,10), points1[i],rec0 );
            getRectSubPix( img2, Size(10,10), points2[i],rec1);
            matchTemplate( rec0,rec1, res, CV_TM_CCOEFF_NORMED);
            similarity[i] = ((float *)(res.data))[0];
            
        } else {
            similarity[i] = 0.0;
        }
    }
    rec0.release();
    rec1.release();
    res.release();
}


bool LKTracker::filterPts(vector<Point2f>& points1,vector<Point2f>& points2){
    //Get Error Medians
    simmed = median(similarity);
    size_t i, k;
    std::cout << "points2.size is: " << points2.size() << "\n" << std::endl;
    for( i=k = 0; i<points2.size(); ++i ){
        if( !status[i])
            continue;
        if(similarity[i]> simmed){
            points1[k] = points1[i];
            points2[k] = points2[i];
            FB_error[k] = FB_error[i];
            k++;
        }
    }
    if (k==0)
        return false;
    points1.resize(k);
    points2.resize(k);
    FB_error.resize(k);
    
    fbmed = median(FB_error);
    for( i=k = 0; i<points2.size(); ++i ){
        if( !status[i])
            continue;
        if(FB_error[i] <= fbmed){
            points1[k] = points1[i];
            points2[k] = points2[i];
            k++;
        }
    }
    points1.resize(k);
    points2.resize(k);
    if (k>0)
        return true;
    else
        return false;
}

void LKTracker::bbPredict(const vector<cv::Point2f>& points1,const vector<cv::Point2f>& points2,
                    const BoundingBox& bb1,BoundingBox& bb2)    {
    int npoints = (int)points1.size();
    vector<float> xoff(npoints);
    vector<float> yoff(npoints);
    printf("tracked points : %d\n",npoints);
    for (int i=0;i<npoints;i++){
        xoff[i]=points2[i].x-points1[i].x;
        yoff[i]=points2[i].y-points1[i].y;
    }
    float dx = median(xoff);
    float dy = median(yoff);
    float s;
    if (npoints>1){
        vector<float> d;
        d.reserve(npoints*(npoints-1)/2);
        for (int i=0;i<npoints;i++){
            for (int j=i+1;j<npoints;j++){
                d.push_back(norm(points2[i]-points2[j])/norm(points1[i]-points1[j]));
            }
        }
        s = median(d);
    }
    else {
        s = 1.0;
    }
    float s1 = 0.5*(s-1)*bb1.width;
    float s2 = 0.5*(s-1)*bb1.height;
    printf("s= %f s1= %f s2= %f \n",s,s1,s2);
    bb2.x = round( bb1.x + dx -s1);
    bb2.y = round( bb1.y + dy -s2);
    bb2.width = round(bb1.width*s);
    bb2.height = round(bb1.height*s);
    printf("predicted bb: %d %d %d %d\n",bb2.x,bb2.y,bb2.br().x,bb2.br().y);
}

void LKTracker::bbPoints(vector<cv::Point2f>& points,const BoundingBox& bb){
    int max_pts=10;
    int margin_h=0;
    int margin_v=0;
    int stepx = ceil((bb.width-2*margin_h)/max_pts);
    int stepy = ceil((bb.height-2*margin_v)/max_pts);
    for (int y=bb.y+margin_v;y<bb.y+bb.height-margin_v;y+=stepy){
        for (int x=bb.x+margin_h;x<bb.x+bb.width-margin_h;x+=stepx){
            points.push_back(Point2f(x,y));
        }
    }
}
/*
 * old OpenCV style
 void LKTracker::init(Mat img0, vector<Point2f> &points){
 //Preallocate
 //pyr1 = cvCreateImage(Size(img1.width+8,img1.height/3),IPL_DEPTH_32F,1);
 //pyr2 = cvCreateImage(Size(img1.width+8,img1.height/3),IPL_DEPTH_32F,1);
 //const int NUM_PTS = points.size();
 //status = new char[NUM_PTS];
 //track_error = new float[NUM_PTS];
 //FB_error = new float[NUM_PTS];
 }
 
 
 void LKTracker::trackf2f(..){
 cvCalcOpticalFlowPyrLK( &img1, &img2, pyr1, pyr1, points1, points2, points1.size(), window_size, level, status, track_error, term_criteria, CV_LKFLOW_INITIAL_GUESSES);
 cvCalcOpticalFlowPyrLK( &img2, &img1, pyr2, pyr1, points2, pointsFB, points2.size(),window_size, level, 0, 0, term_criteria, CV_LKFLOW_INITIAL_GUESSES | CV_LKFLOW_PYR_A_READY | CV_LKFLOW_PYR_B_READY );
 }
 */

 /*void testLKT() {
    LKTracker tracker = LKTracker();
 // TODO: Add LKT
 if (interval) {
     if (preFrame.empty() || (preFrame.rows != img.rows))
         img.copyTo(preFrame);
     if (faces.FacePts[0].x != 0 ) {
         for(int i = 0; i < 8; i++) {
             float x = faces.FacePts[i].x;
             float y = faces.FacePts[i].y;
             pts1.push_back(cv::Point2f(x,y));
         }
 
         pbox = this->current_shape;
         tracker.bbPoints(pts1, pbox);
         // output pts2, input pts1
         bTracked = tracker.trackf2f(preFrame, img, pts1, pts2);
         if (pts1.size() == 0)
             bTracked = false;
         else
             tracker.bbPredict(pts1, pts2, pbox, pbox2);
         if (tracker.getFB() > 10 || pbox2.x > img.rows || pbox2.y > img.rows || pbox2.br().x < 1 || pbox2.br().y < 1){
            bTracked = false;
          }
     }
    }
 }
*/
 

