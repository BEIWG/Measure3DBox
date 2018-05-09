// Measure3DBox.cpp : Defines the entry point for the console application.
//
//author：CuiYongTai
//Emal: cuiyt@beiwg.com


#include "stdafx.h"
#include <iostream>
#include <algorithm>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <OpenNI.h>

#define DEBUG   1
#define GET_REF 0
#define REF_DISTDANCE   2000.0
#define MAX_DEPTH       4000
#define SAMPLE_READ_WAIT_TIMEOUT        2000 //2000ms
#define INTRINSIC_FILE  "camera.yaml"

using namespace std;
using namespace cv;
using namespace openni;

struct Box{
        double width;
        double length;
        double height;
};

/**************************************************
O-------------------> X
|
|       left_up----------------------right_up
|       |                               |
|       |                               |
|       |                               |
|       |                               |
|       left_down---------------------right_down
V
Y

vector<Point> --> 0:left_up, 1:right_up, 2:left_down, 3:right_down
***************************************************/
static void SortRectPoints(vector<Point> &approx)
{
        Point left_up, left_down, right_up, right_down;
        int max = 0, min = 0x7fff, max_index = 0, min_index = 0;
        int right_up_x =0, right_up_index = 0;

        //find left_up and right_down point
        for (auto i = 0; i < approx.size(); i++)
        {
                if ((approx.at(i).x + approx.at(i).y) >= max)
                {
                        max = approx.at(i).x + approx.at(i).y;
                        right_down = approx.at(i);
                        max_index = i;
                }

                if ((approx.at(i).x + approx.at(i).y) < min)
                {
                        min = approx.at(i).x + approx.at(i).y;
                        left_up = approx.at(i);
                        min_index = i;
                }

        }

        //find left_down and right_up point
        for (auto i = 0; i < approx.size(); i++)
        {
                if (i == max_index || i == min_index)
                        continue;

                if (i == approx.size() - 1)
                {
                        if (max > approx.at(i).x)
                        {
                                right_up = approx.at(right_up_index);
                                left_down = approx.at(i);
                        }
                        else
                        {
                                right_up = approx.at(i);
                                left_down = approx.at(right_up_index);
                        }

                        break;
                }

                right_up_x = approx.at(i).x;
                right_up_index = i;
        }
        
        approx.clear();
        approx.push_back(left_up);
        approx.push_back(right_up);
        approx.push_back(left_down);
        approx.push_back(right_down);
}

static double angle(Point pt1, Point pt2, Point pt0)
{
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

/*
 According to the principle of pinhole projection, convert the object size in pixel unit to 
 mm unit, essentially, it is Coordinate transformation: opencv -> world
*/
static void ConvertUnitPixelToMM(Mat intrinsic, short width, short length, double depth, Box &box)
{
        double fx, fy;
        
        fx = intrinsic.at<double>(0, 0);
        fy = intrinsic.at<double>(1, 1);
        
        box.width = depth*width / fx;
        box.length = depth*length / fy;
        box.height = REF_DISTDANCE - depth;
        
        /*Normally, width is bigger than length*/
        if (box.width < box.length)
        {
                double tmp;
                tmp = box.width;
                box.width = box.length;
                box.length = tmp;
        }

}

static void drawSquares(Mat& image, const vector<vector<Point> >& squares)

{
        for (size_t i = 0; i < squares.size(); i++)
        {
                const Point* p = &squares[i][0];
                int n = (int)squares[i].size();

                //dont detect the border
                if (p->x > 3 && p->y > 3)
                        polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 1, LINE_AA);
        }
}

bool Measure3DBox(Mat cameraMatrix, Mat ref, Mat depth_frame, VideoFrameRef frame, Box &box)
{
        Mat dst;
        Mat diff;
        bool isFound = false;
        vector<vector<Point> >squares;

        //帧差  
        absdiff(depth_frame, ref, diff);
        //二值化
        threshold(diff, diff, 0x15, 0xff, THRESH_BINARY);
        //闭运算
        Mat element = getStructuringElement(MORPH_RECT, Size(90, 90));
        morphologyEx(diff, dst, MORPH_CLOSE, element);
        //查找轮廓
        vector<vector<Point>> contours;
        findContours(dst, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        //逼近近似矩形
        vector<Point> approx;
        for (size_t i = 0; i < contours.size(); i++)
        {
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.04, true);
                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if (approx.size() == 4 && fabs(contourArea(Mat(approx))) > 1000 &&
                        isContourConvex(Mat(approx)))
                {
                        double maxCosine = 0;
                        for (int j = 2; j < 5; j++)
                        {
                                // find the maximum cosine of the angle between joint edges
                                double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                                maxCosine = MAX(maxCosine, cosine);
                        }

                        // if cosines of all angles are small
                        // (all angles are ~90 degree) then write quandrange
                        // vertices to resultant sequence
                        if (maxCosine < 0.3)
                        {
                                squares.push_back(approx);
                        }

                        if (squares.size() == 0)
                                continue;

                        if (DEBUG)
                        {
                                for (auto jj = 0; jj < approx.size(); jj++)
                                {
                                        printf("point %d:x = %d, y = %d\n", jj, approx.at(jj).x, approx.at(jj).y);
                                }
                        }

                        //calculate box width and heith in opencv coordinate, unit is pixels
                        int distance[4];
                        distance[0] = sqrt(abs(approx.at(0).x - approx.at(1).x)*abs(approx.at(0).x - approx.at(1).x) + 
                                abs(approx.at(0).y - approx.at(1).y)*abs(approx.at(0).y - approx.at(1).y));

                        distance[1] = sqrt(abs(approx.at(1).x - approx.at(2).x)*abs(approx.at(1).x - approx.at(2).x) +
                                abs(approx.at(1).y - approx.at(2).y)*abs(approx.at(1).y - approx.at(2).y));

                        distance[2] = sqrt(abs(approx.at(2).x - approx.at(3).x)*abs(approx.at(2).x - approx.at(3).x) +
                                abs(approx.at(2).y - approx.at(3).y)*abs(approx.at(2).y - approx.at(3).y));

                        distance[3] = sqrt(abs(approx.at(0).x - approx.at(3).x)*abs(approx.at(0).x - approx.at(3).x) +
                                abs(approx.at(0).y - approx.at(3).y)*abs(approx.at(0).y - approx.at(3).y));

                        sort(distance, distance + 4);

                        double width = (distance[0] + distance[1]) / 2.0;
                        double length = (distance[2] + distance[3]) / 2.0;

                        //calculate depth, unit is mm
                        RotatedRect rect = minAreaRect(approx);
                        
                        if (DEBUG)
                        {
                                Point2f P[4];
                                rect.points(P);
                                for (int j = 0; j <= 3; j++)
                                {
                                        line(depth_frame, P[j], P[(j + 1) % 4], Scalar(255), 2);
                                }
                        }

                        DepthPixel* pDepth = (DepthPixel*)frame.getData();
                        double depth = 0;
                        int sampling_local = rect.center.x + rect.center.y * frame.getWidth();
                        depth = pDepth[sampling_local];

                        ConvertUnitPixelToMM(cameraMatrix, width, length, depth, box);
                        
                        isFound = true;
                }
        }

        //显示
        if (DEBUG)
        {
                // drawSquares(tmp, squares);
                imshow("diff", diff);
                imshow("dst", dst);
                imshow("ref", ref);
                imshow("depth_frame", depth_frame);

                waitKey(10);
        }

        return isFound;
}

int _tmain(int argc, _TCHAR* argv[])
{
        Box box;
        
        //Read ref image
        Mat ref = imread("ref.bmp", IMREAD_GRAYSCALE);
        if (ref.empty())
        {
                printf("Read Ref image Failed!\n");
                return -1;
        }
        
        //Read camera intrinsic file
        FileStorage fs;
        string filename = INTRINSIC_FILE;
        fs.open(filename, FileStorage::READ);
        if (!fs.isOpened()) {
                printf("Load %s Failed...\n", INTRINSIC_FILE);
                return -1;
        }

        Mat cameraMatrix;
        fs["camera_matrix"] >> cameraMatrix;
        
        Status rc = OpenNI::initialize();
        if (rc != STATUS_OK)
        {
                printf("Initialize failed\n%s\n", OpenNI::getExtendedError());
                return 1;
        }

        Device device;
        rc = device.open(ANY_DEVICE);
        if (rc != STATUS_OK)
        {
                printf("Couldn't open device\n%s\n", OpenNI::getExtendedError());
                return 2;
        }

        VideoStream depth;
        if (device.getSensorInfo(SENSOR_DEPTH) != NULL)
        {
                rc = depth.create(device, SENSOR_DEPTH);
                if (rc != STATUS_OK)
                {
                        printf("Couldn't create depth stream\n%s\n", OpenNI::getExtendedError());
                        return 3;
                }
        }

        rc = depth.start();
        if (rc != STATUS_OK)
        {
                printf("Couldn't start the depth stream\n%s\n", OpenNI::getExtendedError());
                return 4;
        }

        VideoFrameRef frame;

        while (1)
        {
                int changedStreamDummy;
                VideoStream* pStream = &depth;
                rc = OpenNI::waitForAnyStream(&pStream, 1, &changedStreamDummy, SAMPLE_READ_WAIT_TIMEOUT);
                if (rc != STATUS_OK)
                {
                        printf("Wait failed! (timeout is %d ms)\n%s\n", SAMPLE_READ_WAIT_TIMEOUT, OpenNI::getExtendedError());
                        continue;
                }

                rc = depth.readFrame(&frame);
                if (rc != STATUS_OK)
                {
                        printf("Read failed!\n%s\n", OpenNI::getExtendedError());
                        continue;
                }

                if (frame.getVideoMode().getPixelFormat() != PIXEL_FORMAT_DEPTH_1_MM && frame.getVideoMode().getPixelFormat() != PIXEL_FORMAT_DEPTH_100_UM)
                {
                        printf("Unexpected frame format\n");
                        continue;
                }

                Mat depth_frame(frame.getHeight(), frame.getWidth(), CV_16U, (void*)frame.getData());
                depth_frame.convertTo(depth_frame, CV_8U, 255.0 / MAX_DEPTH);

                if (GET_REF)
                {
                        imwrite("ref.bmp", depth_frame);
                        imshow("ref", depth_frame);
                        waitKey(10);
                        continue;
                }

                if (Measure3DBox(cameraMatrix, ref, depth_frame, frame, box))
                        printf("BOX width = %f, length = %f, height = %f\n", box.width, box.length, box.height);
                else
                        printf("No Box found...\n");
        }

        depth.stop();
        depth.destroy();
        device.close();
        OpenNI::shutdown();
        return 0;
}

