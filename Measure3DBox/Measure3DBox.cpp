// Measure3DBox.cpp : Defines the entry point for the console application.
//
//author：CuiYongTai
//Emal: cuiyt@beiwg.com


#include "stdafx.h"
#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <OpenNI.h>

#define DEBUG   1
#define GET_REF 0
#define REF_DISTDANCE   2000
#define MAX_DEPTH       4000
#define SAMPLE_READ_WAIT_TIMEOUT        2000 //2000ms
#define INTRINSIC_FILE  "camera.yml"

using namespace std;
using namespace cv;
using namespace openni;

struct Box{
        short width;
        short length;
        short height;
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
        unsigned int max = 0, min = 0xffff, max_index = 0, min_index = 0;

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
                                right_up = approx.at(max_index);
                                left_down = approx.at(i);
                        }
                        else
                        {
                                right_up = approx.at(i);
                                left_down = approx.at(max_index);
                        }
                        break;
                }

                max = approx.at(i).x;
                max_index = i;
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

static void ConvertUnitPixelToMM(Mat intrinsic, int &w, int &h)
{

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

bool Measure3DBox(Mat ref, Mat depth_frame, VideoFrameRef frame, Box &box)
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
                if (DEBUG)
                {
                        printf("appros.size = %d\n", approx.size());
                        printf("approx.area = %f\n", contourArea(Mat(approx)));
                        printf("is contourconvex %d\n", isContourConvex(Mat(approx)));
                }

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
                        
                        SortRectPoints(approx);
                        /********************************************************************
                        *calc rect width and heigth in opencv coordinate
                        *width = {(right_up - left_up) + (right_down - left_down)} / 2
                        *height = {(left_down - left_up) + (right_down - right_up)} / 2
                        *********************************************************************/
                        if (DEBUG)
                        {
                                for (auto jj = 0; jj < approx.size(); jj++)
                                {
                                        printf("point %d:x = %d, y = %d\n", jj, approx.at(jj).x, approx.at(jj).y);
                                }
                        }

                        //calculate box width and heith in opencv coordinate, unit is pixels
                        Point left_up = approx.at(0);
                        Point left_down = approx.at(2);
                        Point right_up = approx.at(1);
                        Point right_down = approx.at(3);

                        int width = ((right_up.x - left_up.x) + (right_down.x - left_down.x)) / 2;
                        int length = ((left_down.y - left_up.y) + (right_down.y - right_up.y)) / 2;

                        if (width < length)
                        {
                                int temp = width;
                                width = length;
                                length = temp;
                        }

                        //calculate depth, unit is mm
                        //sampling four point at box plan and calculate average value
                        /*******************************************************************
                        -----------------------
                        |                     |
                        |--------1------2-----|
                        |                     |
                        |--------3-------4----|
                        |                     |
                        |---------------------|
                        ********************************************************************/
                        int sampling_x_local[2], sampling_y_local[2];
                        sampling_x_local[0] = left_up.x + (right_up.x - left_up.x) / 3;
                        sampling_x_local[1] = left_up.x + (right_up.x - left_up.x) * 2 / 3;

                        sampling_y_local[0] = left_up.y + (left_down.y - left_up.y) / 3;
                        sampling_y_local[1] = left_up.y + (left_down.y - left_up.y) * 2 / 3;

                        DepthPixel* pDepth = (DepthPixel*)frame.getData();
                        int depth = 0;
                        depth += pDepth[sampling_x_local[0] + sampling_y_local[0] * frame.getWidth()];
                        depth += pDepth[sampling_x_local[0] + sampling_y_local[1] * frame.getWidth()];
                        depth += pDepth[sampling_x_local[1] + sampling_y_local[0] * frame.getWidth()];
                        depth += pDepth[sampling_x_local[1] + sampling_y_local[1] * frame.getWidth()];
                        depth /= 4;

                        int height = REF_DISTDANCE - depth;

                        //ConvertUnitPixelToMM()

                        box.width = width;
                        box.length = length;
                        box.height = height;
                        
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
        FileStorage fs(string(INTRINSIC_FILE), FileStorage::READ);
        if (!fs.isOpened()) {
                printf("Load %d Failed...\n", INTRINSIC_FILE);
                return -1;
        }

        Mat cameraMatrix, distCoeffs;
        fs["camera_matrix"] >> cameraMatrix;
        fs["distortion_coefficients"] >> distCoeffs;
        cout << cameraMatrix << endl;
        cout << distCoeffs << endl;

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

                if (Measure3DBox(ref, depth_frame, frame, box))
                        printf("###BOX width = %d, length = %d, height = %d\n", box.width, box.length, box.height);
                else
                        printf("No Box found...\n");
        }

        depth.stop();
        depth.destroy();
        device.close();
        OpenNI::shutdown();
        return 0;
}

