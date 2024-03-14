#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

void computeGradient(const Mat& img, Mat& gradX, Mat& gradY) {
    Mat imgGray;
    cvtColor(img, imgGray, COLOR_BGR2GRAY);

    Sobel(imgGray, gradX, CV_32F, 1, 0);
    Sobel(imgGray, gradY, CV_32F, 0, 1);
    copyMakeBorder(gradX, gradX, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(gradY, gradY, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));
}

void KLT(const Mat& prevImg, const Mat& nextImg, vector<Point2f>& prevPts, vector<Point2f>& nextPts) {
    Mat gradX, gradY;
    computeGradient(prevImg, gradX, gradY);

    for (size_t i = 0; i < prevPts.size(); ++i) {
        int x = static_cast<int>(prevPts[i].x);
        int y = static_cast<int>(prevPts[i].y);
        int halfWindowSize = 10;
        Rect windowRect(x - halfWindowSize, y - halfWindowSize, 2 * halfWindowSize + 1, 2 * halfWindowSize + 1);

        windowRect &= Rect(0, 0, prevImg.cols, prevImg.rows);

        Point2f newPoint = prevPts[i];
        float minSSD = numeric_limits<float>::max();
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                Point2f candidatePoint = prevPts[i] + Point2f(dx, dy);
                float ssd = 0;
                for (int wx = -halfWindowSize; wx <= halfWindowSize; ++wx) {
                    for (int wy = -halfWindowSize; wy <= halfWindowSize; ++wy) {
                        int currX = x + wx;
                        int currY = y + wy;
                        if (currX >= 0 && currX < prevImg.cols && currY >= 0 && currY < prevImg.rows) {
                            float dx = gradX.at<float>(currY, currX);
                            float dy = gradY.at<float>(currY, currX);
                            float dt = nextImg.at<uchar>(currY, currX) - prevImg.at<uchar>(currY, currX);
                            float u = candidatePoint.x - x;
                            float v = candidatePoint.y - y;
                            float intensityError = dt - dx * u - dy * v;
                            ssd += intensityError * intensityError;
                        }
                    }
                }
                if (ssd < minSSD) {
                    minSSD = ssd;
                    newPoint = candidatePoint;
                }
            }
        }
        nextPts[i] = newPoint;
    }
}

int main() {
    try {
        VideoCapture capture("input.gif");
        if (!capture.isOpened()) {
            cerr << "Error: Unable to open GIF file." << endl;
            return -1;
        }

        Mat prevImg, nextImg;
        vector<Point2f> prevPts, nextPts;

        while (true) {
            capture >> nextImg;
            if (nextImg.empty()) break;

            if (prevImg.empty()) {
                prevImg = nextImg.clone();
                continue; 
            }

            if (prevPts.empty()) {
                Mat grayImg;
                cvtColor(prevImg, grayImg, COLOR_BGR2GRAY); // Convert to grayscale
                goodFeaturesToTrack(grayImg, prevPts, 100, 0.01, 10);
            }
            else {
                // Clear previous points
                prevPts.clear();

                // Detect new points in the current frame
                Mat grayImg;
                cvtColor(prevImg, grayImg, COLOR_BGR2GRAY); // Convert to grayscale
                goodFeaturesToTrack(grayImg, prevPts, 100, 0.01, 10);

                // Perform KLT optical flow tracking
                nextPts = prevPts;
                KLT(prevImg, nextImg, prevPts, nextPts);

                // Display results
                Mat resultImg = nextImg.clone();
                for (size_t i = 0; i < prevPts.size(); ++i) {
                    circle(resultImg, nextPts[i], 5, Scalar(0, 255, 0), -1);
                    line(resultImg, prevPts[i], nextPts[i], Scalar(0, 0, 255), 2);
                }

                imshow("Optical Flow", resultImg);
                waitKey(30);

                // Update for the next iteration
                swap(prevImg, nextImg);
            }
        }
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV error: " << e.what() << endl;
        return -1;
    }

    return 0;
}