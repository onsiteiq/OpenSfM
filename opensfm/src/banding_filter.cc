#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace csfm {


using namespace cv;
using namespace std;

void fftshift(const Mat& inputImg, Mat& outputImg);
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);
void synthesizeFilterH(Mat& inputOutput_H, Point center, int radius);
void calcPSD(const Mat& inputImg, Mat& outputImg, int flag = 0);
float medianMat(Mat input);

int IsBandingPresent(char *filename)
{
    Mat imgIn = imread(filename, IMREAD_GRAYSCALE);
    if (imgIn.empty()) //check whether the image is loaded or not
    {
        cout << "ERROR : Image cannot be loaded..!!" << endl;
        return -1;
    }

    imgIn.convertTo(imgIn, CV_32F);

    // it needs to process even image only
    Rect roi = Rect(0, 0, imgIn.cols & -2, imgIn.rows & -2);
    imgIn = imgIn(roi);

    // PSD calculation (start)
    Mat imgPSD;
    calcPSD(imgIn, imgPSD, 0);
    fftshift(imgPSD, imgPSD);
    // PSD calculation (stop)

    // it is observed we normally have horizontal banding at ~6Hz, and significant harmonics
    // at its odd multiples (3x, 5x).
    int testFreqs[] = {5, 6, 7, 19};
    for (int i=0; i<sizeof(testFreqs)/sizeof(testFreqs[0]); i++)
    {
        // test region is a narrow horizontal strip
        Mat testRegion = Mat(imgPSD, Rect(imgPSD.cols/2-5, imgPSD.rows/2-testFreqs[i], 11, 3)).clone();

        float medianVal = medianMat(testRegion);
        float ratio = imgPSD.at<float>(imgPSD.rows/2-testFreqs[i], imgPSD.cols/2) / medianVal;

        if(ratio > 20.0f)
        {
            return testFreqs[i];
        }
    }

    return -1;
}

py::object RunNotchFilter()
{
    py::list retval;

    Mat imgIn = imread("test3.jpg", IMREAD_GRAYSCALE);
    if (imgIn.empty()) //check whether the image is loaded or not
    {
        cout << "ERROR : Image cannot be loaded..!!" << endl;
        return retval;
    }

    imgIn.convertTo(imgIn, CV_32F);

    // it needs to process even image only
    Rect roi = Rect(0, 0, imgIn.cols & -2, imgIn.rows & -2);
    imgIn = imgIn(roi);

    // PSD calculation (start)
    Mat imgPSD;
    calcPSD(imgIn, imgPSD, 1);
    fftshift(imgPSD, imgPSD);
    normalize(imgPSD, imgPSD, 0, 255, NORM_MINMAX);
    // PSD calculation (stop)

    //H calculation (start)
    // NOTE: H = synthesizeFilterH returns a filter that's shifted with 0 at center.
    Mat H = Mat(roi.size(), CV_32F, Scalar(1));
    const int r = 1;
    synthesizeFilterH(H, Point(imgIn.cols/2, imgIn.rows/2-2), r);
    synthesizeFilterH(H, Point(imgIn.cols/2, imgIn.rows/2-3), r);
    synthesizeFilterH(H, Point(imgIn.cols/2, imgIn.rows/2-5), r);
    synthesizeFilterH(H, Point(imgIn.cols/2, imgIn.rows/2-15), r);
    synthesizeFilterH(H, Point(imgIn.cols/2, imgIn.rows/2-25), r);
    synthesizeFilterH(H, Point(imgIn.cols/2, imgIn.rows/2-35), r);
    synthesizeFilterH(H, Point(imgIn.cols/2, imgIn.rows/2-45), r);
    synthesizeFilterH(H, Point(imgIn.cols/2, imgIn.rows/2-55), r);
    synthesizeFilterH(H, Point(imgIn.cols/2, imgIn.rows/2-65), r);
    synthesizeFilterH(H, Point(imgIn.cols/2, imgIn.rows/2-75), r);
    synthesizeFilterH(H, Point(imgIn.cols/2, imgIn.rows/2-85), r);
    //H calculation (stop)

    // filtering (start)
    // NOTE: filtering runs on spectrum of image that's NOT shifted. but H was created above with 0 at center,
    // so shifted it back first, then do filtering
    Mat imgOut;
    fftshift(H, H);
    filter2DFreq(imgIn, imgOut, H);
    // filtering (stop)

    // write result and PSD out
    imgOut.convertTo(imgOut, CV_8U);
    normalize(imgOut, imgOut, 0, 255, NORM_MINMAX);
    imwrite("result.jpg", imgOut);
    imwrite("PSD.jpg", imgPSD);

    // write H out
    // NOTE: shift H again so it has 0 at center for display
    fftshift(H, H);
    normalize(H, H, 0, 255, NORM_MINMAX);
    imwrite("filter.jpg", H);

    return retval;
}

//! [fftshift]
void fftshift(const Mat& inputImg, Mat& outputImg)
{
    outputImg = inputImg.clone();
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;
    Mat q0(outputImg, Rect(0, 0, cx, cy));
    Mat q1(outputImg, Rect(cx, 0, cx, cy));
    Mat q2(outputImg, Rect(0, cy, cx, cy));
    Mat q3(outputImg, Rect(cx, cy, cx, cy));
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
//! [fftshift]

//! [filter2DFreq]
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI, DFT_SCALE);

    Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
    Mat complexH;
    merge(planesH, 2, complexH);
    Mat complexIH;
    mulSpectrums(complexI, complexH, complexIH, 0);

    idft(complexIH, complexIH);
    split(complexIH, planes);
    outputImg = planes[0];
}
//! [filter2DFreq]

//! [synthesizeFilterH]
void synthesizeFilterH(Mat& inputOutput_H, Point center, int radius)
{
    Point c2 = center, c3 = center, c4 = center;
    c2.y = inputOutput_H.rows - center.y;
    c3.x = inputOutput_H.cols - center.x;
    c4 = Point(c3.x,c2.y);
    circle(inputOutput_H, center, radius, 0, -1, 8);
    circle(inputOutput_H, c2, radius, 0, -1, 8);
    circle(inputOutput_H, c3, radius, 0, -1, 8);
    circle(inputOutput_H, c4, radius, 0, -1, 8);
}
//! [synthesizeFilterH]

// Function calculates PSD(Power spectrum density) by fft with two flags
// flag = 0 means to return PSD
// flag = 1 means to return log(PSD)
//! [calcPSD]
void calcPSD(const Mat& inputImg, Mat& outputImg, int flag)
{
    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);
    split(complexI, planes);            // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))

    planes[0].at<float>(0) = 0;
    planes[1].at<float>(0) = 0;

    // compute the PSD = sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)^2
    Mat imgPSD;
    magnitude(planes[0], planes[1], imgPSD);		//imgPSD = sqrt(Power spectrum density)
    pow(imgPSD, 2, imgPSD);							//it needs ^2 in order to get PSD
    outputImg = imgPSD;

    // logPSD = log(1 + PSD)
    if (flag)
    {
        Mat imglogPSD;
        imglogPSD = imgPSD + Scalar::all(1);
        log(imglogPSD, imglogPSD);
        outputImg = imglogPSD;
    }
}
//! [calcPSD]

float medianMat(Mat Input)
{
    Input = Input.reshape(0,1); // spread Input Mat to single row
    std::vector<float> vecFromMat;
    Input.copyTo(vecFromMat); // Copy Input Mat to vector vecFromMat
    std::sort( vecFromMat.begin(), vecFromMat.end() ); // sort vecFromMat

    if (vecFromMat.size()%2==0) {return (vecFromMat[vecFromMat.size()/2-1]+vecFromMat[vecFromMat.size()/2])/2;} // in case of even-numbered matrix
    return vecFromMat[(vecFromMat.size()-1)/2]; // odd-number of elements in matrix
}

}