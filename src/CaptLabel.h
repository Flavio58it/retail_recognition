#ifndef CaptLabel_h
#define CaptLabel_h

#include <string>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

struct CaptLabel
{
    double timeStamp;
    double xPos;
    double yPos;
    double theta;

    int image_id, label_id;
    
    Mat img;
    
    std::string ocr_text;
    std::string label;

    std::vector<Point> contour_points;
    RotatedRect rot_bbox;
    //bool match;


};

#endif
