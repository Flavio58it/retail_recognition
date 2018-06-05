#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/String.h>
#include <tf/transform_listener.h>
#include <tf/tf.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#include <hunspell/hunspell.h>
#include <leptonica/allheaders.h>
#include <tesseract/baseapi.h>

#include <fstream>
#include <iostream>
#include <stdio.h>

#include <string>
#include <stdlib.h>
#include <vector>

#include <iterator>
#include <map>

#include "CaptLabel.h"
#include "SignLabel.h"
#include "SignLabel.cpp"

using namespace cv;

/********************************Global Variables**********************************************************************************/

/***********Subscribed Topics & Map Frame Parameters****************/

//subscribed topic names
std::string usb_cam_topic_name;

//webcam callback
cv_bridge::CvImagePtr cv_ptr;
std::string enable_flag=" ";
int img_counter =0;

/*******************************************************************/

/***********Sign Label Detection Parameters************************/

//resizing
int raw_width_resize;
int raw_height_resize;

//image pre-processing
int minThreshold;
int threshRatio;
int window_size;
int blur_colRange;
int blur_spatialRange;

//contour detection
float min_scaleFactor;
float max_scaleFactor;
float label_width;
float label_height;
float area_img;

//deskew post-processing
int edge_offset;
int cropWidth;
int cropHeight;
int filter_size;
int bt_window_size;
int element_size;

/*******************************************************************/


/***********Sign Label Classification Parameters********************/

std::string aisle_categories_list;
const char* aff_location = "/usr/share/hunspell/en_GB.aff";
const char* dic_location = "/usr/share/hunspell/en_GB.dic";

std::map<std::string, int> sL2Aisle;

int max_edit_distance;
float prob_threshold;

/*******************************************************************/


/**********************************************************************************************************************************/

/***********************************Function Declarations****************************************************************************/
std::vector<std::string> loadSignLabelNames();
std::vector<std::string> createWordList(std::vector<std::string> namelist);

void usb_cam_callback(const sensor_msgs::ImageConstPtr& rawImg_msg);

Mat preprocessImage(Mat* img1);
std::vector< std::vector<Point> >findSignLabelContours(Mat& img2);
std::vector<Point> sortContourPoints(std::vector<Point> contourPoints);
double calcAspectRatio(std::vector< Point > point_array);
std::vector<RotatedRect> contourstoRotBoxes(std::vector< std::vector<Point> >& rectContours2);
std::vector <Mat> extractLabels(Mat& img3, std::vector<RotatedRect> rotRect1, std::vector< std::vector<Point> >& rectContours3 );

std::string extractOCRText(Mat img);
std::string stringComparison(std::string candidate, std::vector<std::string> words);
int edit_distance(std::string sourceWord, std::string targetWord);
string classifySignLabel(std::string ucl_label,std::vector<SignLabel> target_labels);

void drawFeatures_no_labels(Mat& img4, vector<RotatedRect>& rotRect2, vector< vector<Point> >& rectContours4);
void drawFeatures(Mat img, std::vector<CaptLabel> label_array);
/**********************************************************************************************************************************/

/************************************Main Function**********************************************************************************/
int main(int argc, char** argv)
{
    //node initialization
    ros::init(argc, argv, "sign_recognition_node");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    //get parameters from launch file
    nh.getParam("/sign_recognition_node/usb_cam_topic_name", usb_cam_topic_name);
    nh.getParam("/sign_recognition_node/raw_width_resize",raw_width_resize);
    nh.getParam("/sign_recognition_node/raw_height_resize",raw_height_resize);

    nh.getParam("/sign_recognition_node/aisle_categories_list",aisle_categories_list);

    nh.getParam("/sign_recognition_node/minThreshold",minThreshold);
    nh.getParam("/sign_recognition_node/threshRatio",threshRatio);
    nh.getParam("/sign_recognition_node/window_size",window_size);
    nh.getParam("/sign_recognition_node/blur_colRange", blur_colRange);
    nh.getParam("/sign_recognition_node/blur_spatialRange",blur_spatialRange);

    nh.getParam("/sign_recognition_node/min_scaleFactor",min_scaleFactor);
    nh.getParam("/sign_recognition_node/max_scaleFactor",max_scaleFactor);

    nh.getParam("/sign_recognition_node/label_width",label_width);
    nh.getParam("/sign_recognition_node/label_height",label_height);

    nh.getParam("/sign_recognition_node/edge_offset",edge_offset);
    nh.getParam("/sign_recognition_node/cropWidth",cropWidth);
    nh.getParam("/sign_recognition_node/cropHeight",cropHeight);
    nh.getParam("/sign_recognition_node/filter_size",filter_size);
    nh.getParam("/sign_recognition_node/bt_window_size",bt_window_size);
    nh.getParam("/sign_recognition_node/element_size",element_size);

    nh.getParam("/sign_recognition_node/max_edit_distance",max_edit_distance);
    nh.getParam("/sign_recognition_node/prob_threshold",prob_threshold);

    //create subscribers & publishers
    image_transport::Subscriber sub_usb_cam = it.subscribe(usb_cam_topic_name,1,usb_cam_callback);
    ros::Publisher pub_sign_labels = nh.advertise<std_msgs::String>("/aisle_sign_labels",100);

    //load master list of a priori sign label names
    std::vector<std::string> masterlist = loadSignLabelNames();
    std::cout<<"\n# of Labels: "<<masterlist.size();
    std::vector<SignLabel> apriori_labels;
    for(int k=0; k < masterlist.size();k++)
    {
        SignLabel temp_label(masterlist[k]);
        apriori_labels.push_back(temp_label);
    }

    //create list of individual words from label master list
    std::vector<std::string> wordlist = createWordList(masterlist);
    std::cout<<"\n# of Unique Words: "<<wordlist.size();

    Mat img_src, img_filter, img_bbox;
    std::vector<CaptLabel> captured_labels, classification_results;
    int label_counter = 0;
    double t_start, t_current;
    geometry_msgs::PoseStamped pose_robot;
    tf::TransformListener listener;

    t_start=ros::Time::now().toSec();
    while(ros::ok())
    {

        if(cv_ptr)
        {

            /****************************Sign Label Detection******************************************************/
            //resize to 800x600 resolution
            img_src = cv_ptr->image.clone();
            resize(img_src,img_src,Size(raw_width_resize,raw_height_resize),0,0,INTER_LINEAR);
            area_img = cv_ptr->image.size().area();

            //convert to grayscale, blur and detect edges
            img_filter = preprocessImage(&img_src);
            imshow("Canny Image", img_filter);

            //find rectangular contours
            std::vector< std::vector<Point> > rectContours = findSignLabelContours(img_filter);

            //find bounding boxes in image
            std::vector<RotatedRect> rotRect = contourstoRotBoxes(rectContours);

            //deskew & crop image
            Mat img_deSkew=img_src.clone();
            std::vector <Mat> img_segmt = extractLabels(img_deSkew, rotRect,rectContours);

            /******************************************************************************************************/

            /****************************Sign Label Classification*************************************************/
            CaptLabel temp;
            double yaw_angle;
            std::cout<<"\n**********Image "<<img_counter<<"**********";
            std::cout<<"\n"<<img_segmt.size()<<" label(s) found"<<"\n";

            for(int i=0; i < img_segmt.size(); i++)
            {
                label_counter++;

                std::cout<<"\n\tLabel "<<label_counter;

                //find raw OCR extract
                std::string ocr_text = extractOCRText(img_segmt[i]);
                std::cout<<"\n\tRaw OCR text: "<<ocr_text;

                //correct spelling mistakes
                std::string corrected_text = stringComparison(ocr_text,wordlist);
                std::cout<<"\n\tSpellchecked OCR text: "<<corrected_text;

                //make final classification
                std::string classified_text = classifySignLabel(corrected_text,apriori_labels);
                std::cout<<"\n\tClassification: "<<classified_text;

                //find aisle number
                std::map<std::string, int>::iterator it;
                it=sL2Aisle.find(classified_text);

                if(it!=sL2Aisle.end())
                    std::cout<<"\n\tAisle: "<<it->second<<"\n";
                else
                    std::cout<<"\n\tNo Aisle Found\n";

                //add classification, contour, rotated bounding box information to captured label object
                temp.image_id = img_counter;
                temp.label_id = label_counter;
                temp.img = img_segmt[i];
                temp.ocr_text = ocr_text;
                temp.label = classified_text;
                temp.contour_points = rectContours[i];
                temp.rot_bbox = rotRect[i];

                captured_labels.push_back(temp);
                classification_results.push_back(temp);

                //publish label
                if(temp.label!="N/A")
                {
                    std_msgs::String msg_label;
                    msg_label.data = temp.label;
                    pub_sign_labels.publish(msg_label);
                    std::cout<<"\n\tPublished Label";
                }

            }//*/
            /******************************************************************************************************/

            img_bbox=img_src.clone();

            //draw sign label contours
            //drawFeatures_no_labels(img_bbox,rotRect,rectContours);

            //draw labels & sign labels contours
            drawFeatures(img_bbox,captured_labels);

            std::cout<<"\n***************************";

            label_counter=0;

            rectContours.clear();
            rotRect.clear();
            img_segmt.clear();
            captured_labels.clear();
            cv_ptr.reset();
        }
        else
            std::cout<<"\nWaiting for usb_cam topic";

        ros::spinOnce();
    }

    return 0;

}
/**********************************************************************************************************************************/

/*******************************Functions******************************************************************************************/

/*load sign label names from external text file*/
std::vector<std::string> loadSignLabelNames()
{
    std::ifstream infile;
    infile.open(aisle_categories_list.c_str());

    std::vector<string> name_array, line_array;
    char line[100];
    char newline = '\n';
    //string s(newline);

    if(infile.is_open())
    {
        if(infile.peek() == std::ifstream::traits_type::eof())
        {
            std::cout<<"\nFile is empty";
            infile.close();
        }
        else
        {
            std::cout<<"\nImporting sign label information";
            while(infile.good())
            {
                infile.getline(line,256,'\n');
                string label_name(line);

                boost::algorithm::split(line_array,label_name,boost::is_any_of(","));

                if(line_array[0]!="")
                {
                    int num = atoi(line_array[1].c_str());
                    sL2Aisle[line_array[0]]=num;
                    name_array.push_back(line_array[0]);
                }
                //else
                    //std::cout<<"\nNewline character found";

                line_array.clear();
            }

            infile.close();
        }

    }

    else
    {
        std::cout<<"\nFile is not open";
        infile.close();
    }

    return name_array;
}

/*create a list of words that will be used later to compare against OCR output*/
std::vector<std::string> createWordList(std::vector<std::string> namelist)
{
    //std::cout<<"\nEntered createWordList function";
    bool found=false;
    std::vector<string> tempWord, word_array;

    for(int i=0; i<namelist.size(); i++)
    {
        //split multi-word label into a vector of individual words
        boost::algorithm::split(tempWord, namelist[i], boost::is_any_of(" "));

        for(int j=0; j < tempWord.size(); j++)
        {
            //std::cout<<"\n# of words: "<<tempWord.size();
            //determine if word already exists in wordlist
            for(int d=0; d<word_array.size(); d++)
            {
                if(tempWord[j]==word_array[d])
                {
                    found = true;
                    //std::cout<<"\n"<<tempWord[j]<< " already in list";
                    break;
                }
                else
                    found = false;
            }
            //add new words to wordlist
            if(!found)
            {
                //std::cout<<"\nAdding word to wordlist";
                word_array.push_back(tempWord[j]);
            }

        }

        tempWord.clear();
    }
    //std::cout<<"\nSize of Word Array: "<<word_array.size();
    return word_array;
}

/*convert ROS image format to OpenCv format and display converted image*/
void usb_cam_callback(const sensor_msgs::ImageConstPtr& rawImg_msg)
{
    try
    {
        cv_ptr=cv_bridge::toCvCopy(rawImg_msg, sensor_msgs::image_encodings::BGR8);
        //std::cout<<"\nCaptured Raw Image";
        waitKey(30);
        img_counter++;
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

}

/*convert raw image to grayscale, blur and perform edge detection*/
Mat preprocessImage(Mat* img1)
{
    Mat img_temp, img_gray, img_blur, img_canny, img_thresh;
    img_temp = ( (Mat) *(Mat*)img1);
    cvtColor(img_temp, img_gray, CV_BGR2GRAY );
    bilateralFilter(img_gray,img_blur,window_size,blur_colRange,blur_spatialRange);
    Canny(img_blur, img_canny, minThreshold, minThreshold*threshRatio, 3 );

    return img_canny;
}

/*find sign label contours in binary image*/
std::vector< std::vector<Point> >findSignLabelContours(Mat& img2)
{
    std::vector< std::vector<Point> > contours;
    std::vector<Point> tempContours,sortedContours;
    std::vector< std::vector<Point> > approxContours;

    const float label_aspect = (float)(label_width / label_height);
    findContours(img2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    //approximate each contour
    for(int i=0; i<contours.size();i++)
    {
        approxPolyDP(contours[i],tempContours,3,true);
        double tempArea=contourArea(tempContours,false);

        //only consider contours with 4 vertices and that have signficantly small areas
        if(tempContours.size()==4 && (tempArea > area_img*min_scaleFactor && tempArea < area_img*max_scaleFactor))
        {
            sortedContours = sortContourPoints(tempContours);
            double temp_ar = calcAspectRatio(sortedContours);

            //only keep contours that have an aspect ratio greater than ~4.5
            if(temp_ar >=label_aspect)
                approxContours.push_back(sortedContours);
        }

    }

    return approxContours;
}

/*sort contours in a CCW manner, starting with TL point*/
std::vector<Point> sortContourPoints(std::vector<Point> contourPoints)
{
    Point TL,BL,TR,BR;
    std::vector<Point> sortedPoints;

    //use bubble sort to organise points in ascending x-coordinate order
    for(int i =0; i< contourPoints.size();i++)
    {
        for(int j=0; j<contourPoints.size()-1; j++)
        {
            if (contourPoints[j].x > contourPoints[j+1].x )
            {
                Point temp = contourPoints[j+1];
                contourPoints[j+1]=contourPoints[j];
                contourPoints[j]=temp;

            }

        }

    }

    //assign TL and BL points
    //TL is the point with the smaller y coordinate
    //BL is the point with the larger y coordinate
    if(contourPoints[0].y < contourPoints[1].y)
    {
        TL = contourPoints[0];
        BL = contourPoints[1];
    }
    else
    {
        TL = contourPoints[1];
        BL = contourPoints[0];
    }

    //assign TR and BR points
    //TR is the point with the smaller euclidean distance relative to TL
    //BR is the point with the larger euclidean distance relative to TL
    double euclidean1 = sqrt( (TL.x-contourPoints[2].x)^2 + (TL.y-contourPoints[2].y)^2 );
    double euclidean2 = sqrt( (TL.x-contourPoints[3].x)^2 + (TL.y-contourPoints[3].y)^2 );

    if(contourPoints[2].y > contourPoints[3].y)
    {
        BR = contourPoints[2];
        TR = contourPoints[3];

    }
    else
    {
        BR = contourPoints[3];
        TR = contourPoints[2];

    }

    //push back points in a CCW order, starting with the TL point
    sortedPoints.push_back(TL);
    sortedPoints.push_back(BL);
    sortedPoints.push_back(BR);
    sortedPoints.push_back(TR);

    return sortedPoints;

}

/*calculate euclidean distance for a point vector*/
double calcAspectRatio(std::vector< Point > point_array)
{
    Point p1 = point_array[0];
    Point p2 = point_array[1];
    Point p3 = point_array[2];
    Point p4 = point_array[3];

    //calculate euclidean distance b/t points that make up the length and width of the point vector quadrilateral
    double l1,l2,w1,w2;
    l1 = sqrt(pow((p1.x-p4.x),2)+pow((p1.y-p4.y),2));
    l2 = sqrt(pow((p2.x-p3.x),2)+pow((p2.y-p3.y),2));
    w1 = sqrt(pow((p1.x-p2.x),2)+pow((p1.y-p2.y),2));
    w2 = sqrt(pow((p4.x-p3.x),2)+pow((p4.y-p3.y),2));

    double temp_aspect = (l1+l2)/(w1+w2);

    return temp_aspect;

}

/*find rotated bounding boxes associated with each contour*/
std::vector<RotatedRect> contourstoRotBoxes(std::vector< std::vector<Point> >& rectContours2)
{
    //extract rectangles from contours
    std::vector<RotatedRect> minRect;

    for(int i=0; i < rectContours2.size(); i++)
    {
        RotatedRect temp = minAreaRect(rectContours2[i]);

        minRect.push_back(temp);
    }

    return minRect;
}

/*deskew/crop/post-process individual labels from raw image*/
std::vector <Mat> extractLabels(Mat& img3, std::vector<RotatedRect> rotRect1, std::vector< std::vector<Point> >& rectContours3 )
{
    std::vector <Mat> img_noSkew;
    std::vector<Point> tempContours;

    for(int i=0; i < rotRect1.size(); i++)
    {
        /*************************************Deskewing operations *******************************************/
        Point2f sourcePoints[4];
        Point2f targetPoints[4];

        Size contourSize = rotRect1[i].size;

        if(contourSize.height > contourSize.width)
            std::swap(contourSize.width,contourSize.height);

        tempContours = rectContours3[i];

        //create source points from exact contours
        sourcePoints[0]=Point2f(tempContours[0].x, tempContours[0].y);
        sourcePoints[1]=Point2f(tempContours[1].x, tempContours[1].y);
        sourcePoints[2]=Point2f(tempContours[2].x, tempContours[2].y);
        sourcePoints[3]=Point2f(tempContours[3].x, tempContours[3].y);

        //create target points from rotated bounding box
        targetPoints[0]=sourcePoints[0]; //TL point
        targetPoints[1]=Point2f( targetPoints[0].x, (targetPoints[0].y+contourSize.height) ); //BL point
        targetPoints[2]=Point2f( (targetPoints[0].x+contourSize.width), (targetPoints[0].y+contourSize.height) ); //BR point
        targetPoints[3]=Point2f( (targetPoints[0].x+contourSize.width), targetPoints[0].y ); //TR poit

        //perspective transform
        Mat img_pTransform=Mat::zeros(img3.size(),img3.type());
        Mat rot1=getPerspectiveTransform(sourcePoints,targetPoints);
        warpPerspective(img3,img_pTransform,rot1,img_pTransform.size());
        //imshow("Deskewed image", img_pTransform);

        //crop image
        Mat img_crop;
        Size cropSize;
        Rect upRect(targetPoints[0].x, targetPoints[0].y, contourSize.width, contourSize.height);

        if(upRect.height > edge_offset)
            cropSize=Size(upRect.width-edge_offset,upRect.height-edge_offset);
        else
            cropSize=Size(upRect.width-edge_offset,upRect.height);

        Point2f center = Point( (upRect.tl().x+upRect.width/2),(upRect.tl().y+upRect.height/2));
        getRectSubPix(img_pTransform,cropSize,center,img_crop);
        //imshow("Cropped Image",img_crop);
        /************************************************************************************************************/

        /*************************************Post-processing operations *******************************************/
        //process resized image
        resize(img_crop,img_crop,Size(cropWidth,cropHeight));
        Mat img_cropBlur,img_cropThresh,img_cropMorph,img_cropFinal;
        cvtColor(img_crop,img_crop,CV_BGR2GRAY);
        bilateralFilter(img_crop,img_cropBlur,filter_size, filter_size*2, filter_size/2);
        adaptiveThreshold(img_cropBlur,img_cropThresh,255,ADAPTIVE_THRESH_GAUSSIAN_C,CV_THRESH_BINARY_INV,bt_window_size,0);

        Mat elem1 = getStructuringElement(MORPH_ELLIPSE, Size(element_size,element_size));

        morphologyEx(img_cropThresh,img_cropMorph,CV_MOP_CLOSE, elem1);

        img_noSkew.push_back(img_cropMorph);
        //imshow("Deskewed & Cropped Image", img_cropMorph);
        /************************************************************************************************************/
    }

    return img_noSkew;

}

/*convert orientation expression in quaternion into degrees*/
double convertQuat2Degree(geometry_msgs::Pose p)
{
    tf::Quaternion q(p.orientation.x,
                     p.orientation.y,
                     p.orientation.z,
                     p.orientation.w);

    tf::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll,pitch,yaw);

    double yaw_degrees = yaw*180/M_PI;

    return yaw_degrees;
}

/*run Tesseract on a candidate label image and return an OCR extract */
std::string extractOCRText(Mat img)
{
    char *out_text;
    tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
    std::string raw_extract, target_extract;
    std::vector<string> row_text;

    Mat* img_temp = &img;

    // Initialize tesseract-ocr with English, without specifying tessdata path
    if (api->Init(NULL, "eng"))
    {
        std::cout<<"\nCould not initialize tesseract";
        target_extract = "NULL";
    }

    else
    {
        api->TesseractRect(img_temp->data,1,img_temp->step1(),0,0,img_temp->size().width, img_temp->size().height);

        // Get OCR result
        out_text = api->GetUTF8Text();

        if(out_text!=NULL)
        {
            raw_extract = (string)out_text;

            //extract first row from text
            boost::algorithm::split(row_text, raw_extract, boost::is_any_of("\n"));
            target_extract = row_text[0];
        }

        // Destroy used object and release memory
        api->End();
        delete [] out_text;
    }

    return target_extract;
}

/*for words with spelling errors, estimate the closest match to the a priori word list*/
std::string stringComparison(std::string candidate, std::vector<std::string> words)
{
    int best_edit_distance=100, temp_distance, best_index;
    std::string prev_best_word = " ";

    //seperate candidate label into individual words
    std::vector<std::string>temp_words;
    boost::algorithm::split(temp_words, candidate, boost::is_any_of(" "));

    //evaluate the edit distance of each individual word
    for(int n =0; n < temp_words.size(); n++)
    {
        for (int i=0; i < words.size(); i++)
        {
            temp_distance = edit_distance(temp_words[n], words[i]);

            //update the smallest edit_distance value
            if(i==0 || temp_distance < best_edit_distance && words[best_index] != prev_best_word)
            {
                best_edit_distance = temp_distance;
                best_index = i;
            }

        }

        //only correct a word if the amount of deletions, insertions and substitutions is < 2
        if(best_edit_distance <= max_edit_distance)
        {
            boost::algorithm::to_upper(words[best_index]);
            temp_words[n]=words[best_index];
        }

    }

    //combine individual words back into a single label
    std::string combinedWords = boost::algorithm::join(temp_words," ");

    return combinedWords;

}

/*calculates the edit_distance between two words*/
int edit_distance(std::string sourceWord, std::string targetWord)
{
    boost::algorithm::to_upper(targetWord);
	std::size_t len1 = sourceWord.size(), len2 = targetWord.size();
	std::vector< std::vector<int> > d(len1 + 1, std::vector<int>(len2 + 1) );

	//initialize matrix
	d[0][0] = 0;

	for(int i = 1; i <= len1; i++)
        d[i][0] = i;

	for(int i = 1; i <= len2; i++)
        d[0][i] = i;

    //calculate cell entries
	for(int i = 1; i <= len1; i++)
    {
        //assign minimum cost to each cell entry
        for(int j = 1; j <= len2; ++j)
            d[i][j] = std::min( d[i - 1][j] + 1, std::min(d[i][j - 1] + 1, d[i - 1][j - 1] + (sourceWord[i - 1] == targetWord[j - 1] ? 0 : 1) ) );
    }

    //edit distance is (len1,len2)
	return (d[len1][len2]);
}

/*assign an a priori category label to corrected text, if possible*/
string classifySignLabel(std::string ucl_label,std::vector<SignLabel> target_labels)
{
    std::string classification;
    std::vector<string>ucl_labelWords;
    boost::algorithm::split(ucl_labelWords, ucl_label, boost::is_any_of(" "));
    std::vector <float> labelProb(target_labels.size(),0.0);

    //compare each unclassified text word against the master label list
    for(int q=0; q<ucl_labelWords.size(); q++)
    {
        std::string ucl_temp =  ucl_labelWords[q];

        for(int m=0; m<target_labels.size(); m++)
        {
            //seperate master list labels into individual words
            if(target_labels[m].numWords() >= 1)
            {

                for(int x=0; x < target_labels[m].numWords(); x++)
                {
                    //std::cout<<"\n"<<targetLabels[m].getWord(x);
                    if(boost::iequals(ucl_temp, target_labels[m].getWord(x)) )
                    //if(image_words[q]==word[x])
                    {
                        labelProb[m] += 1/(float)target_labels[m].numWords();
                        //cout<<"\n"<<word[x]<<" matches "<<image_words[q];
                    }

                }

            }

        }
    }

    //check label probabilities; create a subset of labels with probabilities >= 0.5
    std::vector<float> high_prob_labels;
    std::vector<int> high_prop_idxs;

    //std::cout<<"\n\tPossible Labels: ";
    for(int d=0; d<labelProb.size(); d++)
    {

        if(labelProb[d] >= prob_threshold )
        {
            high_prob_labels.push_back(labelProb[d]);
            high_prop_idxs.push_back(d);
            //std::cout<<target_labels[d].getLabel()<<" ";
            //std::cout<<labelProb[d]<<",";
        }


    }

    //assign label with highest probability
    //Case 1: no labels exceed probability threshold; no label found
    if(high_prob_labels.size()==0)
        classification = "N/A";

    //Case 2: 1 label exceeds probability threshold; 1 label found
    else if(high_prob_labels.size()== 1)
    {

        int index = high_prop_idxs[0];
        classification = target_labels[index].getLabel();
    }

    //Case 3: Two or more labels exceed probability threshold
    else
    {
        //std::cout<<"\n\tCase 2";

        int temp_index = 0;

        //search for index corresponding to subset's highest probability
        for(int b =1; b < high_prob_labels.size(); b++)
        {
            //std::cout<<"\n\tEntered Case 2 loop";
            if(high_prob_labels[b] > high_prob_labels[temp_index] )
            {
                temp_index = b;
                //std::cout<<"\n\t"<<highProbLabels[b]<<" is greater than "<< highProbLabels[tempIndex];
            }


            //for equal probabilities, favour the label that has multiple words
            else if(high_prob_labels[temp_index] == high_prob_labels[b] )
            {
                //std::cout<<"\n\tEqual probabilities";
                if(target_labels[high_prop_idxs[temp_index]].numWords() < target_labels[high_prop_idxs[b]].numWords() )
                {
                    //std::cout<<"\n\t"<<targetLabels[probCounter[b]].getLabel() << " has more words than " <<targetLabels[probCounter[tempIndex]].getLabel();
                    temp_index = b;
                }

            }
            //else
               // std::cout<<"\n\tCurrent label, "<<targetLabels[maxIndex].getLabel()<<" is the best";

        }

        int max_index = high_prop_idxs[temp_index];
        classification = target_labels[max_index].getLabel();
    }

    high_prob_labels.clear();
    return classification;

}

/*display captured contours and vertices on original image*/
void drawFeatures_no_labels(Mat& img4, vector<RotatedRect>& rotRect2, vector< vector<Point> >& rectContours4)
{
    vector<Vec4i> hierarchy;

    for(int c=0; c<rectContours4.size();c++)
    {
        Point sP[4];
        vector<Point> temp = rectContours4[c];

        sP[0]=temp[0];
        sP[1]=temp[1];
        sP[2]=temp[2];
        sP[3]=temp[3];

        drawContours(img4,rectContours4,c,Scalar(0,255,0),2,8, hierarchy,0, Point() );

        for(int d=0; d<4; d++)
        {
            Scalar col;

            if(d==0)
                col=Scalar(255,0,0);
            else
                col=Scalar(0,0,255);

            circle(img4,sP[d],5,col,2,8);

            //point label
            std::stringstream index;
            index << d+1;
            std::string label = "P"+index.str();
            putText(img4, label, sP[d], FONT_HERSHEY_PLAIN,1.0,Scalar(0,0,255),2.0);
        }
    }

    namedWindow("Image with Bounding Box", WINDOW_AUTOSIZE);
    imshow("Image with Bounding Box", img4);

}

/*display captured contours and classiied labels on original image*/
void drawFeatures(Mat img, std::vector<CaptLabel> label_array)
{
    std::vector<Vec4i> hierarchy;

    for(int i=0; i<label_array.size();i++)
    {
        //draw contours
        std::vector<std::vector<Point> > temp_point_array;
        temp_point_array.push_back(label_array[i].contour_points);
        drawContours(img,temp_point_array,0,Scalar(0,255,0),2,8, hierarchy,0, Point() );

        //draw contour vertices
        Point sP[4];

        sP[0]=label_array[i].contour_points[0];
        sP[1]=label_array[i].contour_points[1];
        sP[2]=label_array[i].contour_points[2];
        sP[3]=label_array[i].contour_points[3];

        circle(img,sP[0],5,Scalar(0,0,255),2,8);
        circle(img,sP[1],5,Scalar(0,0,255),2,8);
        circle(img,sP[2],5,Scalar(0,0,255),2,8);
        circle(img,sP[3],5,Scalar(0,0,255),2,8);

        //add classification label to contour
        putText(img, label_array[i].label , sP[0], FONT_HERSHEY_PLAIN,1.5,Scalar(0,0,255),2.0);

        temp_point_array.clear();
    }

    imshow("Classified Image", img);

}

/**********************************************************************************************************************************/
