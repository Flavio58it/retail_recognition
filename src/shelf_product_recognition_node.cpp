#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/String.h>
#include <tf/transform_listener.h>
#include <tf/tf.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>

using namespace cv;

/**********************************************************Global Variables**********************************************************/

/**************************Subscribed Topic & Map Frame Parameters*******************/
//subscribed topic names
std::string usb_cam_topic_name;
int img_counter = 0;

//template image directory
std::string templateDir;
std::string templateList;
std::string product1, product2, product3;

// webcam callback parameters
cv_bridge::CvImagePtr cv_ptr;
std::string enable_flag=" ";

/***********************************************************************************/


/**************************Pre-processing Parameters********************************/

//image Crop Parameters
int input_height;
int input_width;

//bilateral Filter
int blur_colRange;
int blur_spatialRange;

//dilation
int element_size;

//shelf mask horizontal & vertical lines
int h_scale;
int v_scale;

/***********************************************************************************/


/**************************Object Detection parameters******************************/
//region proposals
double ar_upper;
double ar_lower;
int minContourArea;
int min_stddev;
int overlap_minThresh;

//sliding window
int sx;
double scale_array[]={1.0};//,0.25};
int n_scales;
/***********************************************************************************/


/**************************Feature Matching Parameters******************************/
double maxRatio; //ratio test
int match_threshold; //miniumum number of matches needed for RANSAC
float nms_min_overlap; //min overlap ratio for NMS
/***********************************************************************************/


/***********************************************************************************************************************************/


/*******************************struct RegionProp***********************************************************************************/
/*RegionProp is defined by vector of contour array points, contour area and bounding box members*/
struct RegionProp
{
    std::vector<Point> contour;
    double region_area;
    Rect bounding_box;
};
/************************************************************************************************************************************/


/*********************************struct ShelfObj************************************************************************************/
/*ShelfObj is defined by a bounding box, SIFT descriptor, SIFT keypoints, Mat img and classification members */
struct ShelfObj
{
    //image id
    int img_id;

    //map pose
    double xPos, yPos, theta;
    double timeStamp;

    // bounding box
    Rect bounding_box;

    //features
    Mat img,descriptor_sift,descriptor_surf,descriptor_orb, descriptor_freak;
    std::vector<KeyPoint> keypoints_sift;

    bool overlap;

    //classification
    string soft_label,hard_label;
    int numMatches;


};
/************************************************************************************************************************************/


/*********************************struct TemplateObj*********************************************************************************/
/*TemplateObj is defined by a classification label, SIFT descriptor and SIFT keypoints members*/
struct TemplateObj
{
    string soft_label,hard_label;
    Mat img,descriptor_sift,descriptor_surf,descriptor_orb,descriptor_freak;
    std::vector<KeyPoint> keypoints_sift;
};
/************************************************************************************************************************************/



/**********************************************************Function Declarations****************************************************/
std::vector<TemplateObj> loadTemplateImages();
void templateListSummary(std::vector<TemplateObj> templateList);

void usb_cam_callback(const sensor_msgs::ImageConstPtr& msg_rawImg);

Mat preprocessingImage(Mat img);
Mat createShelfMask(Mat img);
std::vector<RegionProp> findBoundingBoxes(Mat img_binary, Mat img_raw);
std::vector<int> findEnclosedBoundingBoxes(std::vector<RegionProp> proposals);

std::vector<Rect> slidingWindow(Mat img, std::vector<RegionProp> proposals);
ShelfObj createShelfObject(Mat img, Rect r);

int findKeypointMatch(TemplateObj templateImg, ShelfObj candidate);
double findTemplateMatch(TemplateObj templateImg, ShelfObj candidate);
std::vector<int> nonMaximumSupression(std::vector<ShelfObj> obj_list);
void save_results(std::vector<ShelfObj> obj_labels);
/************************************************************************************************************************************/


/**********************************************************Main function*************************************************************/
int main(int argc, char** argv)
{
    //node initialization
    ros::init(argc, argv, "shelf_product_recognition_node");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    //get parameters from launch file
    nh.getParam("/shelf_product_recognition_node/usb_cam_topic_name", usb_cam_topic_name);

    nh.getParam("/shelf_product_recognition_node/templateDir",templateDir);
    nh.getParam("/shelf_product_recognition_node/templateList",templateList);
    nh.getParam("/shelf_product_recognition_node/product1",product1);
    nh.getParam("/shelf_product_recognition_node/product2",product2);
    nh.getParam("/shelf_product_recognition_node/product3",product3);

    nh.getParam("/shelf_product_recognition_node/input_width",input_width);
    nh.getParam("/shelf_product_recognition_node/input_height",input_height);

    nh.getParam("/shelf_product_recognition_node/input_width",input_width);
    nh.getParam("/shelf_product_recognition_node/input_height",input_height);

    nh.getParam("/shelf_product_recognition_node/blur_colRange",blur_colRange);
    nh.getParam("/shelf_product_recognition_node/blur_spatialRange",blur_spatialRange);
    nh.getParam("/shelf_product_recognition_node/element_size",element_size);

    nh.getParam("/shelf_product_recognition_node/h_scale",h_scale);
    nh.getParam("/shelf_product_recognition_node/v_scale",v_scale);

    nh.getParam("/shelf_product_recognition_node/ar_upper",ar_upper);
    nh.getParam("/shelf_product_recognition_node/ar_lower",ar_lower);
    nh.getParam("/shelf_product_recognition_node/minContourArea",minContourArea);
    nh.getParam("/shelf_product_recognition_node/min_stddev",min_stddev);
    nh.getParam("/shelf_product_recognition_node/overlap_minThresh",overlap_minThresh);

    nh.getParam("/shelf_product_recognition_node/n_scales", n_scales);

    nh.getParam("/shelf_product_recognition_node/maxRatio", maxRatio);
    nh.getParam("/shelf_product_recognition_node/match_threshold", match_threshold);
    nh.getParam("/shelf_product_recognition_node/nms_min_overlap", nms_min_overlap);


    //initialize subscribers & publishers
    image_transport::Subscriber sub_usb_cam;
    sub_usb_cam = it.subscribe(usb_cam_topic_name,1,usb_cam_callback);

    ros::Publisher pub_shelf_item_labels = nh.advertise<std_msgs::String>("/shelf_product_labels",100);

    //load template images
    std::vector<TemplateObj> template_array = loadTemplateImages();
    std::cout<<"\nSuccessfully loaded "<<template_array.size()<<" images";
    templateListSummary(template_array);

    Mat img_src, img_binary, img_bbox, frame;
    std::vector<Rect> objRects;
    std::vector<ShelfObj> obj_array, obj_scored, obj_nms;
    geometry_msgs::PoseStamped pose_robot;
    tf::TransformListener listener;

    while(ros::ok())
    {
        if(cv_ptr)
        {

            img_src= cv_ptr->image.clone();
            resize(img_src,img_src,Size(input_width,input_height));

            img_binary=preprocessingImage(img_src);

            /***Region Proposals******************************************************************************************************************************************/
            //find region proposals
            std::vector<RegionProp> props = findBoundingBoxes(img_binary,img_src);

            img_bbox=img_src.clone();
            for(int k=0; k < props.size(); k++)
                rectangle(img_bbox,props[k].bounding_box.tl(),props[k].bounding_box.br(),Scalar(0,0,255),2,8);

            imshow("Region Proposals", img_bbox);//*/
            /***************************************************************************************************************************************************************/

            /***Sliding Window**********************************************************************************************************************************************/
            //generate all individual object candidate bounding boxes
            objRects = slidingWindow(img_src,props);

            //create object candidate features
            for(int n=0; n<objRects.size(); n++)
            {
                ShelfObj temp = createShelfObject(img_src,objRects[n]);
                obj_array.push_back(temp);
            }

            std::cout<<"\n**********Image "<<img_counter<<"**********";
            std::cout<<"\n# of Object Proposals: "<<objRects.size();

            //*/
            /***************************************************************************************************************************************************************/

            /***Multi Template Classification********************************************************************************************************************************/
            std::vector<int> match_array;
            Mat img_matches = img_src.clone();

            //compare each object candidate to all templates
            for(int i =0; i < obj_array.size(); i++)
            {
                //find keypoint matches to each template
                //std::cout<<"\n\t\tObject"<<i+1;
                for(int j=0; j < template_array.size(); j++)
                {

                    double num_matches = findKeypointMatch(template_array[j],obj_array[i]);
                    match_array.push_back(num_matches);
                }

                //record index in match_array of object-template pair with the highest matches
                int bestMatchIdx = std::distance(match_array.begin(), std::max_element(match_array.begin(), match_array.end()) );
                int bestNumMatches = match_array[bestMatchIdx];

                //only record best object matches that have at least 1 keypoint match
                if(bestNumMatches > 0)
                {
;
                    obj_array[i].img_id = img_counter;
                    obj_array[i].timeStamp = pose_robot.header.stamp.toSec();
                    obj_array[i].xPos = pose_robot.pose.position.x;
                    obj_array[i].yPos = pose_robot.pose.position.y;
                    obj_array[i].numMatches = bestNumMatches;
                    obj_array[i].hard_label = template_array[bestMatchIdx].hard_label;
                    obj_array[i].soft_label = template_array[bestMatchIdx].soft_label;
                    obj_scored.push_back(obj_array[i]);//*/

                    //check if classified products match products on the grocery list
                    if(obj_array[i].hard_label==product1 || obj_array[i].hard_label==product2 || obj_array[i].hard_label==product3)
                    {
                        rectangle(img_matches,obj_array[i].bounding_box.tl(),obj_array[i].bounding_box.br(),Scalar(0,255,0),2,8);
                        putText(img_matches,obj_array[i].hard_label,obj_array[i].bounding_box.tl(),FONT_HERSHEY_COMPLEX_SMALL,0.75,Scalar(255,255,0),1);
                        std::stringstream counter;
                        counter << i;
                        int height = obj_array[i].bounding_box.height;
                        Point p(obj_array[i].bounding_box.tl().x,obj_array[i].bounding_box.tl().y+height);
                        putText(img_matches,counter.str(),p,FONT_HERSHEY_COMPLEX_SMALL,0.5,Scalar(255,255,0),1);
                    }

                }

                match_array.clear();
            }

            std::cout<<"\n# of Initial Matches: "<<obj_scored.size();
            for(int i=0; i < obj_array.size(); i++)
            {
                std::cout<<"\n\tObject "<<i+1<<":"<<obj_array[i].hard_label;

                if(obj_array[i].hard_label!="NONE")
                {
                    std_msgs::String msg_label;
                    msg_label.data = obj_array[i].hard_label;
                    pub_shelf_item_labels.publish(msg_label);

                }

            }

            imshow("Initial Matches",img_matches);//*/
            /*****************************************************************************************************************************************************************/

            /***Non-Maximum Suppression***************************************************************************************************************************************/
            /*std::vector<int>non_repeat_idxs = nonMaximumSupression(obj_scored);
            std::cout<<"\n# of Final Matches: "<<non_repeat_idxs.size();
            //std::cout<<"\nFiltered Objects:";
            Mat img_nms = img_src.clone();
            for(int i=0; i < non_repeat_idxs.size(); i++)
            {
                //add bounding boxes & classification labels to raw image
                std::cout<<"\n\tObject "<<i+1<<":"<<obj_scored[non_repeat_idxs[i]].hard_label;
                rectangle(img_nms,obj_scored[non_repeat_idxs[i]].bounding_box.tl(),obj_scored[non_repeat_idxs[i]].bounding_box.br(),Scalar(0,255,0),2,8);
                putText(img_nms,obj_scored[non_repeat_idxs[i]].hard_label,obj_scored[non_repeat_idxs[i]].bounding_box.tl(),FONT_HERSHEY_COMPLEX_SMALL,0.75,Scalar(255,255,0),1);

                //publish classified shelf item labels
                //std_msgs::String msg_label;
                //msg_label.data = obj_scored[non_repeat_idxs[i]].hard_label;
                //pub_shelf_item_labels.publish(msg_label);
            }

            imshow("NMS",img_nms);//*/
            /*****************************************************************************************************************************************************************/


            props.clear();
            objRects.clear();
            obj_array.clear();
            obj_scored.clear();
            //non_repeat_idxs.clear();

            std::cout<<"\n***************************"; //*/
        }

        ros::spinOnce();
    }

    return 0;
}
/************************************************************************************************************************************/


/**********************************************************Functions*****************************************************************/

/*load product template images from a specified directory*/
std::vector<TemplateObj> loadTemplateImages()
{
    std::ifstream infile;
    infile.open(templateList.c_str());

    string img_location, img_filename, hlabel, slabel;
    TemplateObj temp;
    std::vector<TemplateObj> tempArray;

    if(infile.is_open())
    {
        if(infile.peek() == std::ifstream::traits_type::eof())
        {
            std::cout<<"\nFile is empty";
            infile.close();
        }
        else
        {

            std::cout<<"\nImporting template information";
            while(infile>>slabel>>hlabel>>img_filename)
            {
                temp.soft_label = slabel;
                temp.hard_label = hlabel;

                string img_location = templateDir+img_filename;
                Mat img = imread(img_location);
                //std::cout<<"\n"<<img_location;
                //imshow("Template image",img);
                temp.img = img;

                //SIFT keypoints & descriptors
                SiftFeatureDetector d_sift;
                SiftDescriptorExtractor e_sift;
                d_sift.detect(temp.img,temp.keypoints_sift);
                e_sift.compute(temp.img,temp.keypoints_sift,temp.descriptor_sift);

                tempArray.push_back(temp);
                temp.keypoints_sift.clear();

            }

            infile.close();
        }

    }

    else
    {
        std::cout<<"\nFile is not open";
        infile.close();
    }

    return tempArray;
}

/*print a summary of loaded product template information*/
void templateListSummary(std::vector<TemplateObj> templateList)
{
    std::cout<<"\n******Template List Summary*******/";
    std::cout<<"\n# of Template Images: "<<templateList.size();

    for(int i=0; i < templateList.size(); i++)
    {
        std::cout<<"\nTemplate "<<i+1<<":";
        std::cout<<"\n\tSoft Label: "<<templateList[i].soft_label;
        std::cout<<"\n\tHard Label: "<<templateList[i].hard_label;
        std::cout<<"\n\t# of SIFT Keypoints: "<<templateList[i].keypoints_sift.size();
    }

    std::cout<<"\n**********************************/";
}

/*convert ROS image format to OpenCv format and display converted image*/
void usb_cam_callback(const sensor_msgs::ImageConstPtr& msg_rawImg)
{
    try
    {
        cv_ptr=cv_bridge::toCvCopy(msg_rawImg, sensor_msgs::image_encodings::BGR8);
        img_counter++;
        //imshow("Raw Image",cv_ptr->image);
        //std::cout<<"\nRaw Image Size: "<<cv_ptr->image.size();
        waitKey(30);
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

}

/*create binary of image without shelving wall features*/
Mat preprocessingImage(Mat img)
{
    Mat channels[3];
    Mat img_blur_b, img_blur_g, img_blur_r, img_thresh_b, img_thresh_g, img_thresh_r,img_add1,img_add2;
    split(img,channels);

    normalize(channels[0], channels[0],0,255,NORM_MINMAX);
    normalize(channels[1], channels[1],0,255,NORM_MINMAX);
    normalize(channels[2], channels[2],0,255,NORM_MINMAX);

    bilateralFilter(channels[0],img_blur_b,11,blur_colRange,blur_spatialRange);
    bilateralFilter(channels[1],img_blur_g,11,blur_colRange,blur_spatialRange);
    bilateralFilter(channels[2],img_blur_r,11,blur_colRange,blur_spatialRange);

    Mat elem1 = getStructuringElement(MORPH_ELLIPSE, Size(1+2*element_size,1+2*element_size));

    threshold(img_blur_b,img_thresh_b,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
    Mat bmask = createShelfMask(img_thresh_b);
    subtract(img_thresh_b,bmask,img_thresh_b);

    threshold(img_blur_g,img_thresh_g,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
    Mat gmask = createShelfMask(img_thresh_g);
    subtract(img_thresh_g,gmask,img_thresh_g);

    threshold(img_blur_r,img_thresh_r,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);

    Mat rmask = createShelfMask(img_thresh_r);
    subtract(img_thresh_r,rmask,img_thresh_r);

    //OR
    bitwise_or(img_thresh_b,img_thresh_g,img_add1);
    bitwise_or(img_thresh_r,img_add1,img_add2);

    /*imshow("BT-B", img_thresh_b);
    imshow("BT-G", img_thresh_g);
    imshow("BT-R", img_thresh_r);

    imshow("B Mask",bmask);
    imshow("G Mask",gmask);
    imshow("R Mask",rmask);*/
    imshow("Added Image",img_add2);

    return img_add2;

}

/*create a mask that identifies horizontal & vertical walls of shelving unit*/
Mat createShelfMask(Mat img)
{
    Mat img_horizontal, img_vertical, mask;

    //horizontal lines
    img_horizontal=img.clone();
    int horizontalSize = img_horizontal.cols/h_scale;
    Mat elem_h = getStructuringElement(MORPH_RECT,Size(horizontalSize,1));
    Mat elem2 = getStructuringElement(MORPH_ELLIPSE, Size(1+2*element_size,1+2*element_size));
    erode(img_horizontal,img_horizontal,elem_h);
    dilate(img_horizontal,img_horizontal,elem_h);
    dilate(img_horizontal,img_horizontal,elem2);

    //vertical lines
    img_vertical=img.clone();
    int verticalSize = img_vertical.rows/v_scale;
    Mat elem_v = getStructuringElement(MORPH_RECT,Size(1,verticalSize));
    erode(img_vertical,img_vertical,elem_v);
    dilate(img_vertical,img_vertical,elem_v);
    dilate(img_vertical,img_vertical,elem2);

    //create mask
    mask = img_horizontal;//+img_vertical;
    //imshow("Shelf Mask",mask);

    return mask;

}

//find bounding boxes in hue channel image
std::vector<RegionProp> findBoundingBoxes(Mat img_binary, Mat img_raw)
{

    std::vector< vector<Point> > contours;
    std::vector<Vec4i> hierarchy;
    findContours(img_binary,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
    std::vector<RegionProp> proposalArray, proposalsFinal;

    for(int i=0; i < contours.size(); i=hierarchy[i][0])
    {
        Rect temp = boundingRect(contours[i]);
        double aspect_ratio = ((double)temp.height/(double)temp.width);
        bool backgroundDetected=true;

        if(aspect_ratio>=ar_upper || aspect_ratio<ar_lower)
            backgroundDetected=false;


        if( contourArea(contours[i],false) > minContourArea && backgroundDetected )
        {
            //std::cout<<"\nAspect Ratio: "<<aspect_ratio;

            RegionProp candidate;
            candidate.bounding_box=temp;
            candidate.contour = contours[i];
            candidate.region_area=contourArea(contours[i],false);

            //find standard deviation of region
            Mat img_region = img_raw.clone();
            img_region=img_region(candidate.bounding_box);
            cvtColor(img_region,img_region,CV_BGR2GRAY);
            //cvtColor(img_region,img_region,CV_BGR2HSV);

            //Mat r_channels[3];
            //split(img_region,r_channels);
            Scalar mean, stddev;
            meanStdDev(img_region,mean,stddev);
            //meanStdDev(r_channels[0],mean,stddev);
            //std::cout<<"\nRegion "<<i<<" stddev: "<<stddev[0];

            if(stddev[0] > min_stddev)
                proposalArray.push_back(candidate);
        }
     }

     std::vector<int> goodIndices = findEnclosedBoundingBoxes(proposalArray);

     for(int k=0; k < goodIndices.size(); k++ )
         proposalsFinal.push_back(proposalArray[goodIndices[k]]);

    return proposalsFinal;

}

/*find interior bounding boxes and remove them from the final region proposal list*/
std::vector<int> findEnclosedBoundingBoxes(std::vector<RegionProp> proposals)
{
    std::vector<int> indices;

    for(int i=0; i < proposals.size(); i++)
    {
        Rect ri = proposals[i].bounding_box;
        int width_ri = ri.width;

        std::vector<RegionProp> resultantArray = proposals;
        resultantArray.erase(resultantArray.begin()+i);
        bool contained = false;
        bool overlap = false;

        for(int j=0; j< resultantArray.size(); j++)
        {
            Rect rj = resultantArray[j].bounding_box;
            int width_rj = rj.width;

            if( ri.tl().x >= rj.tl().x &&
                ri.tl().x+width_ri <= rj.tl().x+width_rj &&
                ri.tl().y >= rj.tl().y &&

                ri.br().x <= rj.br().x &&
                ri.br().x-width_ri >= rj.br().x-width_rj &&
                ri.br().y <= rj.br().y )
            {
                contained = true;
                break;
            }

            Rect r_intersect = ri&rj;
            Rect r_union = ri|rj;
            double intersect_ratio = ((double)r_intersect.area())/((double)ri.area());

            if( intersect_ratio > overlap_minThresh && intersect_ratio < 1 && (rj.area() > ri.area()) )
            {
                //std::cout<<"\nIntersection Area: "<<intersect_ratio;
                overlap = true;
                break;
            }

        }

        if(!contained && !overlap)
            indices.push_back(i);

        resultantArray.clear();
    }

    return indices;
}

/*divide each region proposal into a set of object candidates using a sliding window*/
std::vector<Rect> slidingWindow(Mat img, std::vector<RegionProp> proposals)
{
    std::vector<Rect> rect_array;
    Mat img_sw=img.clone();
    Scalar col;

    for(int i=0; i < proposals.size();i++)
    {
        //double ratio = (double) proposals[i].bounding_box.width / (double)proposals[i].bounding_box.height;
        //do not apply sliding windows where the region proposal height > regiona proposal width
        if(proposals[i].bounding_box.width < proposals[i].bounding_box.height)
        {
            rect_array.push_back(proposals[i].bounding_box);
            rectangle(img_sw,proposals[i].bounding_box.tl(),proposals[i].bounding_box.br(),Scalar(0,0,255),2,8);
        }
        else
        {
            for(int k=0; k < n_scales; k++)
            {
                int width_window = proposals[i].bounding_box.width*scale_array[k];
                int height_window= proposals[i].bounding_box.height;
                int x_start = proposals[i].bounding_box.tl().x;
                int x_end = proposals[i].bounding_box.br().x;

                sx = width_window;
                //sliding window
                for(int j=x_start; j+width_window <= x_end;j+=sx)
                {
                    Rect window(j,proposals[i].bounding_box.tl().y,width_window,height_window);
                    rect_array.push_back(window);

                    if(k==0)
                        col = Scalar(0,255,0);
                    else
                        col = Scalar(255,0,0);

                    rectangle(img_sw,window.tl(),window.br(),col,2,8);
                }
            }
            //rect_array.clear();

        }

    }

    //for(int j=0; j < rect_array.size(); j++)
        //rectangle(img_sw,rect_array[j].tl(),rect_array[j].br(),Scalar(0,255,0),2,8);

    //imshow("Sliding Window",img_sw);

    return rect_array;

}

/*create shelf objects using sliding window object proposals*/
ShelfObj createShelfObject(Mat img, Rect r)
{
    ShelfObj o;

    o.bounding_box=r;
    o.img = img(r).clone();

    //Sift keypoints & descriptors
    SiftFeatureDetector d_sift;
    SiftDescriptorExtractor e_sift;
    d_sift.detect(o.img,o.keypoints_sift);
    e_sift.compute(o.img,o.keypoints_sift,o.descriptor_sift);

    o.soft_label="NONE";
    o.hard_label="NONE";

    o.numMatches=0;

    o.overlap=false;

    return o;

}

/*return number of keypoint matches between an object proposal and a template image*/
int findKeypointMatch(TemplateObj templateImg, ShelfObj candidate)
{
    //std::cout<<"\n# of Candidate keypoints"<<candidate.keypoints_orb.size();
    //std::cout<<"\n# of Template keypoints"<<templateImg.keypoints_orb.size();

    BFMatcher matcher_bf_knn(4,false); //4=NORM_L2, 6=NORM_HAMMING
    std::vector< vector<DMatch> > matches_bf_knn;
    std::vector<DMatch> goodMatches_knn;

    matcher_bf_knn.knnMatch(candidate.descriptor_sift,templateImg.descriptor_sift,matches_bf_knn,2);

    //apply ratio test
    for(int i=0; i < matches_bf_knn.size(); i++)
    {
        double ratio_temp = matches_bf_knn[i][0].distance / matches_bf_knn[i][1].distance;
        if( ratio_temp < maxRatio)
        {
            goodMatches_knn.push_back(matches_bf_knn[i][0]);

        }
    }

    //apply RANSAC
    if(goodMatches_knn.size() > match_threshold)
    {
        //find keypoint goodMatches_knn pairs
        std::vector<Point2f> matchScenePts, matchTemplatePts;
        for(int i=0; i < goodMatches_knn.size();i++)
        {
            int objectIndex = goodMatches_knn[i].queryIdx;
            int templateIndex = goodMatches_knn[i].trainIdx;

            matchScenePts.push_back(candidate.keypoints_sift[objectIndex].pt);
            matchTemplatePts.push_back(templateImg.keypoints_sift[templateIndex].pt);
        }

        //calculate homography matrix b/t scene points and template points
        Mat mask;
        Mat H = findHomography(matchScenePts,matchTemplatePts,CV_RANSAC,match_threshold,mask);

        //find indices of inlier points
        std::vector<DMatch> goodMatches_bf_ransac;
        for(int i=0; i < mask.size().height; i++)
        {
            //std::cout<<"\n"<<(unsigned int)mask.at<uchar>(i);
            if((unsigned int)mask.at<uchar>(i))
                goodMatches_bf_ransac.push_back(goodMatches_knn[i]);
        }

        return goodMatches_bf_ransac.size();

    }
    else//*/
        return 0;

}

/*remove overlapping object proposals of the same category and return a final list of proposal indices */
std::vector<int> nonMaximumSupression(std::vector<ShelfObj> obj_list)
{
    std::vector<int> idx_list, r_indices, good_indices;

    //add all object candidate indices to index list
    for(int i=0; i<obj_list.size(); i++)
        idx_list.push_back(i);

    while(idx_list.size() > 0)
    {
        //update source candidate index
        int source_idx = idx_list[0];
        ShelfObj temp = obj_list[source_idx];

        for(int j=1; j < idx_list.size(); j++)
        {
            int search_idx = idx_list[j];

            //determine IOU b/t object candidates with the same hard label
            if(temp.hard_label==obj_list[search_idx].hard_label )//&& !obj_list[search_idx].overlap)
            {
                Rect r_intersect = temp.bounding_box & obj_list[search_idx].bounding_box;
                Rect r_union = temp.bounding_box | obj_list[search_idx].bounding_box;

                double iou = ((double)r_intersect.area())/((double)r_union.area());

                //for small overlaps, assume objects are located at different locations in image
                if(iou < nms_min_overlap)
                    r_indices.push_back(search_idx);

                //for large overlaps, assume object candidate bounding boxes are overlapping the same object
                if(iou > nms_min_overlap)
                {
                    //query candidate has more matches than source candidate, assign overlap to source candidate
                    if(temp.numMatches < obj_list[search_idx].numMatches)
                    {
                        r_indices.push_back(search_idx);
                        temp.overlap = true;
                    }
                }

            }
            //source and query candidates do not have the same label
            else
                r_indices.push_back(search_idx);
        }

        //add source candidate to good_indices if no overlap is found
        if(!temp.overlap)
            good_indices.push_back(source_idx);

        idx_list.clear();
        idx_list = r_indices;
        r_indices.clear();

    }

    return good_indices;
}


/************************************************************************************************************************************/
