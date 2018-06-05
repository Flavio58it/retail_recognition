cmake_minimum_required(VERSION 2.8.3)
project(retail_recognition)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  message_generation
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
)


################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages( DEPENDENCIES geometry_msgs std_msgs sensor_msgs)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
catkin_package(
CATKIN_DEPENDS cv_bridge geometry_msgs image_transport message_runtime roscpp rospy sensor_msgs std_msgs tf
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

#aisle_sign_recognition_node.cpp
add_executable(sign_recognition_node src/sign_recognition_node.cpp)
target_link_libraries(sign_recognition_node ${catkin_LIBRARIES} lept tesseract hunspell boost_system)

#product_recognition_node.cpp
add_executable(shelf_product_recognition_node src/shelf_product_recognition_node.cpp)
target_link_libraries(shelf_product_recognition_node ${catkin_LIBRARIES} opencv_nonfree)




