## Launch File Summary

### bringup_usb_cam.launch
- launches video stream for v4l2 driver-based webcam
- user must specify appropriate video device name (i.e. "/dev/video1") 


### bringup_sign_recognition.launch
- launches bringup_usb_cam.launch + sign_recognition node


### bringup_shelf_product_recognition.launch
- launches bringup_usb_cam.launch + shelf_product_recognition node
