#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <serial/serial.h>
#include <string>

#define PI 3.14159
#define DISPLAY 1
#define TEST 0

using namespace std;
using namespace cv;

/* Function Headers */
void detectFace(Mat frame);
bool compareBigger(Rect face1, Rect face2);
bool compareDistance(Rect face1, Rect face2); 
void writeToMbed(double angled, serial::Serial &mbed);
void test();
void testSerial(); 

/* global variables */
Rect priorFace(0, 0, 0, 0);
CascadeClassifier face_cascade, eyes_cascade;
String display_window = "Display";
String face_window = "Face View";
int frame_width = 0;
/*camera calibration matrices */
Matx33f K_logitech(1517.6023, 0, 0, 0, 1517.6023, 0, 959.5, 539.5, 1);
Matx33f K_facetime(1006.2413, 0, 0, 0, 1006.2413, 0, 639.5, 359.5, 1); //ordered by cols

/*angle look-up tables*/
//const float angleThreshold[15] = {26, 22, 18, 14, 10, 6, 2, -2, -6, -10, -14, -18, -22, -26, -30}; 
const float angleThreshold[15] = {-26, -22, -18, -14, -10, -6, -2, 2, 6, 10, 14, 18, 22, 26, 30}; 

int main() {
  if (TEST) {
    testSerial();
    return 1;
  }

  VideoCapture cap(0); // capture from default camera
  Mat frame;
  Mat displayFrame;
  Point faceCenter(0, 0);  
  double angle = 0;
  double angled = 0;

  if (!(face_cascade.load("src/classifiers/haarcascade_frontalface_alt.xml"))) {
    cout << "error loading face classifier" << endl;
    return -1;
  }
  if (!(eyes_cascade.load("src/classifiers/haarcascade_eye_tree_eyeglasses.xml"))) {
    cout << "error loading eye classifier" << endl;
    return -1;
  }

  unsigned long baud = 9600;
  std::string port("/dev/tty.usbmodem1412");
  serial::Serial mbed(port, baud, serial::Timeout::simpleTimeout(1000)); 

  if (!mbed.isOpen()) {
   
    cout << "could not open connection with mbed" << endl;
    return -1;
  } else {
    printf("mbed opened with baud rate:%u\n", mbed.getBaudrate());
  }
  if (DISPLAY) {
    namedWindow(display_window,
          CV_WINDOW_AUTOSIZE |
          CV_WINDOW_KEEPRATIO |
          CV_GUI_EXPANDED);
  }
  
  // Loop to capture frames
  while(cap.read(frame)) {
//    cv::resize(oriFrame, frame, cv::Size(960, 720));
    
    // Apply the classifier to the frame, i.e. find face
    detectFace(frame);
    frame_width = frame.cols;
    faceCenter.x = priorFace.x + priorFace.width/2;
    faceCenter.y = priorFace.y + priorFace.height/2;
    angle = atan2(faceCenter.x - K_logitech(1, 3), K_logitech(1, 1));
    angled = angle * 180 / PI;
    writeToMbed(angled, mbed);

    //printf("faceX: %d, faceY: %d, angle: %.2f\n", faceCenter.x, faceCenter.y, angled); 

    if (DISPLAY) {
      ellipse(frame, faceCenter, Size(priorFace.width/2, priorFace.height/2),
          0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0);
      cv::resize(frame, displayFrame, cv::Size(960, 720));
      
      imshow(display_window, displayFrame);
    }
      
    
    if(waitKey(30) >= 0) // spacebar
      break;
  }
  return 0;
}

/** 
function that writes an int value to MBED based on angle value
quantizes the angle into 16 ranges:
{ -Inf to -26, -26 to 22, ... 26 to 30, 30 to Inf }
write an int based on the input range index
*/
void writeToMbed(double angled, serial::Serial &mbed) {
  std::string angleString("15");
  //only compare to second last threshold, since last is infinite
  for (int i = 0; i < 15; i++) { 
    if ((angled < angleThreshold[i]) && (mbed.isOpen())) {
      std::string angleString = std::to_string(i) + std::string("\n");
      mbed.flushOutput(); //only write the most recent value
      mbed.write(angleString);
      return;
    }
  }
//write 15 if haven't written yet
  mbed.flushOutput(); //only write the most recent value
  mbed.write(angleString);
}

/**
compare function
return area(face1) > area(face2)
*/

bool compareBigger(Rect face1, Rect face2) { //biggest -> smallest
  return face1.width * face1.height > face2.width * face2.height;
}
/**
compare function
return distance_from_priorFace(face1) < distance_from_priorFace((face2)
*/

bool compareDistance(Rect face1, Rect face2) { //sort smallest -- biggest
  Point2d priorFaceCenter(priorFace.x + priorFace.width/2, priorFace.y + priorFace.height/2);

  double sq_x1 = pow(face1.x + face1.width/2 - priorFaceCenter.x, 2);
  double sq_y1 = pow(face1.y + face1.height/2 - priorFaceCenter.y, 2);

  double sq_x2 = pow(face2.x + face2.width/2 - priorFaceCenter.x, 2);
  double sq_y2 = pow(face2.y + face2.height/2 - priorFaceCenter.y, 2);

  return (sq_x1 + sq_y1) < (sq_x2 + sq_y2);

}

bool comparePeripheral(Rect face1, Rect face2) {
  return (abs((face1.x + face1.width/2) - frame_width/2) > abs((face2.x + face2.width/2) - frame_width/2));
}

/** 
Detect face and sets the global variable priorFace 
If priorFace is uninitialized, set to biggest face.
If multiple faces are found, set to face with nearest distance to priorFace.
If no face is found, don't set to new value.
*/
void detectFace(Mat frame) {
  
  std::vector<Rect> faces;
  Mat frame_gray, frame_lab;
  int minNeighbors = 2;

  cvtColor(frame, frame_gray, COLOR_BGR2GRAY);   // Convert to gray
  equalizeHist(frame_gray, frame_gray);          // Equalize histogram
  
  // Detect face with open source cascade
  face_cascade.detectMultiScale(frame_gray, faces,
				1.1, minNeighbors,
				0|CASCADE_SCALE_IMAGE, Size(30, 30));

  if (faces.size() == 0) { //return old face
    return; 
  }

  //if priorCenter is not initalized, initialize to biggest face
  if (priorFace.x == 0) { 
    //sort by size to get biggest face
    std::sort(faces.begin(), faces.end(), compareBigger); 
  } else {
    //sort by distance from prior face (farthest to nearest)
    std::sort(faces.begin(), faces.end(), compareDistance); 
  }
  priorFace = Rect_<double>(faces[0].x, faces[0].y, faces[0].width, faces[0].height);
  return;
}

void test(){

  priorFace = Rect_<double>(0, 0, 0, 0);
  Rect f1(0, 0, 1, 1);
  Rect f2(5, 6, 3, 3);
  Rect f3(-1, -1, 2, 2); 
  std::vector<Rect> faces;
  faces.push_back(f1);
  faces.push_back(f2);
  faces.push_back(f3);
  std::sort(faces.begin(), faces.end(), compareBigger);
  assert(faces[0].x == 5);
  assert(faces[1].x == -1);
  assert(faces[2].x == 0);
  cout << "sort compareBigger passed" << endl;

  std::sort(faces.begin(), faces.end(), compareDistance);
  assert(faces[0].x == -1);
  assert(faces[1].x == 0);
  assert(faces[2].x == 5);
  cout << "sort compareDistance passed" << endl;
  
} 

void testSerial() {
  unsigned long baud = 9600;
  std::string port("/dev/tty.usbmodem1412");
  serial::Serial mbed(port, baud, serial::Timeout::simpleTimeout(1000)); 

/*
  const uint8_t data[3] = "ab";
  int nbuf = 32;
  uint8_t buffer[nbuf];
    //for (int i = 0; i < 20; i++) {
    while(1) {
      if (mbed.isOpen()) {
        size_t available = mbed.available();
        //printf("available: %u", available);
        if (available > 0) {
          if (available > nbuf) {
            available = nbuf;
          }
          mbed.read(buffer, available);
          printf("%s\n", (char *)buffer); 
        }
        
      }
  }
*/
  printf("baudrate: %u, isOpen: %d \n", mbed.getBaudrate(), mbed.isOpen()); 
  int i = 0; 
  size_t wrote;
  std::string newl("\n");
  const std::string testString("hello\n");
  while(1) {
    if (mbed.isOpen()) {
      if (i >= 3) {
        i = 0;
      } else {
        i++;
      } 
      std::string dat = to_string(i++);
      dat = dat + newl;
      mbed.flushOutput();
      wrote = mbed.write((const std::string)dat);
      //wrote = mbed.write(testString);
    
      printf("wrote: %lu\n", wrote);
    }
  }

}
    
    
