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
#define CAM 1

using namespace std;
using namespace cv;

/* Function Headers */
void detectFace(Mat frame);
bool compareBigger(Rect face1, Rect face2);
bool compareDistance(Rect face1, Rect face2); 
bool comparePeripheral(Rect face1, Rect face2); 
void setMouseLocation(int event, int x, int y, int, void*); 
void writeToMbed(double angled, serial::Serial &mbed);
void test();
void testSerial(); 

/* global variables */
Rect priorFace(0, 0, 0, 0);
vector<Rect> faces;
CascadeClassifier face_cascade, eyes_cascade;
String display_window = "Display";
Point mouseLocation(0, 0);
int newMouseClick = 0;
int frame_width = 0;

int mouseCounter = 0;
/*camera calibration matrices */
Matx33f K_logitech(1517.6023, 0, 0, 0, 1517.6023, 0, 959.5, 539.5, 1);
Matx33f K_facetime(1006.2413, 0, 0, 0, 1006.2413, 0, 639.5, 359.5, 1); //ordered by cols

/*angle look-up tables*/
const float angleThreshold[15] = {-26, -22, -18, -14, -10, -6, -2, 2, 6, 10, 14, 18, 22, 26, 30}; 

int main() {
  if (TEST) {
    testSerial();
    return 1;
  }

  VideoCapture cap(0); // capture from default camera
  Mat frame;
  Mat displayFrame;
  double scale = 2.0;
  Point faceCenter(0, 0);  
  double angle = 0;
  double angled = 0;
  double w_half, h_half;
  unsigned long baud = 9600;
  std::string port("/dev/tty.usbmodem1412");
  serial::Serial mbed(port, baud, serial::Timeout::simpleTimeout(1000)); 
  Matx33f K;
  if (CAM == 0) {
    K = K_facetime; 
  } else {
    K = K_logitech;
  }

  const String displayText = "L-click a face to track it. R-click a point to lock onto it";
  Point org(50, 50);
  //initialize frame dimensions
  int displayW = cvRound(cap.get(CV_CAP_PROP_FRAME_WIDTH)/scale);
  int displayH = cvRound(cap.get(CV_CAP_PROP_FRAME_HEIGHT)/scale);
  
  if (!(face_cascade.load("src/classifiers/haarcascade_frontalface_alt.xml"))) {
    cout << "error loading face classifier" << endl;
    return -1;
  }
  if (!(eyes_cascade.load("src/classifiers/haarcascade_eye_tree_eyeglasses.xml"))) {
    cout << "error loading eye classifier" << endl;
    return -1;
  }

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
    setMouseCallback(display_window, setMouseLocation, NULL);

  }
  // Loop to capture frames
  while(cap.read(frame)) {
    cv::resize(frame, displayFrame, cv::Size(displayW, displayH));
    
    // Apply the classifier to the frame, i.e. find face
    detectFace(displayFrame);
    w_half = priorFace.width/2;
    h_half = priorFace.height/2;
    faceCenter.x = priorFace.x + w_half;
    faceCenter.y = priorFace.y + h_half;
    angled = fastAtan2(scale*faceCenter.x - K(1, 3), K(1, 1));
    if (angled > 180) {
      angled = 360.0 - angled;
    }
    //angled = angle * 180 / PI;
    writeToMbed(angled, mbed);

    printf("faceX: %d, faceY: %d, angle: %.2f\n", faceCenter.x, faceCenter.y, angled); 

    if (DISPLAY) {
      //selected face is pink
      ellipse(displayFrame, faceCenter, Size(w_half, h_half),
          0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0);
      //other faces are white
      for (int i = 1; i < faces.size(); i++) {
        w_half = faces[i].width/2;
        h_half = faces[i].height/2;
        faceCenter.x = faces[i].x + w_half;
        faceCenter.y = faces[i].y + h_half;
        ellipse(displayFrame, faceCenter, Size(w_half, h_half),
          0, 0, 360, Scalar( 255, 255, 255 ), 4, 8, 0);
      }
      putText(displayFrame, displayText, Point(30, 30), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 0));
      imshow(display_window, displayFrame);
      
    }

    faces.clear();
      
    if(waitKey(30) >= 0) // spacebar
      break;
  }
  return 0;

}

/** 
  callback function when mouse is clicked
  set priorFace to face closest to mouse click
*/
void setMouseLocation(int event, int x, int y, int, void*) {
  if (event == EVENT_LBUTTONDOWN || event == EVENT_RBUTTONDOWN) {
    mouseLocation.x = x;
    mouseLocation.y = y;
    newMouseClick = event;
    cout << "click" << event << endl;
    
  }
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
      angleString = std::to_string(i) + std::string("\n");
      break;
    }
  }
//write 15 if haven't written yet
  cout << angleString;
  mbed.write(angleString);
  mbed.flushOutput(); //only write the most recent value
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
  
  Mat frame_gray, frame_lab;
  int minNeighbors = 2;

  cvtColor(frame, frame_gray, COLOR_BGR2GRAY);   // Convert to gray
  equalizeHist(frame_gray, frame_gray);          // Equalize histogram
  
  // Detect face with open source cascade
  face_cascade.detectMultiScale(frame_gray, faces,
				1.1, minNeighbors,
				0|CASCADE_SCALE_IMAGE, Size(30, 30));

//if mouse has been left clicked, set priorFace to that face (Track that face)
//if mouse has been right clicked, set face to that location and don't track
  if (newMouseClick) {
    //arbitrary size needed in case click is R to draw the ellipse
    priorFace.x = mouseLocation.x - 50;
    priorFace.y = mouseLocation.y - 50;
    priorFace.width = 100;
    priorFace.height = 100;
  //only L click deactivates the lock
    if (newMouseClick == EVENT_RBUTTONDOWN) {
      faces.clear();
      return;
    }  else {
      newMouseClick = 0; 
    }
  } 

  if (faces.size() == 0) { // no face detected return old face
    return; 
  }

  //if priorCenter is not initalized, initialize to most peripheral face
  if (priorFace.x == 0) { 
    std::sort(faces.begin(), faces.end(), comparePeripheral); 
  } else {
    //sort by distance from prior face (farthest to nearest)
    std::sort(faces.begin(), faces.end(), compareDistance); 
  }

  //record face0 as priorFace for future sorting
  priorFace.x = faces[0].x; 
  priorFace.y = faces[0].y;
  priorFace.width = faces[0].width;
  priorFace.height = faces[0].height;
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
    
      printf("wrote: %lu\n", wrote);
    }
  }

}
    
    
