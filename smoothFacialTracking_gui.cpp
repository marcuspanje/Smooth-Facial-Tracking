#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <string>

#define PI 3.14159
#define DISPLAY 1
#define TEST 0

using namespace std;
using namespace cv;

/* Function Headers */
void drawEllipses(Mat frame);
void detectFace(Mat frame);
bool compareBigger(Rect face1, Rect face2);
bool compareToMouse(Rect face1, Rect face2);
void mouseClickSort(int event, int x, int y, int,  void* data);
//bool compareDistance(Rect face1, Rect face2); 
void test();

/* global variables */
int frame_size = 0;
std::vector<Rect> priorFaces;
Rect currFace;
CascadeClassifier face_cascade, eyes_cascade;
String display_window = "Display";
String face_window = "Face View";
Point mouseLocation = Point(0,0);
const int NUM_FACES = 4;
bool sortWithMouse = false;

/*camera calibration matrices */
Matx33f K_logitech(1517.6023, 0, 0, 0, 1517.6023, 0, 959.5, 539.5, 1);
Matx33f K_facetime(1006.2413, 0, 0, 0, 1006.2413, 0, 639.5, 359.5, 1); //ordered by cols

/*angle look-up tables*/
const float angleThreshold[15] = {-26, -22, -18, -14, -10, -6, -2, 2, 6, 10, 14, 18, 22, 26}; 


int main() {
  if (TEST) {
    test();
    return 1;
  }

  VideoCapture cap(0); // capture from default camera
  Mat frame;

  if (!(face_cascade.load("src/classifiers/haarcascade_frontalface_alt.xml"))) {
    cout << "error loading face classifier" << endl;
    return -1;
  }
  if (!(eyes_cascade.load("src/classifiers/haarcascade_eye_tree_eyeglasses.xml"))) {
    cout << "error loading eye classifier" << endl;
    return -1;
  }

  namedWindow(display_window,
	      CV_WINDOW_NORMAL |
	      CV_WINDOW_KEEPRATIO |
	      CV_GUI_EXPANDED);
  
  // Loop to capture frames
  while(cap.read(frame)) {
    
    // Apply the classifier to the frame, i.e. find face
     frame_size = frame.cols;
     detectFace(frame);

    if (DISPLAY) {
      drawEllipses(frame);
      setMouseCallback(display_window, mouseClickSort, NULL);
      imshow(display_window, frame);
    }
      
    
    if(waitKey(30) >= 0) // spacebar
      break;
  }
  return 0;
}

void drawEllipses(Mat frame) {
   int numPFaces = static_cast<int>(priorFaces.size());
   Scalar currColor = Scalar(255, 255, 255);
   for(int n = 0; n < std::min(NUM_FACES, numPFaces); n++) {
      if( n == 1){
        currColor = Scalar(255, 0, 255);
      }
      Rect priorFace = priorFaces[n];
      Point faceCenter(priorFace.x + priorFace.width/2, priorFace.y + priorFace.height/2);
      ellipse(frame, faceCenter, Size(priorFace.width/2, priorFace.height/2),
                        0, 0, 360, currColor, 4, 8, 0);
    // (debugging)  cout << "face: " << n << " x: " << priorFace.x << " y: " << priorFace.y << endl;
   }
}

void mouseClickSort(int event, int x, int y,int, void* data) {

   if( event == EVENT_LBUTTONDOWN ){
      mouseLocation.x = x;
      mouseLocation.y = y;
      sortWithMouse = true;
      cout << "CLICK!" << endl;
   }

}

bool compareBigger(Rect face1, Rect face2) { //biggest -> smallest
  return face1.width * face1.height > face2.width * face2.height;
}

/**
Class used to compare faces by which one is closest to the mouse.
*/
bool compareToMouse(Rect face1, Rect face2){
    double sq_x1 = pow(face1.x + face1.width/2 - mouseLocation.x, 2);
    double sq_y1 = pow(face1.y + face1.height/2 - mouseLocation.y, 2);
                                                                    
    double sq_x2 = pow(face2.x + face2.width/2 - mouseLocation.x, 2);
    double sq_y2 = pow(face2.y + face2.height/2 - mouseLocation.y, 2);

    return (sq_x1 + sq_y1) < (sq_x2 + sq_y2);
}

bool comparePeripheral(Rect face1, Rect face2) {
     return abs((face1.x + face1.width/2) - (frame_size)/2) > abs((face2.x + face2.width/2) - (frame_size)/2);
}

/**
Class used to compare faces by closeness to another face.
*/

bool compareDistance(Rect face1, Rect face2)  {
        Point2d priorFaceCenter(currFace.x + currFace.width/2, currFace.y + currFace.height/2);
       
        double sq_x1 = pow(face1.x + face1.width/2 - priorFaceCenter.x, 2);
        double sq_y1 = pow(face1.y + face1.height/2 - priorFaceCenter.y, 2);
                                                                             
        double sq_x2 = pow(face2.x + face2.width/2 - priorFaceCenter.x, 2);
        double sq_y2 = pow(face2.y + face2.height/2 - priorFaceCenter.y, 2);

        return (sq_x1 + sq_y1) < (sq_x2 + sq_y2);
}
 
/** 
Detect face and sets the global variable priorFace 
If priorFace is uninitialized, set to biggest face. Arnab: Set to face closest to left or right edge.
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

  if (faces.size() == 0) { //return old faces
    return; 
  }
  int seenFaces = static_cast<int>(faces.size());

  //if priorCenter is not initalized, initialize the first faces to the faces closest to the edges 
  if (priorFaces.size() == 0) { 
    //sort by size to get faces nearest the edges
    std::sort(faces.begin(), faces.end(), comparePeripheral);
    
    for( int i = 0; i < std::min(NUM_FACES, seenFaces); i++) {
      priorFaces.push_back(faces[i]);
    }
    
    //If not enough faces are recognized, set the rest to "empty" rectangles
    if(seenFaces < NUM_FACES) {   
       for( int m = seenFaces; m < NUM_FACES; m++){
         priorFaces.push_back(Rect(0,0,0,0));
       }
    }
 
  } else {
    for( int j = 0; j < std::min(NUM_FACES, seenFaces); j++) {
      if(sortWithMouse) {
         std::sort(faces.begin(), faces.end(), compareToMouse);
         priorFaces[j] = faces[j];
      } else if (!sortWithMouse){
         currFace = priorFaces[j];
         std::sort(faces.begin(), faces.end(), compareDistance); //Sort faces by distance to the face previously in that index.
         priorFaces[j] = faces[0];
      }
    }

    sortWithMouse = false;
    if(seenFaces < NUM_FACES) { 
       for(int y = seenFaces; y < NUM_FACES; y++){
          priorFaces[y] = Rect(0,0,0,0);
       }
    }
  }
  return;
}

void test(){
  Rect f1(0, 0, 1, 1);
  Rect f2(5, 6, 3, 3);
  Rect f3(-1, -1, 2, 2); 
  std::vector<Rect> faces;
  faces.push_back(f1);
  faces.push_back(f2);
  faces.push_back(f3);
  mouseLocation.x = 5;
  mouseLocation.y = 6;
  std::sort(faces.begin(), faces.end(), compareToMouse);
  assert(faces[0].x == 5);
  assert(faces[1].x == -1);
  assert(faces[2].x == 0);
  cout << "sort compareBigger passed" << endl;

} 

