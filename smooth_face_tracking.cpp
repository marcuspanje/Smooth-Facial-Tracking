#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <algorithm>
#define PI 3.14159
#define DISPLAY 1

using namespace std;
using namespace cv;

/* Function Headers */
Rect detectFace(Mat frame);
Rect priorFace(0, 0, 0, 0);

CascadeClassifier face_cascade, eyes_cascade;

String display_window = "Display";
String face_window = "Face View";

/*camera calibration matrices */
Matx33f K_logitech(1517.6023, 0, 0, 0, 959.5,  0, 1517.6023, 539.5, 1);
Matx33f K_facetime(1006.2413, 0, 0, 0, 1006.2413, 0, 639.5, 359.5, 1); //ordered by cols

int main() {

  VideoCapture cap(0); // capture from default camera
  Mat frame;
  Rect face; //primary face;
  Point faceCenter(0, 0);  
  Matx31f priorCenterH(0, 0, 1); 
  Matx31f priorCenter3D(0, 0, 0); 
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

  namedWindow(display_window,
	      CV_WINDOW_NORMAL |
	      CV_WINDOW_KEEPRATIO |
	      CV_GUI_EXPANDED);
  
  // Loop to capture frames
  while(cap.read(frame)) {
    
    // Apply the classifier to the frame, i.e. find face
    face = detectFace(frame);
    faceCenter.x = face.x + face.width/2;
    faceCenter.y = face.y + face.height/2;
    angle = atan2(faceCenter.x - K_facetime(1, 3), K_facetime(1, 1));
    angled = angle * 180 / PI;

    cout << "angle: " << angled << endl;

    if (DISPLAY) {
      ellipse(frame, faceCenter, Size(face.width, face.height),
          0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0);
      imshow(display_window, frame);
    }
      
    
    if(waitKey(30) >= 0) // spacebar
      break;
  }
  return 0;
}

/**
 * Output a frame of only the the rectangle centered at point
 */
Mat outputFrame(Mat frame, Point center, int w, int h) {

  int x = (center.x - w/2);
  int y = (center.y - 3*h/5);

  if(x + w > frame.size().width - 2 || x < 0 ||
     y + h > frame.size().height - 2 || y < 0 &&
     frame.size().width > 16 &&
     frame.size().height > 16)
    return frame(Rect(5, 5, 10, 10));
  
  // output frame of only face
  return frame(Rect(x, y, w, h));
}

// Find face from eyes
Point faceFromEyes(Point priorCenter, Mat face) {

  std::vector<Rect> eyes;
  int avg_x = 0;
  int avg_y = 0;

  // Try to detect eyes, if no face is found
  eyes_cascade.detectMultiScale(face, eyes, 1.1, 2,
				0 |CASCADE_SCALE_IMAGE, Size(30, 30));

  // Iterate over eyes
  for(size_t j = 0; j < eyes.size(); j++) {

    // centerpoint of eyes
    Point eye_center(priorCenter.x + eyes[j].x + eyes[j].width/2,
		     priorCenter.y + eyes[j].y + eyes[j].height/2);

    // Average center of eyes
    avg_x += eye_center.x;
    avg_y += eye_center.y;
  }

  // Use average location of eyes
  if(eyes.size() > 0) {
    priorCenter.x = avg_x / eyes.size();
    priorCenter.y = avg_y / eyes.size();
  }

  return priorCenter;
}

// Rounds up to multiple
int roundUp(int numToRound, int multiple) {
  
  if (multiple == 0)
    return numToRound;

  int remainder = abs(numToRound) % multiple;
  if (remainder == 0)
    return numToRound;
  if (numToRound < 0)
    return -(abs(numToRound) - remainder);
  return numToRound + multiple - remainder;
}

bool compareBigger(Rect face1, Rect face2) { //biggest -> smallest
  return face1.width * face1.height > face2.width * face2.height;
}

bool compareDistance(Rect face1, Rect face2) { //sort smallest -- biggest
  Point2d priorFaceCenter(priorFace.x + priorFace.width/2, priorFace.y + priorFace.height/2);

  double sq_x1 = pow(face1.x + face1.width/2 - priorFaceCenter.x, 2);
  double sq_y1 = pow(face1.y + face1.height/2 - priorFaceCenter.y, 2);

  double sq_x2 = pow(face2.x + face2.width/2 - priorFaceCenter.x, 2);
  double sq_y2 = pow(face2.y + face2.height/2 - priorFaceCenter.y, 2);

  return (sq_x1 + sq_y1) < (sq_x2 + sq_y2);

}

// Detect face and display it
Rect detectFace(Mat frame) {
  
  std::vector<Rect> faces;
  Mat frame_gray, frame_lab, output, temp;
  int h = frame.size().height - 1;
  int w = frame.size().width - 1;
  int minNeighbors = 2;

  cvtColor(frame, frame_gray, COLOR_BGR2GRAY);   // Convert to gray
  equalizeHist(frame_gray, frame_gray);          // Equalize histogram
  
  // Detect face with open source cascade
  face_cascade.detectMultiScale(frame_gray, faces,
				1.1, minNeighbors,
				0|CASCADE_SCALE_IMAGE, Size(30, 30));

  if (faces.size() == 0) { //return old face
    return priorFace;
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
  return priorFace;
}
  
 /* 

  // iterate over faces
  for(size_t i = 0; i < faces.size(); i++) {
    
    

    // Find center of face
    Point center(faces[i].x + faces[i].width/2,
		 faces[i].y + faces[i].height/2);

    // Generate width and height of face, round to closest 1/4 of frame height
    h = roundUp(faces[i].height, frame.size().height / 4);
    w = 3 * h / 5;

    // If priorCenter not yet initialized, initialize
    if(priorCenter.x == 0) {
      priorCenter = center;
      temp = outputFrame(frame, center, w, h);
      break;// get the first face
    }
    
    // Check to see if it's probably the same user
    if(abs(center.x - priorCenter.x) < frame.size().width / 6 &&
       abs(center.y - priorCenter.y) < frame.size().height / 6) {

      // Check to see if the user moved enough to update position
      if(abs(center.x - priorCenter.x) < 7 &&
	 abs(center.y - priorCenter.y) < 7){
	center = priorCenter; //keep old center
      }

      // Smooth new center compared to old center
      center.x = (center.x + 2*priorCenter.x) / 3;
      center.y = (center.y + 2*priorCenter.y) / 3;
      priorCenter = center;
      
      // output frame of only face
      //temp = outputFrame(frame, center, w, h);
                 
      break; // exit, primary users face probably found
      
    } else { //new face region is different 
      faceNotFound = true;
    }
  }

//iterated through all faces, and primary user face not found

  if(faceNotFound) {
      
    cout << "detect from eyes" << endl;
    // Findface from eyes
    Rect r(priorCenter.x, priorCenter.y, w, h);
    if(priorCenter.x + w > frame_gray.size().width - 2 &&
       priorCenter.y + h > frame_gray.size().height - 2){

      priorCenter = faceFromEyes(priorCenter, frame_gray(r));
    
      // Generate temporary face location
      temp = outputFrame(frame, priorCenter, w, h);
    }    
  }
  
  // Check to see if new face found
  if(temp.size().width > 2)
    output = temp;
  else
    output = frame;
  
  // Display only face
  imshow(face_window, output);

  if(output.size().width > 2)
    // Draw ellipse around face
    ellipse(frame, priorCenter, Size(w/2, h/2),
	    0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0);
  
  // Display output
  imshow( display_window, frame );
  
  return priorCenter;
}
*/
