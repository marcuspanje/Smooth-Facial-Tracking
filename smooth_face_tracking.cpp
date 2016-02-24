#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>
#include <cmath>
#define PI 3.14159

using namespace std;
using namespace cv;

/* Function Headers */
Point detectFace( Mat frame, Point priorCenter );

CascadeClassifier face_cascade, eyes_cascade;

String display_window = "Display";
String face_window = "Face View";

/*camera calibration matrices */
Matx33f K_logitech(1517.6023, 0, 959.5,  0, 1517.6023, 539.5, 0, 0, 1);
Matx33f K_facetime(1006.2413, 0, 639.5, 0, 1006.2413, 359.5, 0, 0, 1);
double f_logitech = 1517.6023;
double f_facetime = 1006.2413;

int main() {

  VideoCapture cap(0); // capture from default camera
  Mat frame;
  Point priorCenter(0, 0);  
  Matx31f priorCenterH(0, 0, 1); 
  Matx31f priorCenter3D(0, 0, 0); 
 // Mat priorCenterHomogenous(3, 1, CV_64F);
  //P priorCenter3D(0, 0, 0);
  double angle = 0;
  double angled = 0;

  face_cascade.load("src/classifiers/haarcascade_frontalface_alt.xml"); // load face classifiers
  eyes_cascade.load("src/classifiers/haarcascade_eye_tree_eyeglasses.xml"); // load eye classifiers

  namedWindow(face_window,
	      CV_WINDOW_AUTOSIZE |
	      CV_WINDOW_FREERATIO |
	      CV_GUI_EXPANDED);
  
  // Loop to capture frames
  while(cap.read(frame)) {
    
    // Apply the classifier to the frame, i.e. find face
    priorCenter = detectFace(frame, priorCenter);
    cout << "x: " << priorCenter.x << ", y: " << priorCenter.y << endl;


    //convert to angle 
    priorCenterH(1, 1) = priorCenter.x;
    priorCenterH(2, 1) = priorCenter.y;

    priorCenter3D = K_facetime.inv() * priorCenterH; 


    angle = atan2(priorCenter3D(1, 1), priorCenter3D(3, 1)); //theta = arctan(x/z)
    angled = angle * 180 / PI;

    cout << "angle: " << angled << endl;
    
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

// Detect face and display it
Point detectFace(Mat frame, Point priorCenter) {
  
  std::vector<Rect> faces;
  Mat frame_gray, frame_lab, output, temp;
  int h = frame.size().height - 1;
  int w = frame.size().width - 1;
  int minNeighbors = 2;
  bool faceNotFound = false;

  cvtColor(frame, frame_gray, COLOR_BGR2GRAY);   // Convert to gray
  equalizeHist(frame_gray, frame_gray);          // Equalize histogram
  
  // Detect face with open source cascade
  face_cascade.detectMultiScale(frame_gray, faces,
				1.1, minNeighbors,
				0|CASCADE_SCALE_IMAGE, Size(30, 30));

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
      break;
    }
    
    // Check to see if it's probably the same user
    if(abs(center.x - priorCenter.x) < frame.size().width / 6 &&
       abs(center.y - priorCenter.y) < frame.size().height / 6) {

      // Check to see if the user moved enough to update position
      if(abs(center.x - priorCenter.x) < 7 &&
	 abs(center.y - priorCenter.y) < 7){
	center = priorCenter;
      }

      // Smooth new center compared to old center
      center.x = (center.x + 2*priorCenter.x) / 3;
      center.y = (center.y + 2*priorCenter.y) / 3;
      priorCenter = center;
      
      // output frame of only face
      temp = outputFrame(frame, center, w, h);
                 
      break; // exit, primary users face probably found
      
    } else {
      faceNotFound = true;
    }
  }

  if(faceNotFound) {

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
