#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <serial/serial.h>
#include <string>

#define PI 3.14159
#define DISPLAY 0
#define TEST 0

using namespace std;
using namespace cv;

int main() {
  double x[10] = {
