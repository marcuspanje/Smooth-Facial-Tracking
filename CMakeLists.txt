cmake_minimum_required(VERSION 2.8)
add_definitions(-std=c++11)
project( FT )
find_package( OpenCV REQUIRED )
find_library(SERIAL serial)

add_executable( guiSmoothFaceTracking smooth_face_tracking_gui.cpp )
add_executable( smoothFaceTracking smooth_face_tracking.cpp )
add_executable( basicFaceTracking basic_face_detection.cpp )
add_executable( improvedFaceTracking improved_face_detection.cpp )

target_link_libraries( smoothFaceTracking ${OpenCV_LIBS} ${SERIAL} )
target_link_libraries( guiSmoothFaceTracking ${OpenCV_LIBS} ${SERIAL} )
target_link_libraries( basicFaceTracking ${OpenCV_LIBS} )
target_link_libraries( improvedFaceTracking ${OpenCV_LIBS} )


