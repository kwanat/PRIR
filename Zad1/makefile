gauss_omp: gauss_omp.cpp
	g++ -fopenmp gauss_omp.cpp -o gauss_omp `pkg-config --cflags --libs opencv`
 
 # g++ `pkg-config opencv --cflags` -O3 -fopenmp -Wall gauss_omp.cpp `pkg-config opencv --libs` -o gauss_omp
 # g++ -fopenmp -Wall -std=c++11 gauss_omp.cpp -lm -o gauss_omp -I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib -lopencv_shape -lopencv_stitching -lopencv_objdetect -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_core