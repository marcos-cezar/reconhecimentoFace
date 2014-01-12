#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stddef.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "CSVReader.hpp"

using namespace cv;
using namespace std;

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
Ptr<FaceRecognizer> model;
RNG rng(12345);

int im_width;
int im_height;

void detectAndDisplay(Mat frame) 
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    
    //-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2,
                                  0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

    for (size_t i = 0; i < faces.size(); i++) {
        Mat original = frame.clone();

        Mat gray;

        cvtColor(original, gray, CV_BGR2GRAY);

        Rect face_i = faces[i];
        Mat face = gray(face_i);

        Mat face_resized;
        cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0,
                   INTER_CUBIC);

        int prediction = model->predict(face_resized);

        rectangle(frame, face_i, CV_RGB(0, 255, 0), 1);

        string box_text = format("PessoaID = %d", prediction);
        
        //Obtem o canto superior esquerdo da imagem.
        int pos_x = std::max(face_i.tl().x - 10, 0);
        int pos_y = std::max(face_i.tl().y - 10, 0);

        // And now put it into the image:
        if (prediction == 0) {
            box_text = "Desconhecido";
        } else if (prediction == 1) {
            box_text = "Oi, Allan!";
        } else if (prediction == 2) {
            box_text = "Oi, Laionara!";
        }

        Point center(faces[i].x + faces[i].width * 0.5,
                     faces[i].y + faces[i].height * 0.5);
        putText(frame, box_text, center, FONT_HERSHEY_PLAIN, 1.0,
                CV_RGB(0, 255, 0), 2.0);

        /*ellipse(frame, center,
          Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360,
          Scalar(0, 255, 0), 1, 8, 0);*/

        Mat faceROI = frame_gray(faces[i]);
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2,
                                      0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

        for (size_t j = 0; j < eyes.size(); j++) {
            Point center(faces[i].x + eyes[j].x + eyes[j].width * 0.5,
                         faces[i].y + eyes[j].y + eyes[j].height * 0.5);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            circle(frame, center, radius, Scalar(0, 0, 255), 1, 8, 0);
        }
    }
    //-- Show what you got
    imshow(window_name, frame);
}

int readFromVideo(string csvPath) 
{

    vector<Mat> images;
    vector<int> labels;
    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    try {
        CSVReader::read_csv(csvPath, images, labels);
    } catch (cv::Exception& e) {
        cout << "Error opening file \"" << csvPath << "\". Reason: " << e.msg
             << endl;
        // nothing more we can do
        exit(1);
    }

    cout << "images = " << images.size() << endl;
    cout << "labels = " << labels.size() << endl;

    im_width = images[0].cols;
    im_height = images[0].rows;

    /* cout << "im_width = "<< im_width << endl;
       cout << "im_height = "<< im_height << endl;*/
    // Create a FaceRecognizer and train it on the given images:
    model = createFisherFaceRecognizer();
    model->train(images, labels);

    VideoCapture cap(0);

    if (!cap.isOpened()) {
        cout << "Deu pau." << endl;
        return -1;
    }

    double fps = cap.get(CV_CAP_PROP_FRAME_WIDTH);

    namedWindow(window_name, CV_WINDOW_AUTOSIZE);

    while (true) {
        
        Mat frame;
        bool success = cap.read(frame);

        if (!success) {
            break;
        }

        detectAndDisplay(frame);

        //imshow("The Video", frame);

        if (waitKey(30) == 27) {
            cout << "esc key pressed";
            break;
        }

    }

    return 0;

}

int main(int argc, char** argv) {

    if(argc < 3) {
        cout << "File to feed pattern recognizer engine not passed throuth command line." << endl;
        return EXIT_FAILURE;
    }

    if (!face_cascade.load(argv[1])) {
        cout << "Deu erro no carregamento do xml" << endl;
        return EXIT_FAILURE;
    }


    if (!eyes_cascade.load(argv[2])) {
        cout << "Deu erro no carregamento\n" << endl;
        return EXIT_FAILURE;
    }


    if (argv[3] == NULL) {
        cout << "Path of CSV file not passed throuth command line." << endl;
        return EXIT_FAILURE;
    }

    string csvPath = argv[3];

    readFromVideo(csvPath);

    waitKey(0);

    return EXIT_SUCCESS;
}
