#include "CSVReader.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

//Construtor
CSVReader::CSVReader() {
}

//Destrutor
CSVReader::~CSVReader() {
}

void CSVReader::read_csv(const string &filename, vector<Mat> &images, vector<int> &labels, char separator) 
{
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message =
            "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}



















