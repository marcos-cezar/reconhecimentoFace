#ifndef _CSVREADER_H_
#define _CSVREADER_H_

#include <string>
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;


class CSVReader 
{

private:
    string absoluteFilename;
    
public:

    CSVReader();
    virtual ~CSVReader();
    static void read_csv(const string &filename, vector<Mat> &images, vector<int> &labels, char separator = ';');

};

#endif /* _CSVREADER_H_ */










