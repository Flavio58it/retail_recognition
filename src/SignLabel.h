#ifndef SignLabel_h
#define SignLabel_h

#include <string>

using namespace std;

class SignLabel
{
    private:
        string labelText;
        vector<string> labelWords;
        int wordCount;
        bool findStatus;

    public:
        float prob;

        SignLabel(string text); //constructor
        string getLabel(); //return label text
        string getWord(int index);//return specific word in label
        bool getStatus(); //return find status
        void setStatus(bool flag);
        int numWords(); //return number of words in label
};

#endif // signLabel

