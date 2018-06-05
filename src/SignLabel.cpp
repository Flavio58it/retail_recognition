#include "SignLabel.h"
#include <iostream>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/asio.hpp>
#include <vector>

using namespace std;

SignLabel::SignLabel(string text)
{
    //prob = 0.0;
    labelText = text;
    findStatus=false;


    boost::algorithm::split(labelWords, text, boost::is_any_of(" "));

    //if(labelWords[0]==" ")
        //labelWords.erase(labelWords.begin());

    wordCount = labelWords.size();

}

string SignLabel::getLabel()
{
    return labelText;
}

string SignLabel::getWord(int index)
{
    return labelWords[index];
}

bool SignLabel::getStatus()
{
    return findStatus;
}

void SignLabel::setStatus(bool flag)
{
    findStatus = flag;
}

int SignLabel::numWords()
{
    return wordCount;
}
