//
//  hashing.hpp
//  Geometric hashing - code adapted from GeoHash by Daniel Mesham
//

#ifndef hashing_hpp
#define hashing_hpp

#include "hash_2d.h"

#include <opencv2/core/core.hpp>
#include <iostream>
#include <stdio.h>
#include <map>

using namespace std;
using namespace cv;

class HashTable {
public:
    HashTable(hash_2d table_in, vector<int> basis_in) : table(table_in), basis(basis_in) {};
    
    hash_2d table;
    vector<int> basis;
    float votes = 0;
    
    bool operator > (const HashTable& ht) const
    {
        return (votes > ht.votes);
    }
    
    bool operator < (const HashTable& ht) const
    {
        return (votes < ht.votes);
    }
};

// correspondance map
class CorrInfo {
public:
    CorrInfo(map<int, int> corr_map_in, float scale_in, float rot_in, float votes_in) : corr_map(corr_map_in), scale(scale_in), rot(rot_in), votes(votes_in) {};
    
    map<int, int> corr_map;
    float votes = 0;
    float scale;
    float rot;
    
    bool operator > (const CorrInfo& ci) const
    {
        //return (corr_map.size() > 2) && (votes > ci.votes);
        return (corr_map.size() > ci.corr_map.size()) || ((corr_map.size() == ci.corr_map.size()) && (votes > ci.votes));
        
    }
};

class hashing {
public:
    static HashTable createTable(vector<int> basisID, vector<Point2f> modelPoints, float sigma);
    static vector<HashTable> voteForTables(vector<HashTable> tables, vector<Point2f> imgPoints, vector<int> imgBasis);
    static vector<HashTable> clearVotes(vector<HashTable> tables);
    static vector<Mat> getOrderedPoints(vector<int> imgBasis, HashTable ht, vector<Point2f> modelPoints, vector<Point2f> imgPoints);
    static map<int, int> getMatchedPoints(vector<int> imgBasis, HashTable ht, vector<Point2f> modelPoints, vector<Point2f> imgPoints, float &scale, float &rot, float distThresh, float estScale=-1);
    static Point2f basisCoords(vector<Point2f> basis, Point2f p);
    static Mat pointsToMat2D(vector<Point2f> points);
    static Mat pointsToHomog(Mat m);
    static float calcBinWidth(vector<Point2f> coords, float sigma);
};

#endif /* hashing_hpp */
