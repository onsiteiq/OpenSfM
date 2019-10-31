//
//  hashing.cpp
//  Geometric hashing - code adapted from GeoHash by Daniel Mesham
//

#include <map>
#include <opencv2/calib3d.hpp>
#include "hashing.hpp"

float hashing::calcBinWidth(vector<Point2f> basis, float sigma) {
    /*
     * half of the distance between the two basis points will be equal to
     * 1 bin, if binWidth is 1. therefore, to have 1 bin correspond to
     * 3*sigma, we need binWidth to be the following
     */
    //return 0.125;
    return 2.5*sigma/(norm(basis[0]-basis[1])/2);
}

HashTable hashing::createTable(vector<int> basisID, vector<Point2f> modelPoints, float sigma=50.0) {
    /*
     * Creates a hash table using the given point indices to determine the basis points.
     * All remaining model points are then hashed acording to their basis-relative
     * coordinates
     */
    vector<Point2f> basis = { modelPoints[basisID[0]], modelPoints[basisID[1]] };
    float binWidth = calcBinWidth(basis, sigma);
    hash_2d gh(binWidth);
    
    for (int j = 0; j < modelPoints.size(); j++) {
        // Do not hash basis
        if (j == basisID[0] || j == basisID[1]) continue;
        Point2f bc = basisCoords(basis, modelPoints[j]);
        gh.add_point( point(bc.x, bc.y, j) );
    }
    
    return HashTable(gh, basisID);
}


vector<HashTable> hashing::voteForTables(vector<HashTable> tables, vector<Point2f> imgPoints, vector<int> imgBasis) {
    /*
     * Generates a vote for each table based on how many model points lie in the same
     * bin as a given image point when in the coordinate system of the given basis.
     * Sorts the tables based on the number of votes.
     */
    
    tables = clearVotes(tables);
    
    vector<Point2f> basis = { imgPoints[imgBasis[0]], imgPoints[imgBasis[1]] };
    
    for (int i = 0; i < imgPoints.size(); i++) {
        if (i == imgBasis[0] || i == imgBasis[1]) continue;
        Point2f bc = basisCoords(basis, imgPoints[i]);
        point pt = point(bc.x, bc.y);
        
        // Check for matches in each table
        for (int j = 0; j < tables.size(); j++) {
            vector<point> points = tables[j].table.points_in_bin(pt);
            if (points.size() > 0) tables[j].votes += 1.0;
            
            // partial votes for neighboring bins (see "Optimal Affine-Invariant Point Matching")
            //cout << "# points in neighborhood = " << tables[j].table.points_in_neighborhood(pt) << endl;
            tables[j].votes += 0.25 * tables[j].table.points_in_neighborhood(pt);
        }
    }
    sort(tables.begin(), tables.end(), greater<HashTable>());
    return tables;
}

vector<HashTable> hashing::clearVotes(vector<HashTable> tables) {
    for (int i = 0; i < tables.size(); i++) {
        tables[i].votes = 0;
    }
    return tables;
}

vector<Mat> hashing::getOrderedPoints(vector<int> imgBasis, HashTable ht, vector<Point2f> modelPoints, vector<Point2f> imgPoints) {
    /*
     * Returns a Mat of model points and image points for use in the least squares
     * algorithm. The orders of both are the same (i.e. the i-th model point corresponds
     * to the i-th image point).
     */
    vector<int> basisIndex = ht.basis;
    vector<Point2f> orderedModelPoints;
    vector<Point2f> orderedImgPoints;
    vector<Point2f> basis = { imgPoints[imgBasis[0]], imgPoints[imgBasis[1]] };
    
    for (int j = 0; j < imgPoints.size(); j++) {
        Point2f bc = basisCoords(basis, imgPoints[j]);
        
        // If a basis point...
        if (j == imgBasis[0]) {
            orderedModelPoints.push_back(modelPoints[basisIndex[0]]);
            orderedImgPoints.push_back(imgPoints[j]);
        }
        else if (j == imgBasis[1]) {
            orderedModelPoints.push_back(modelPoints[basisIndex[1]]);
            orderedImgPoints.push_back(imgPoints[j]);
        }
        
        // If not a basis point...
        else {
            point pt = point(bc.x, bc.y);
            vector<point> binPoints = ht.table.points_in_bin(pt);
            
            if (binPoints.size() > 0) {
                // Take the first point in the bin
                int modelPt_ID = binPoints[0].getID();
                
                orderedModelPoints.push_back(modelPoints[modelPt_ID]);
                orderedImgPoints.push_back(imgPoints[j]);
            }
        }
    }
    
    Mat newModel = pointsToMat2D(orderedModelPoints).t();
    Mat imgTarget = pointsToMat2D(orderedImgPoints).t();
    
    return {newModel, imgTarget};
}

map<int,int> hashing::getMatchedPoints(vector<int> imgBasis, HashTable ht, vector<Point2f> modelPoints, vector<Point2f> imgPoints, float& scale, float& rot, float distThresh, float estScale) {
    
    map<int,int> retVal;

    /*
     * Returns a map of image points (key) and model points (value)
     */
    vector<Mat> orderedPoints = getOrderedPoints(imgBasis, ht, modelPoints, imgPoints);
    
    Mat newModel = orderedPoints[0];
    Mat newTarget = orderedPoints[1];
    
    // estimate similarity transformation. it's important to use LMEDS not RANSAC
    Mat simTrans = estimateAffinePartial2D(newTarget, newModel, noArray(), LMEDS);
    simTrans.convertTo(simTrans, CV_32F);
    
    rot = 180.0 * atan2(simTrans.at<float>(1, 0), simTrans.at<float>(0, 0))/M_PI;
    scale = sqrt(pow(simTrans.at<float>(0, 0), 2) + pow(simTrans.at<float>(1, 0), 2));
    cout << "estimated rotation =" << rot << ", scale = " << scale << endl;
    
    // check if obtained scale is close to estimated scale. return empty map if far apart
    if(estScale != -1) {
        float ratio = scale/estScale;
        if(ratio < 0.5 || ratio > 2.0) return retVal;
    }

    // convert image points to homogeneous coordinates, then transform with estimated similarity
    Mat imgPointsMat = pointsToHomog(pointsToMat2D(imgPoints)); // 3xN
    Mat imgPointsMatTrans = simTrans*imgPointsMat;

    for (int i = 0; i < imgPointsMatTrans.cols; i++) {
        cv::Point2f p(imgPointsMatTrans.at<float>(0,i), imgPointsMatTrans.at<float>(1,i));
        
        // for each transformed image point, calculates its distance of each model points.
        vector<float> distances;
        for (int j=0; j < modelPoints.size(); j++) {
            distances.push_back(norm(modelPoints[j]-p));
        }
        
        // find the min and 2nd smallest distances of this image point to model points. if the
        // two are not very close, count this as a correspondance
        vector<float>::iterator itr = min_element(distances.begin(), distances.end());
        if (distThresh > *itr) {
            int k = (int) distance(distances.begin(), itr);
            
            nth_element(distances.begin(), distances.begin()+1, distances.end());
            if ((distances[1] - distances[0]) > 3.0*distThresh) {
                retVal[i] = k;
            }
            // debugging
            /*
             else {
                cout << i << "->" << k << " discarded, distance to 2nd min = " << (distances[1] - distances[0]) << endl;
            }
            */
        }
    }

    return retVal;
}


Point2f hashing::basisCoords(vector<Point2f> basis, Point2f p) {
    // Converts the coordinates of point p into the reference frame with the given basis
    Point2f O = (basis[0] + basis[1])/2;
    basis[0] -= O;
    basis[1] -= O;
    p = p - O;
    
    float B = sqrt(pow(basis[1].x, 2) + pow(basis[1].y, 2));
    float co = basis[1].x / B;
    float si = basis[1].y / B;
    
    float u =  co * p.x + si * p.y;
    float v = -si * p.x + co * p.y;
    
    return Point2f(u, v)/B;
}

Mat hashing::pointsToMat2D(vector<Point2f> points) {
    int rows = 2;
    int cols = int(points.size());
    
    float table[rows][cols];
    for (int c = 0; c < cols; c++) {
        table[0][c] = points[c].x;
        table[1][c] = points[c].y;
    }
    return Mat(rows, cols, CV_32FC1, table).clone() ;
}

Mat hashing::pointsToHomog(Mat m) {
    Mat one = Mat::ones(1, m.cols, CV_32FC1);
    vconcat(m, one, m);
    
    return m.clone();
}
