//
//  hlf.cc
//  match high level features detected in images to floor plan annotations
//

#include "hash_2d.h"
#include "hashing.hpp"

#include <opencv2/core/core.hpp>
#include <iostream>
#include <chrono>
#include <stdlib.h>     /* srand, rand */
#include <map>
#include <numeric>

namespace csfm {

using namespace std;
using namespace cv;

// set to 1 to merge results from all basis that passed detThresh;
// set to 0 to use result from the basis that received max votes
#define MERGE_RES 1

// nominal floor scale factor - parameters were tested with this
#define NOMINAL_SCALE_FACTOR 0.02

// actual floor scale factor
//#define ACTUAL_SCALE_FACTOR 0.02        // for Moinian/2_Wash, test data sets 1 & 2
//#define ACTUAL_SCALE_FACTOR 0.0399      // for Moinian/123_Linden, test data sets 3 & 4
float actual_scale_factor = 0.02;

// threshold for detection (other than 2 basis points). this threshold needs
// to be adjusted based on how many input image points we have, because the
// more we have, the more likely it is to have random erroneous result. it
// should be between 6-10
static float detThresh = 8.0;

// threshold for correspondance after applying estimated similarity transform
// to sfm coordinates. needs to adjust based on floor plan scale factor
static float distThresh = 100.0 / (actual_scale_factor/NOMINAL_SCALE_FACTOR);

// std dev of sfm error in pixels. needs to adjust based on floor plan scale factor
static float sigma = 50.0 / (actual_scale_factor/NOMINAL_SCALE_FACTOR);

// number of detections to use from all detections. set to -1 to use all
static int subsetSize = -1;

// inputs
vector<Point2f> planCoords;
vector<Point2f> sfmCoords;
vector<int> groundTruth;

// output
map<int, int> final_map;

void initRand() {
    auto now = chrono::system_clock::now();
    auto now_ms = chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto value = now_ms.time_since_epoch();
    unsigned int duration = (unsigned int)value.count();

    srand (duration);
}

map<vector<float>, int> getRandSubset(map<vector<float>, int> all) {
    if(subsetSize == -1)
        return all;

    map<vector<float>, int> retVal;

    initRand();

    vector<vector<float>> allKeys;
    for (auto const& element : all)
        allKeys.push_back(element.first);

    while(retVal.size() < min(subsetSize, (int)allKeys.size())) {
        vector<float> v = allKeys[rand()%allKeys.size()];

        if(retVal.find(v) == retVal.end()) {
            retVal[v] = all[v];
        }
    }
    return retVal;
}

// returns mean and stddev of distances
vector<float> getModelStat(vector<Point2f> coords)
{
    vector<float> distances;
    for (int i = 0; i < coords.size(); i++) {
        for (int j = i+1; j < coords.size(); j++) {
            distances.push_back(norm(coords[i]-coords[j]));
        }
    }
    float mean = accumulate(distances.begin(), distances.end(), 0.0)/distances.size();
    float sq_sum = inner_product(distances.begin(), distances.end(), distances.begin(), 0.0);
    float std_dev = sqrt(sq_sum / distances.size() - mean * mean);

    sort(distances.begin(), distances.end());
    float median = distances[distances.size()/2];

    return {mean, std_dev, median};
}

int runAlgo() {
    int retVal = 0;

    // * * * * * * * * * * * * * *
    //      BUILDING MODEL
    // * * * * * * * * * * * * * *
    auto startHash = chrono::system_clock::now(); // Start hashing timer
    vector<HashTable> tables;

    vector<float> stat = getModelStat(planCoords);
    float thresh = 0; //stat[2];//stat[0] + 0.0*stat[1];
    //cout << "mean = " << stat[0] << ", stddev = " << stat[1] << ", median = " << stat[2] << endl;

    for (int i = 0; i < planCoords.size(); i++) {
        for (int j = i+1; j < planCoords.size(); j++) {
            // basis points need to be farther apart to be more reliable
            if(norm(planCoords[i]-planCoords[j]) < thresh) continue;

            vector<int> basisIndex {i, j};
            tables.push_back(hashing::createTable(basisIndex, planCoords, sigma));
        }
    }
    auto endHash = chrono::system_clock::now();

    // * * * * * * * * * * * * * *
    //        REGISTRATION
    // * * * * * * * * * * * * * *

    auto startReg = chrono::system_clock::now();

    vector<vector<int>> valid_basis_array;
    vector<HashTable> valid_table_array;
    
    for (int i = 0; i < sfmCoords.size(); i++) {
        for (int j = i+1; j < sfmCoords.size(); j++) {
            vector<int> sfmBasis {i, j};
            vector<HashTable> votedTables = hashing::voteForTables(tables, sfmCoords, sfmBasis);

            if (votedTables[0].votes >= detThresh) {
                valid_basis_array.push_back(sfmBasis);
                valid_table_array.push_back(votedTables[0]);
            }
        }
    }
    
    if(valid_table_array.size() == 0) {
        return 0;
    }
        
    /*
     * in the following, we try to merge the results. error rate may be higher
     */
    vector<CorrInfo> corrinfo_array;
    for(int i = 0; i < valid_table_array.size(); i++) {
        cout << "model basis = " << valid_table_array[i].basis[0] << " " << valid_table_array[i].basis[1] << ", VOTES = " << valid_table_array[i].votes << endl;

        // estimated scale (floor plan / sfm). this needs to come from pdr predictions.
        float estScale = 1.0/(actual_scale_factor*0.3048);

        // perform geometric verification
        float scale, rot;
        map<int,int> corr_map = hashing::getMatchedPoints(valid_basis_array[i], valid_table_array[i], planCoords, sfmCoords, scale, rot, distThresh, estScale);

        // debugging
        string debug_str = "model points identified = ";
        for(auto& c: corr_map) {
            debug_str += to_string(c.second) + " ";
        }
        cout << debug_str << endl;
        
        corrinfo_array.push_back(CorrInfo(corr_map, scale, rot, valid_table_array[i].votes));
    }
    
    // sort correspondance maps by votes
    sort(corrinfo_array.begin(), corrinfo_array.end(), greater<CorrInfo>());
    
    // merge all correspondances into a final map.
    final_map = corrinfo_array[0].corr_map;

#if MERGE_RES
    cout << "merging model points from different basis" << endl;
    
    float final_scale = corrinfo_array[0].scale;
    float final_rot = corrinfo_array[0].rot;
    
    for(int i=1; i<corrinfo_array.size(); i++) {
        CorrInfo corrinfo = corrinfo_array[i];
        map<int, int> new_map = corrinfo.corr_map;
        float new_scale = corrinfo.scale;
        float new_rot = corrinfo.rot;
        
        int num_common_points = 0;
        int num_conflicts = 0;
        for(auto& c: new_map) {
            if(final_map.find(c.first) != final_map.end()) {
                if(final_map[c.first] == c.second) {
                    num_common_points++;
                } else {
                    num_conflicts++;
                }
            }
        }
        
        // if there's no conflict with existing accumulated map, and if rotation
        // and scale are fairly consistent, then add this to accumulated map
        if(num_conflicts == 0) {
            float rot_diff = fabs(new_rot - final_rot);
            float scale_ratio = new_scale/final_scale;
            if((rot_diff < 5.0) && (scale_ratio < 1.1) && (scale_ratio > 0.9)) {
                final_map.insert(new_map.begin(), new_map.end());
            }
        }
    }
#endif

    if(final_map.size() == 0) {
        retVal = 0;
    } else {
        retVal = (int) final_map.size();
        
        // debugging
        string debug_str = "final model points identified = ";
        int err_cnt = 0;
        for(auto& c: final_map) {
            debug_str += to_string(c.second) + " ";

            int gt = groundTruth[c.first];
            if(gt != -1 && gt != c.second) {
                float doorDistance = cv::norm(cv::Mat(planCoords[gt]),cv::Mat(planCoords[c.second]));
                
                cout << gt << " mistaken as " << c.second << ", door distance = " << doorDistance <<
                    ((doorDistance <= 10.0/actual_scale_factor)?", permissable":", failure") << endl;
                
                // if we mistake for a door that's more than 10 feet from the true door, then a failure
                if(doorDistance > 10.0/actual_scale_factor) {
                    err_cnt++;
                    retVal = -1;
                }
            }
        }
        cout << debug_str << "err_cnt = " << err_cnt << endl;
        
        if(err_cnt != 0) retVal = -1;
    }

    auto endReg = chrono::system_clock::now();
    
    chrono::duration<double> timeHash = endHash-startHash;
    cout << "Hashing time     = " << timeHash.count()*1000.0 << " ms" << endl;
    chrono::duration<double> timeReg = endReg-startReg;
    cout << "Registration time = " << timeReg.count()*1000.0 << " ms" << endl;
        
    return retVal;
}

py::dict RunHlfMatcher(const py::list &hlf_list, const py::list &det_list,
    const py::list &ground_truth, float scale_factor) {
    for (int i = 0; i < py::len(hlf_list); ++i) {
        pyarray_f hlf_array = hlf_list[i].cast<pyarray_f>();
        const float *h = hlf_array.data();

        planCoords.push_back(Point2f(h[0], h[1]));
    }

    for (int i = 0; i < py::len(det_list); ++i) {
        pyarray_f det_array = det_list[i].cast<pyarray_f>();
        const float *d = det_array.data();

        sfmCoords.push_back(Point2f(d[0], d[1]));
        groundTruth.push_back(ground_truth[i].cast<int>());
    }

    actual_scale_factor = scale_factor;

    runAlgo();

#if 0
	py::dict retVal;
    map<int, int>::iterator iter;
    for (iter = final_map.begin(); iter != final_map.end(); ++iter) {
		retVal[to_string(iter->first).c_str()] = to_string(iter->second);
	}
    return retVal;
#endif
    return py::cast(final_map);
}

int main(int argc, const char * argv[]) {
    int numRuns = 1;
    int errCnt = 0;
    int noResCnt = 0;
    int numCorr = 0;
    for(int i=0; i<numRuns; i++) {
        int res = runAlgo();
        if(res == 0) {
            noResCnt++;
            cout << "no result" << endl << endl;
        } else if(res == -1) {
            errCnt++;
            cout << "invalid result" << endl << endl;
        } else {
            numCorr+=res;
            cout << "valid result" << endl << endl;
        }
    }
    cout << "numRuns = " << numRuns << ", no result = " << noResCnt << ", error = " << errCnt << endl;
    cout << "Detection = " << (float)(numRuns - noResCnt) * 100.0 / (float)numRuns << "%" << endl;
    cout << "Accuracy  = " << (float)(numRuns - errCnt - noResCnt)*100.0 / (float)(numRuns - noResCnt) << "%" << endl;
    cout << "Avg correspondances  = " << (float)numCorr / (float)(numRuns - noResCnt - errCnt) << endl;
    
    return 0;
}

}
