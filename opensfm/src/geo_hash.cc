//
//  geo_hash.cc
//  Geometric hashing - code adapted from GeoHash by Daniel Mesham
//

#include <iostream>
#include <algorithm>
#include <cmath>
#include <map>
#include <opencv2/calib3d.hpp>
#include "geo_hash.h"

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

// 2D hashing
bool operator< ( point const& left, point const& right )
{
    return left.x() < right.x() || ( left.x() == right.x() && left.y() < right.y() );
}

bool operator== ( bin_index const& left, bin_index const& right )
{
    return left.i() == right.i() && left.j() == right.j();
}


hash_2d::hash_2d( float bin_width )
  : m_table(4), m_width(bin_width), m_num_bin_entries(0), m_num_points(0)
{}


void
hash_2d::add_point( point loc )
{
    int            index;
    table_entry_iterator itr;

    //  Find the bin.
    if ( this->find_entry_iterator( loc, index, itr ) )
    {
        // If it is already there, just add the point
        itr->points.push_back(loc);
    }
    else
    {
        // Need to create a new entry in the table
        table_entry entry;
        entry.bin = this->point_to_bin( loc );
        entry.points.push_back( loc );
        m_table[index].push_back( entry );
        m_num_bin_entries ++ ;

        //  Resize the table right here if needed
        const float resize_multiplier = 1.5;
        if ( m_num_bin_entries > resize_multiplier * m_table.size() )
        {
            std::vector< std::list<table_entry> > new_table( 2*m_table.size() + 1 );
            for ( unsigned int i = 0; i<m_table.size(); ++i )
                for ( table_entry_iterator p = m_table[i].begin();
                     p != m_table[i].end(); ++p )
                {
                    unsigned k = hash_value( p->bin ) % new_table.size();
                    new_table[k].push_back( *p );
                }
            m_table = new_table;
        }
    }
    m_num_points ++ ;
}


void
hash_2d::add_points( std::vector<point> const& locs )
{
    //  Just repeated apply add_point for an individual point.
    for ( unsigned int i=0; i<locs.size(); ++i )
        this->add_point( locs[i] );
}

//
inline float square( float a ) { return a*a; }


std::vector<point>
hash_2d::points_in_circle( point center, float radius ) const
{
    // Establish bounds on the bins that could intersect the circle.
    bin_index lower = point_to_bin( point( center.x()-radius, center.y()-radius) );
    bin_index upper = point_to_bin( point( center.x()+radius, center.y()+radius) );

    int   index;
    const_table_entry_iterator itr;
    std::vector<point> points_found;  // record the points found

    //  Check each bin falling within the bounds.
    for ( int i = lower.i(); i<=upper.i(); ++i )
        for ( int j = lower.j(); j<=upper.j(); ++j )
            //  Is the bin occupied?
            if ( find_entry_iterator( bin_index(i,j), index, itr ) )
            {
                //  Check each point
                for ( std::list<point>::const_iterator p = itr->points.begin();
                     p != itr->points.end(); ++p )
                {
                    //  If inside the circle, save the point
                    if ( square(p->x() - center.x()) + square(p->y() - center.y())
                        <= square(radius) )
                        points_found.push_back( *p );
                }
            }

    std::sort( points_found.begin(), points_found.end() );
    return points_found;
}


std::vector<point>
hash_2d::points_in_rectangle( point min_point, point max_point ) const
{
    //  Establish bounds on the bins
    bin_index lower = point_to_bin( min_point );
    bin_index upper = point_to_bin( max_point );

    int   index;
    const_table_entry_iterator itr;
    std::vector<point> points_found;

    //  Check each bin
    for ( int i = lower.i(); i<=upper.i(); ++i )
        for ( int j = lower.j(); j<=upper.j(); ++j )
            //  Is the bin occupied?
            if ( find_entry_iterator( bin_index(i,j), index, itr ) )
            {
                // Check each point
                for ( std::list<point>::const_iterator p = itr->points.begin();
                     p != itr->points.end(); ++p )
                {
                    //  If it is actually inside the rectangle then save it
                    if ( min_point.x() <= p->x() && p->x() <= max_point.x() &&
                        min_point.y() <= p->y() && p->y() <= max_point.y() )
                        points_found.push_back( *p );
                }
            }

    std::sort( points_found.begin(), points_found.end() );
    return points_found;
}

int
hash_2d::erase_points( point center, float radius )
{
    // Find the search range of bins
    bin_index lower = point_to_bin( point( center.x()-radius, center.y()-radius) );
    bin_index upper = point_to_bin( point( center.x()+radius, center.y()+radius) );

    int   index;
    table_entry_iterator itr;

    int num_erased = 0;    // keep track of the number of points erased

    //  For each bin
    for ( int i = lower.i(); i<=upper.i(); ++i )
        for ( int j = lower.j(); j<=upper.j(); ++j )
            //  If the bin is non-empty
            if ( find_entry_iterator( bin_index(i,j), index, itr ) )
            {
                // For  each point in the bin
                for ( std::list<point>::iterator p = itr->points.begin();
                     p != itr->points.end(); )
                {
                    //  If the point is withn the radius
                    if ( square(p->x() - center.x()) + square(p->y() - center.y())
                        <= square(radius) )
                    {
                        //  Erase it
                        p = itr->points.erase(p);
                        m_num_points -- ;
                        num_erased ++ ;
                    }
                    else
                        ++p;
                }
                //  Remove the bin if it is empty
                if ( itr->points.empty() )
                {
                    m_table[index].erase( itr );
                    m_num_bin_entries -- ;
                }
            }

    return num_erased;
}


//  Point location to bin index
bin_index
hash_2d::point_to_bin( point loc ) const
{
    int i = int( floor(loc.x() / m_width) );
    int j = int( floor(loc.y() / m_width) );
    return bin_index(i,j);
}


unsigned int
hash_2d::hash_value( bin_index bin ) const
{
    return std::abs( bin.i() * 378551 + bin.j() * 63689 );
}


unsigned int
hash_2d::hash_value( point loc ) const
{
    return hash_value( point_to_bin(loc) );
}


//  Find all points in the bin associated with point loc
std::vector<point>
hash_2d::points_in_bin( point loc ) const
{
    std::vector<point> points_found;
    int   index;
    const_table_entry_iterator itr;

    if ( find_entry_iterator( loc, index, itr ) )
    {
        points_found.resize( itr->points.size() );
        std::copy( itr->points.begin(), itr->points.end(), points_found.begin() );
        std::sort( points_found.begin(), points_found.end() );
    }
    return points_found;
}

int
hash_2d::points_in_neighborhood(point loc)
{
    int retVal = 0;
    bin_index bin = this->point_to_bin( loc );

    for ( int i = -1; i<=1; ++i ) {
        for ( int j = -1; j<=1; ++j ) {
            if ((i==0) && (j==0)) continue;
            bin_index new_bin = bin_index(bin.i()+i, bin.j()+j);

            int   index;
            const_table_entry_iterator itr;

            if(find_entry_iterator(new_bin, index, itr)) {
                retVal++;
            }
        }
    }

    return retVal;
}


int
hash_2d::num_non_empty() const
{
    return m_num_bin_entries;
}

int
hash_2d::num_points() const
{
    return m_num_points;
}

int
hash_2d::table_size() const
{
    return int( m_table.size() );
}


bool
hash_2d::find_entry_iterator( point                    loc,
                               int                    & table_index,
                               table_entry_iterator   & itr)
{
    bin_index bin = this->point_to_bin( loc );
    return find_entry_iterator( bin, table_index, itr );
}

bool
hash_2d::find_entry_iterator( point                        loc,
                               int                        & table_index,
                               const_table_entry_iterator & itr) const
{
    bin_index bin = this->point_to_bin( loc );
    return find_entry_iterator( bin, table_index, itr );
}


bool
hash_2d::find_entry_iterator( bin_index              bin,
                               int                  & table_index,
                               table_entry_iterator & itr)
{
    table_index = this->hash_value( bin ) % this->table_size();
    for ( itr = m_table[table_index].begin();
        itr != m_table[table_index].end() && ! (itr->bin == bin); ++itr )
        ;
    return ( itr != m_table[table_index].end() );
}


bool
hash_2d::find_entry_iterator( bin_index                    bin,
                               int                        & table_index,
                               const_table_entry_iterator & itr) const
{
    table_index = this->hash_value( bin ) % this->table_size();
    for ( itr = m_table[table_index].begin();
        itr != m_table[table_index].end() && ! (itr->bin == bin); ++itr )
        ;
    return ( itr != m_table[table_index].end() );
}
