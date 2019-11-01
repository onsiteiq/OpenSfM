//
//  hashing.hpp
//  Geometric hashing - code adapted from GeoHash by Daniel Mesham
//

#ifndef hashing_hpp
#define hashing_hpp

#include <opencv2/core/core.hpp>
#include <iostream>
#include <stdio.h>
#include <map>
#include <list>
#include <vector>


using namespace std;
using namespace cv;

// 2D hashing

//  A point location in 2d
class point {
public:
  point( float in_x, float in_y, int in_id = -1 ) : m_x(in_x), m_y(in_y), m_id(in_id) {}
  point() : m_x(0), m_y(0), m_id(-1) {}
  float x() const { return m_x; }
  float y() const { return m_y; }
  int getID() const { return m_id; }
private:
  float m_x, m_y;
  int m_id;
};


//  The index for a bin in a 2d grid
class bin_index {
public:
  bin_index( int in_i, int in_j ) : m_i(in_i), m_j(in_j) {}
  bin_index() : m_i(0), m_j(0) {}
  int i() const { return m_i; }
  int j() const { return m_j; }
private:
  int m_i, m_j;
};

class hash_2d {
public:
  //  Construct a geometric hash with square bins having the specified
  //  bin width.
  hash_2d( float bin_width=10.0 );

  //  Add a point to the geometric hash
  void add_point( point loc );

  //  Add a vector of points to the geometric hash
  void add_points( std::vector<point> const& locs );

  //  Find all points in the geometric hash that fall within the given
  //  circle.  Order them by increasing x and for ties, by increasing
  //  y
  std::vector<point> points_in_circle( point center, float radius ) const;

  //  Find all points in the geometric hash that fall within the given
  //  rectangle defined by the min_point (smallest x and y) and the
  //  max_point (greatest x and y).  Order the points found by
  //  increasing x and for ties, by increasing y
  std::vector<point> points_in_rectangle( point min_point, point max_point ) const;

  //  Erase the points that fall within the given circle
  int erase_points( point center, float radius=1e-6 );

  //  Find the bin index associated with a given point location
  bin_index point_to_bin( point loc ) const;

  //  Find the hash value of the given point location
  unsigned int hash_value( point loc ) const;

  //  What points are in the bin associated with the given point
  //  location.
  std::vector<point> points_in_bin( point loc ) const;

  //  How many points are in bins immediately next to target bin?
  int points_in_neighborhood(point loc);

  //  How many non-empty bins are there?
  int num_non_empty() const;

  //  How many points are in the geometric hash?
  int num_points() const;

  //  What is the size of the hash table?
  int table_size() const;

private:
  //  This is an internal record for an entry in the table.
  struct table_entry {
  public:
    bin_index bin;
    std::list<point> points;
  };

private:
  //  Iterator and cons iterator typedefs for the hash table
  typedef std::list<table_entry>::iterator table_entry_iterator;
  typedef std::list<table_entry>::const_iterator const_table_entry_iterator;

  //  Compute the hash value for the given bin index
  unsigned int hash_value( bin_index bin ) const;

  //  Find the table location and list iterator within the table for
  //  the given point.  Used when changes to the table are possible.
  bool find_entry_iterator( point                  loc,
                            int                  & table_index,
                            table_entry_iterator & itr);

  //  Find the table location and list iterator within the table for
  //  the given point.  Used when changes to the table are not
  //  possible.
  bool find_entry_iterator( point                        loc,
                            int                        & table_index,
                            const_table_entry_iterator & itr) const;

  //  Find the table location and list iterator within the table for
  //  the given bin.  Used when changes to the table are possible.
  bool find_entry_iterator( bin_index              bin,
                            int                  & table_index,
                            table_entry_iterator & itr);

  //  Find the table location and list iterator within the table for
  //  the given bin.  Used when changes to the table are not
  //  possible.
  bool find_entry_iterator( bin_index                    bin,
                            int                        & table_index,
                            const_table_entry_iterator & itr) const;


private:
  //  The table itself
  std::vector< std::list<table_entry> > m_table;

  //  Size of the square bins
  float m_width;

  //  Counters
  int   m_num_bin_entries, m_num_points;
};


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
