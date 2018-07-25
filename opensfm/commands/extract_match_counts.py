import logging
import time

from opensfm import dataset
from opensfm import matching

logger = logging.getLogger(__name__)


class Command:
    name = 'extract_match_counts'
    help = "Extract pairwise match counts"

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')
        parser.add_argument('--not-redundant',
                            help='do not store redundant match counts',
                            action='store_true')

    def run(self, args):
        start = time.time()
        data = dataset.DataSet(args.dataset)
        images = data.images()

        # Read matches and extract match counts. We store the data
        # with some redundancy just for the convenience.
        matches = {}
        for im1 in images:
            matches[im1] = {}
            
            if ( args.not_redundant ):
            
                # Alternative where we don't store redundant information
                try:
                    im1_matches = data.load_matches(im1)
                except IOError:
                    continue
                for im2 in im1_matches:
                    matches1to2 = data.find_matches(im1,im2)
                    matches[im1][im2] = len(matches1to2)  
            else:
            
                for im2 in images:
                    matches1to2 = data.find_matches(im1,im2)
                    
                    if len(matches1to2) > 0:
                        matches[im1][im2] = len(matches1to2)
                    
        data.save_match_counts( matches )
        
        end = time.time()
        with open(data.profile_log(), 'a') as fout:
            fout.write('create_tracks: {0}\n'.format(end - start))
