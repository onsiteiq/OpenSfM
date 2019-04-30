import copy
import logging
import time
import os

from opensfm import dataset
from opensfm import exif


logger = logging.getLogger(__name__)
logging.getLogger("exifgpsupdate").setLevel(logging.WARNING)


class Command:
    name = 'update_gps_metadata'
    help = "Update extracted EXIF metadata from gps_list.txt"

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        start = time.time()
        data = dataset.DataSet(args.dataset)

        exif_files = data.get_exif_files()
        
        exif_img_names = map( lambda x: os.path.splitext(x)[0], exif_files )

        # Load GPS list if present ( these override any GPS read from EXIF metadata )
        gps_points = {}
        if data.gps_points_exist():
            gps_points = data.load_gps_points()
            logger.info( str( gps_points ) )

        for image in exif_img_names:
            if data.exif_exists(image):
                logging.info('Loading existing EXIF for {}'.format(image))
                d = data.load_exif(image)
                
                # Clear the gps field
                d.pop('gps', None)

                # Set from gps_list.txt only
                lla_fen_spl_comp = gps_points.get( image )
                if lla_fen_spl_comp is not None:
                    
                    gps_md = {}
                    d['gps'] = gps_md
                        
                    gps_md['latitude'] = lla_fen_spl_comp[0]
                    gps_md['longitude'] = lla_fen_spl_comp[1]
                    gps_md['altitude'] = lla_fen_spl_comp[2] 
                    # There is a question about what should be here. It may need to be set based on the floorplan
                    # resolution to correspond to a particular pixel neighborhood size. Or we should move toward
                    # physical coordinates by georeferencing our floorplans.
                    gps_md['dop'] = data.config.get( 'default_gps_dop', 45 )
                    
                    # Compass
                    if len(lla_fen_spl_comp) > 5:
                        d['compass'] = lla_fen_spl_comp[5]
                
                data.save_exif(image, d)

        end = time.time()
        with open(data.profile_log(), 'a') as fout:
            fout.write('update_gps_metadata: {0}\n'.format(end - start))

    
