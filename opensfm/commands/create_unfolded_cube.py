import logging
from timeit import default_timer as timer
import os
import cv2
import numpy as np

from opensfm import dataset
from opensfm import features
from opensfm import io
from opensfm import log
from opensfm import types
from opensfm.commands import undistort
from opensfm.context import parallel_map

logger = logging.getLogger(__name__)


class Command:
    name = 'create_unfolded_cube'
    help = 'Compute unfolded cube images from an equirectangular panorama'

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
    
        data = dataset.DataSet(args.dataset)
        images = data.images()

        arguments = [(image, data) for image in images]

        start = timer()
        
        processes = data.config['processes']
        parallel_map( equi_to_unfolded_cube, arguments, processes )
        
        end = timer()
        
        with open(data.profile_log(), 'a') as fout:
            fout.write('create_unfolded_cube: {0}\n'.format(end - start))

        self.write_report( data, end - start )


    def write_report(self, data, wall_time):
    
        image_reports = []
        for image in data.images():
            try:
                txt = data.load_report( 'unfolding/{}.json'.format( image ) )
                image_reports.append( io.json_loads( txt ) )
            except IOError:
                logger.warning( 'No unfolding report image {}'.format(image) )

        report = {
            "wall_time": wall_time,
            "image_reports": image_reports
        }
        
        data.save_report( io.json_dumps(report), 'unfolding.json' )


def equi_to_unfolded_cube(args):

    log.setup()

    image, data = args
    logger.info('Extracting unfolded cube for image {}'.format( image ) )

    start = timer()

    exif = data.load_exif( image )
    camera_models = data.load_camera_models()
    image_camera_model = camera_models[ exif[ 'camera' ] ]

    if image_camera_model.projection_type in ['equirectangular', 'spherical']:
        
        logger.info('Equirectangular to unfolded cube.')

        # For spherical cameras create an undistorted image (feature finding and/or AI purposes)
        
        max_size = data.config.get( 'ai_process_size', 4096 )
        if max_size == -1:
            max_size = img.shape[1]
        
        img = data.load_image( image )
        
        undist_tile_size = max_size//4
        
        undist_img = np.zeros( (max_size//2, max_size, 3 ), np.uint8 )
        
        spherical_shot = types.Shot()
        spherical_shot.pose = types.Pose()
        spherical_shot.id = image
        spherical_shot.camera = image_camera_model
        
        perspective_shots = undistort.perspective_views_of_a_panorama( spherical_shot, undist_tile_size )
        
        for subshot in perspective_shots:
            
            undistorted = undistort.render_perspective_view_of_a_panorama( img, spherical_shot, subshot )
            
            subshot_id_prefix = '{}_perspective_view_'.format( spherical_shot.id )
            
            subshot_name = subshot.id[ len(subshot_id_prefix): ] if subshot.id.startswith( subshot_id_prefix ) else subshot.id
            ( subshot_name, ext ) = os.path.splitext( subshot_name )
            
            if subshot_name == 'front':
                undist_img[ :undist_tile_size, :undist_tile_size ] = undistorted
            elif subshot_name == 'left':
                undist_img[ :undist_tile_size, undist_tile_size:2*undist_tile_size ] = undistorted
            elif subshot_name == 'back':
                undist_img[ :undist_tile_size, 2*undist_tile_size:3*undist_tile_size ] = undistorted
            elif subshot_name == 'right':
                undist_img[ :undist_tile_size, 3*undist_tile_size:4*undist_tile_size ] = undistorted
            elif subshot_name == 'top':
                undist_img[ undist_tile_size:2*undist_tile_size, 3*undist_tile_size:4*undist_tile_size ] = undistorted
            elif subshot_name == 'bottom':
                undist_img[ undist_tile_size:2*undist_tile_size, :undist_tile_size ] = undistorted
            
        data.save_undistorted_image( image + '_unfolded_cube.jpg', undist_img)

    else:
        logger.info('Could not create unfolded cube for image {}. Type is not equirectangular'.format( image ) )

    end = timer()
    
    report = {
        "image": image,
        "wall_time": end - start,
    }
    
    data.save_report(io.json_dumps(report),
                     'unfolding/{}.json'.format(image))
