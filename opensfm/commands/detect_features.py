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
    name = 'detect_features'
    help = 'Compute features for all images'

    def __init__(self):
        self.checkpoint_callback = None

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
    
        data = dataset.DataSet( args.dataset )
        images = data.images()

        # Group processing inputs into 150 images at a time.
        
        arg_groups = [ [] ]
        
        ag_ind = 0
        for ind,img in enumerate(images):
            cur_arg_group = arg_groups[ ag_ind ]
            if not data.feature_index_exists( img ):
                cur_arg_group.append( (img, data) )
            if len(cur_arg_group) == 150:
                arg_groups.append( [] )
                ag_ind += 1
        
        processes = data.config['feature_processes']

        run_time = 0

        for pargs in arg_groups:
            
            start = timer()
        
            parallel_map( detect, pargs, processes )
        
            end = timer()
        
            run_time += end - start
            
            self.write_report( data, run_time )
            
            if self.checkpoint_callback is not None:
                self.checkpoint_callback( args.dataset )
            
        with open(data.profile_log(), 'a') as fout:
                fout.write( 'detect_features: {0}\n'.format( run_time ) )

    def set_checkpoint_callback( self, callback ):
        self.checkpoint_callback = callback

    def write_report(self, data, wall_time):
        image_reports = []
        for image in data.images():
            try:
                txt = data.load_report('features/{}.json'.format(image))
                image_reports.append(io.json_loads(txt))
            except IOError:
                logger.warning('No feature report image {}'.format(image))

        report = {
            "wall_time": wall_time,
            "image_reports": image_reports
        }
        data.save_report(io.json_dumps(report), 'features.json')


def unfolded_cube_to_equi_normalized_image_coordinates( pixel_coords, image_camera_model ):
    
    bearings = image_camera_model.unfolded_pixel_bearings( pixel_coords )
    
    norm_pix_x, norm_pix_y = image_camera_model.project( ( bearings[:, 0], bearings[:, 1], bearings[:, 2] ) )
    
    norm_pixels = np.column_stack([norm_pix_x.ravel(), norm_pix_y.ravel()])
    
    return norm_pixels


# Temporary testing code #############

def denormalized_image_coordinates(norm_coords, width, height):
    size = max(width, height)
    p = np.empty((len(norm_coords), 2))
    p[:, 0] = norm_coords[:, 0] * size - 0.5 + width / 2.0
    p[:, 1] = norm_coords[:, 1] * size - 0.5 + height / 2.0
    return p
    

def normalized_image_coordinates(pixel_coords, width, height):
    size = max(width, height)
    p = np.empty((len(pixel_coords), 2))
    p[:, 0] = (pixel_coords[:, 0] + 0.5 - width / 2.0) / size
    p[:, 1] = (pixel_coords[:, 1] + 0.5 - height / 2.0) / size
    return p
    
    
def resized_image(image, config):
    """Resize image to feature_process_size."""
    max_size = config.get('feature_process_size', -1)
    h, w, _ = image.shape
    size = max(w, h)
    if 0 < max_size < size:
        dsize = w * max_size // size, h * max_size // size
        return cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_AREA)
    else:
        return image

######################################

def detect(args):
    log.setup()

    image, data = args
    logger.info('Extracting {} features for image {}'.format(
        data.feature_type().upper(), image))

    if not data.feature_index_exists(image):
        start = timer()

        exif = data.load_exif( image )
        camera_models = data.load_camera_models()
        image_camera_model = camera_models[ exif[ 'camera' ] ]

        if image_camera_model.projection_type in ['equirectangular', 'spherical'] and data.config['matching_unfolded_cube']:
            
            logger.info('Features unfolded cube.')

            # For spherical cameras create an undistorted image for the purposes of
            # feature finding (and later matching).
            
            max_size = data.config.get('feature_process_size', -1)
            if max_size == -1:
                max_size = img.shape[1]
            
            img = data.load_image( image )
            
            undist_tile_size = max_size//4
            
            undist_img = np.zeros( (max_size//2, max_size, 3 ), np.uint8 )
            undist_mask = np.full( (max_size//2, max_size, 1 ), 255, np.uint8 )
            
            undist_mask[ undist_tile_size:2*undist_tile_size, 2*undist_tile_size:3*undist_tile_size ] = 0
            undist_mask[ undist_tile_size:2*undist_tile_size, undist_tile_size:2*undist_tile_size ] = 0
            
            # The bottom mask to remove the influence of the camera person should be configurable. It depends on the forward
            # direction of the camera and where the camera person positions themselves in relation to this direction. It'save_feature_index
            # probably worth it to take care with this because the floor could help hold the reconstructions together.
            #undist_mask[ 5*undist_tile_size//4:7*undist_tile_size//4, undist_tile_size//3:undist_tile_size ] = 0
            #undist_mask[ 3*undist_tile_size//2:2*undist_tile_size, undist_tile_size//2:undist_tile_size ] = 0
            
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
                    #print( 'front')
                elif subshot_name == 'left':
                    undist_img[ :undist_tile_size, undist_tile_size:2*undist_tile_size ] = undistorted
                    #print( 'left')
                elif subshot_name == 'back':
                    undist_img[ :undist_tile_size, 2*undist_tile_size:3*undist_tile_size ] = undistorted
                    #print( 'back')
                elif subshot_name == 'right':
                    undist_img[ :undist_tile_size, 3*undist_tile_size:4*undist_tile_size ] = undistorted
                    #print( 'right')
                elif subshot_name == 'top':
                    undist_img[ undist_tile_size:2*undist_tile_size, 3*undist_tile_size:4*undist_tile_size ] = undistorted
                    #print( 'top')
                elif subshot_name == 'bottom':
                    undist_img[ undist_tile_size:2*undist_tile_size, :undist_tile_size ] = undistorted
                    #print( 'bottom')
                
                #data.save_undistorted_image(subshot.id, undistorted)
                
            #data.save_undistorted_image(subshot.id, undist_img)

            # We might consider combining a user supplied mask here as well
            
            p_unsorted, f_unsorted, c_unsorted = features.extract_features( undist_img, data.config, undist_mask )
            
            # Visualize the features on the unfolded cube
            # --------------------------------------------------------------
            
            if False:

                h_ud, w_ud, _ = undist_img.shape
                denorm_ud = denormalized_image_coordinates( p_unsorted[:, :2], w_ud, h_ud )
                
                print( p_unsorted.shape )
                print( denorm_ud.shape )

                rcolors = []
                
                for point in denorm_ud:
                    color = np.random.randint(0,255,(3)).tolist()
                    cv2.circle( undist_img, (int(point[0]),int(point[1])), 1, color, -1 )
                    rcolors.append( color )
                
                data.save_undistorted_image( image + '_unfolded_cube.jpg', undist_img)

            # --------------------------------------------------------------
            
            if len(p_unsorted) > 0:
            
                # Mask pixels that are out of valid image bounds before converting to equirectangular image coordinates

                bearings = image_camera_model.unfolded_pixel_bearings( p_unsorted[:, :2] )

                p_mask = np.array([ point is not None for point in bearings ])
                
                p_unsorted = p_unsorted[ p_mask ]
                f_unsorted = f_unsorted[ p_mask ]
                c_unsorted = c_unsorted[ p_mask ]
                
                p_unsorted[:, :2] = unfolded_cube_to_equi_normalized_image_coordinates( p_unsorted[:, :2], image_camera_model )
            
            # Visualize the same features converted back to equirectangular image coordinates
            # -----------------------------------------------------------------------------------------
            
            if False:

                timg = resized_image( img, data.config )
                
                h, w, _ = timg.shape
                
                denorm = denormalized_image_coordinates( p_unsorted[:, :2], w, h )
                
                for ind, point in enumerate( denorm ):
                    cv2.circle( timg, (int(point[0]),int(point[1])), 1, rcolors[ind], -1 )
                
                data.save_undistorted_image('original.jpg', timg)

            #------------------------------------------------------------------------------------------
        else:
            mask = data.load_combined_mask(image)
            if mask is not None:
                logger.info('Found mask to apply for image {}'.format(image))
            
            p_unsorted, f_unsorted, c_unsorted = features.extract_features(
                data.load_image(image), data.config, mask)
        
        if len(p_unsorted) == 0:
            return

        preemptive_max = data.config['preemptive_max']
        size = p_unsorted[:, 2]
        order = np.argsort(size)
        p_sorted = p_unsorted[order, :]
        f_sorted = f_unsorted[order, :]
        c_sorted = c_unsorted[order, :]
        p_pre = p_sorted[-preemptive_max:]
        f_pre = f_sorted[-preemptive_max:]
        data.save_features(image, p_sorted, f_sorted, c_sorted)
        data.save_preemptive_features(image, p_pre, f_pre)

        if data.config['matcher_type'] == 'FLANN':
            index = features.build_flann_index(f_sorted, data.config)
            data.save_feature_index(image, index)

        end = timer()
        report = {
            "image": image,
            "num_features": len(p_sorted),
            "wall_time": end - start,
        }
        data.save_report(io.json_dumps(report),
                         'features/{}.json'.format(image))
