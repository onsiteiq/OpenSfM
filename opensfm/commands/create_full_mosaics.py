import logging
from timeit import default_timer as timer
import os
import cv2
import math
import math_utils
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
    name = 'create_full_mosaics'
    help = 'Compute full mosaics if the camera type is not equirectangular/spherical. These are used in PVR visualization.'

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
    
        data = dataset.DataSet(args.dataset)
        reconstructions = data.load_reconstruction()

        if not reconstructions:
            logger.error( 'No reconstruction could be found.' )
            return

        arguments = []
        
        # We assume the same camera throughout the reconstruction at the moment
        # so we use the first camera of the reconstruction. Otherwise, we could
        # undistort the reconstruction and use the cameras saved in that process.
        # In which case each shot has it's own camera. See the 'undistort' command.
        
        shot_one = list(reconstructions[0].shots.values())[0]
        
        calculate_new_camera_matrix( shot_one, data )
        
        # We can also optimize due to the above assumption by reusing our maps.
        
        ( r_map_x, r_map_y, dst_mask_x, dst_mask_y ) = create_full_mosaic( ( shot_one, data ) )
        
        for reconstruction in reconstructions:
            for shot in list(reconstruction.shots.values())[1:]:
                arguments.append( ( shot, r_map_x, r_map_y, dst_mask_x, dst_mask_y, data ) )
        
        start = timer()
        
        processes = data.config['processes']
        parallel_map( create_full_mosaic_precom_maps, arguments, processes )
        
        end = timer()
        
        with open(data.profile_log(), 'a') as fout:
            fout.write('create_full_mosaics: {0}\n'.format(end - start))

        self.write_report( data, reconstructions, end - start )


    def write_report(self, data, reconstructions, wall_time):
    
        image_reports = []
        for reconstruction in reconstructions:
            for shot in reconstruction.shots.values():
                try:
                    txt = data.load_report( 'full_mosaic_reprojection/{}.json'.format( shot.id ) )
                    image_reports.append( io.json_loads( txt ) )
                except IOError:
                    logger.warning( 'No full mosaic report image {}'.format( shot.id ) )

        report = {
            "wall_time": wall_time,
            "image_reports": image_reports
        }
        
        data.save_report( io.json_dumps(report), 'full_mosaic_reprojection.json' )


def calculate_new_camera_matrix( shot, data ):

    projection_type = shot.camera.projection_type

    if projection_type == 'perspective':
        
        camera = shot.camera
                
    elif projection_type == 'brown':
                
        camera = undistort.perspective_camera_from_brown( shot.camera )
                
    elif projection_type == 'fisheye':
                
        camera = undistort.perspective_camera_from_fisheye( shot.camera )

    K_pix = camera.get_K_in_pixel_coordinates()

    matrix_file = os.path.join( data.data_path, 'full_mosaic_camera.xml' )

    file_storage = cv2.FileStorage( matrix_file, cv2.FILE_STORAGE_WRITE )
    
    file_storage.write( name = 'camera_matrix', val = K_pix )
    file_storage.release()


def initialize_mosaic_image( width, height, seed_img ):

    seed_h, seed_w = seed_img.shape[:2]

    small_img = cv2.resize( seed_img, (int(seed_w/512), int(seed_h/512)) , interpolation = cv2.INTER_AREA )

    mosaic_img = cv2.resize( small_img, ( width, height ), interpolation = cv2.INTER_CUBIC ) #cv2.INTER_LANCZOS4 INTER_LINEAR )#np.zeros( ( height, width, 3 ), dtype = np.uint8 )
            
    line_width = 3 #int(0.0005*height)
    
        
    y_labels = [ 'TOP ', '', 'BOTTOM ' ]
    x_labels = [ 'BACK', 'LEFT', 'FRONT', 'RIGHT', 'BACK' ]
    
    center_y = int( height/2 )
    center_x = int( width/2 )
    
    arrow_dir_offset = int( height/20 )
    arrow_x_offset = 45
    arrow_y_offset = -45
        
    for ind_y, y_label in enumerate(y_labels):
        for ind_x, x_label in enumerate(x_labels):
        
            y_pix = int(height/4) + ind_y*int(height/4)
            x_pix = ind_x*int(width/4)
            
            color_sample = mosaic_img[ math_utils.clamp( y_pix+line_width, 0, height-1), math_utils.clamp( x_pix+line_width, 0, width-1 ) ]
            
            opp_color = [255,255,255] - color_sample
            
            text_color = tuple( [ int(x) for x in opp_color ] )
            
            cv2.putText( mosaic_img, y_label + x_label,  (x_pix, y_pix), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=text_color, thickness=2, lineType=cv2.LINE_AA ) 
            
            # Guide arrows
            
            arrow_vec = np.array( [ center_x, center_y ], dtype = float ) - np.array( [ x_pix, y_pix ], dtype = float )
            
            arrow_vec_norm = np.linalg.norm( arrow_vec )
            
            if arrow_vec_norm > 0.0:
            
	            arrow_vec = arrow_vec/np.linalg.norm( arrow_vec )
	            
	            arrow_start = np.array( [x_pix, y_pix] ) + arrow_dir_offset*arrow_vec + np.array( [arrow_x_offset, arrow_y_offset] )
	            arrow_end = arrow_start + 0.5*arrow_dir_offset*arrow_vec
                
	            cv2.arrowedLine( mosaic_img, tuple(arrow_start.astype(np.int32)), tuple(arrow_end.astype(np.int32)), text_color, line_width ) #, tipLength = 0.2 )
	
    # Draw latitude and longitude lines to help users keep orientation
    
    yinc = int(height/4)
    
    for i in range(1,4):
    
        line_mult = 1
        if i == 2:
            line_mult = 2
    
        cv2.line(mosaic_img, (0,i*yinc), (width, i*yinc), (130, 171, 176), line_mult*line_width )
    
    xinc = int(width/8)
    
    for i in range(0,9):
        cv2.line(mosaic_img, (i*xinc,0), (i*xinc, height), (130, 171, 176), line_width )

    
    return mosaic_img


def create_full_mosaic_precom_maps( args ):

    log.setup()

    shot, r_map_x, r_map_y, dst_mask_x, dst_mask_y, data = args
    logger.info('Creating full mosaic for image {}'.format( shot.id ) )

    config = data.config

    start = timer()

    projection_type = shot.camera.projection_type

    if projection_type in ['perspective', 'brown', 'fisheye']:

        img = data.load_image( shot.id )
        
        camera = types.SphericalCamera()
        camera.id = "Spherical Projection Camera"
        
        # Determine the correct mosaic size from the focal length of the camera
        # Limit this to a maximum of a 16K image which is the highest resolution
        # currently supported by PVR.
        
        K_pix = shot.camera.get_K_in_pixel_coordinates()
        
        camera.height = int( np.clip( math.pi*K_pix[0,0], 0, 8192 ) )
        camera.width = int( np.clip( 2*math.pi*K_pix[0,0], 0, 16384 ) )
        
        interp_mode = data.config.get( 'full_mosaic_proj_interpolation', 'linear' )
        
        if interp_mode == 'linear':
        
            mosaic_img = initialize_mosaic_image( camera.width, camera.height, img )
            
            # Sample source imagery colors
            
            colors = cv2.remap( img, r_map_x, r_map_y, cv2.INTER_LINEAR , borderMode=cv2.BORDER_CONSTANT )
            
            mosaic_img[ dst_mask_y, dst_mask_x ] = colors
            
            blend_projection_border( mosaic_img, dst_mask_y, dst_mask_x )

        elif interp_mode == 'nearest':
            
            mosaic_img = cv2.remap( img, r_map_x, r_map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT )
        
        else:
            raise NotImplementedError( 'Interpolation type not supported: {}'.format( interp_mode ) )
            
    data.save_full_mosaic_image( os.path.splitext( shot.id )[0], mosaic_img )

    end = timer()
    
    report = {
        "image": shot.id,
        "wall_time": end - start,
    }
    
    data.save_report( io.json_dumps(report),
                      'full_mosaic_reprojection/{}.json'.format( shot.id ) )


def blend_projection_border( mosaic_img, dst_mask_y, dst_mask_x ):
            
    # Initialize blurring and alpha mask kernels
            
    half_chunk_size = 75
    border = 41
    
    half_size = half_chunk_size + border
    
    kernel_1d = cv2.getGaussianKernel( 2*half_chunk_size+1,  1.5*(0.3*((2*half_chunk_size+1-1)*0.5 - 1) + 0.8) , cv2.CV_32F )
    kernel_1d/=kernel_1d[ half_chunk_size ]
    
    half_kernel_1d = kernel_1d[ half_chunk_size : 2*half_chunk_size ]
    
    alpha = np.zeros( ( 2*half_chunk_size, 2*half_chunk_size, 3 ), dtype = np.float32 ) #np.float32 uint8)
    
    for y in range(0,2*half_chunk_size):
        for x in range(0,2*half_chunk_size):
            yt = y - half_chunk_size
            xt = x - half_chunk_size
            
            r = int( math.sqrt( yt*yt + xt*xt ) )
            
            if r > half_chunk_size-1:
                r = half_chunk_size-1
            
            kv = half_kernel_1d[r]
            
            alpha[ y, x, 0] = alpha[ y, x, 1] = alpha[ y, x, 2] = kv
    
    
    # Grab the indices of pixels along the projected image border and blend into the
    # background with a gaussian blur and alpha map.
    
    dst_mask_y_border = np.concatenate( [ dst_mask_y[ 0:,0 ], 
                                          dst_mask_y[ 0:, -1 ],
                                          dst_mask_y[ 0, 0: ],
                                          dst_mask_y[-1, 0: ] ] )
                    
    dst_mask_x_border = np.concatenate( [ dst_mask_x[ 0:,0 ], 
                                          dst_mask_x[ 0:, -1 ],
                                          dst_mask_x[ 0, 0: ],
                                          dst_mask_x[-1, 0: ] ] )
    
    dst_mask_border = np.column_stack( [ dst_mask_y_border, dst_mask_x_border ] )
    
    for border_pix in dst_mask_border[::75]:
        
        border_y = border_pix[0] #dst_mask_y[y_ind,0]
        border_x = border_pix[1] #dst_mask_x[y_ind,0]
    
        sub_img = mosaic_img[ border_y - half_size : border_y + half_size, border_x - half_size : border_x + half_size ].copy()
    
        sub_rng = border + 2*half_chunk_size
    
        sub_img[border:sub_rng,border:sub_rng] = cv2.GaussianBlur( sub_img[border:sub_rng,border:sub_rng], (81,81), 0 )
    
        mosaic_img[ border_y - half_chunk_size : border_y + half_chunk_size, border_x - half_chunk_size : border_x + half_chunk_size ] = \
            np.multiply( sub_img[border:sub_rng,border:sub_rng].astype( np.float32 ), alpha ) + \
            np.multiply( mosaic_img[ border_y - half_chunk_size : border_y + half_chunk_size, border_x - half_chunk_size : border_x + half_chunk_size ].astype( np.float32 ), 1 - alpha )

       
def create_full_mosaic(args):

    log.setup()

    shot, data = args
    logger.info('Creating full mosaic for image {}'.format( shot.id ) )

    config = data.config

    start = timer()

    projection_type = shot.camera.projection_type
    
    r_map_x = None
    r_map_y = None
    dst_mask_x = None
    dst_mask_y = None

    if projection_type in ['perspective', 'brown', 'fisheye']:

        img = data.load_image( shot.id )
        
        camera = types.SphericalCamera()
        camera.id = "Spherical Projection Camera"
        
        # Determine the correct mosaic size from the focal length of the camera
        # Limit this to a maximum of a 16K image which is the highest resolution
        # currently supported by PVR.
        
        K_pix = shot.camera.get_K_in_pixel_coordinates()
        
        camera.height = int( np.clip( math.pi*K_pix[0,0], 0, 8192 ) )
        camera.width = int( np.clip( 2*math.pi*K_pix[0,0], 0, 16384 ) )
        
        shot_cam = shot.camera
    
        # Project shot's pixels to the spherical mosaic image
    
        src_shape = ( shot_cam.height, shot_cam.width )
        src_y, src_x = np.indices( src_shape ).astype( np.float32 )
        
        src_pixels_denormalized = np.column_stack( [ src_x.ravel(), src_y.ravel() ] )

        src_pixels = features.normalized_image_coordinates( src_pixels_denormalized, shot_cam.width, shot_cam.height )

        # Convert to bearings
        
        src_bearings = shot_cam.pixel_bearing_many( src_pixels )
    
        # Project to spherical mosaic pixels
        
        dst_x, dst_y = camera.project( ( src_bearings[:, 0],
                                         src_bearings[:, 1],
                                         src_bearings[:, 2] ) )
                                            
        dst_pixels = np.column_stack( [ dst_x.ravel(), dst_y.ravel() ] )
        
        interp_mode = data.config.get( 'full_mosaic_proj_interpolation', 'linear' )
        
        if interp_mode == 'linear':
            
            # Snap to pixel centers to generate a projection index mask. This will be slower then finding 
            # the ROI using the projected border but it's far easier and covers wrap around and the poles with
            # minimal effort. It will also probably be more efficient when wrap around or crossing the poles does occur.
            
            dst_pixels_denormalized_int = features.denormalized_image_coordinates( dst_pixels, camera.width, camera.height ).astype( np.int32 )
            
            dst_pixels_snap = features.normalized_image_coordinates( dst_pixels_denormalized_int.astype( np.float32 ), camera.width, camera.height )
            
            dst_bearings_re = camera.pixel_bearing_many( dst_pixels_snap )
            
            # Project mosaic pixel center bearings back into the source image
            
            src_re_x, src_re_y = shot_cam.project( ( dst_bearings_re[:, 0],
                                                     dst_bearings_re[:, 1],
                                                     dst_bearings_re[:, 2] ) )
                                                     
            src_re_pixels = np.column_stack( [ src_re_x.ravel(), src_re_y.ravel() ] )
            
            src_re_denormalized = features.denormalized_image_coordinates( src_re_pixels, shot_cam.width, shot_cam.height )
            
            mosaic_img = initialize_mosaic_image( camera.width, camera.height, img )
            
            # Reshape arrays for cv.remap efficiency reasons and due to the SHRT_MAX limit of array size.
            # Another option is to process in chunks of linear array of shize SHRT_MAX. However, this 
            # approach was probably 4x slower.
            
            x = src_re_denormalized[:, 0].reshape( src_x.shape ).astype(np.float32)
            y = src_re_denormalized[:, 1].reshape( src_y.shape ).astype(np.float32)
            
            r_map_x = x
            r_map_y = y
            
            # Sample source imagery colors
            
            colors = cv2.remap( img, x, y, cv2.INTER_LINEAR , borderMode=cv2.BORDER_CONSTANT )
            
            dst_mask_y = dst_pixels_denormalized_int[:, 1].reshape( src_y.shape )
            dst_mask_x = dst_pixels_denormalized_int[:, 0].reshape( src_x.shape )
            
            mosaic_img[ dst_mask_y, dst_mask_x ] = colors
            
            blend_projection_border(  mosaic_img, dst_mask_y, dst_mask_x )
            
            
            # Initialize blurring and alpha mask kernels
            
            # half_chunk_size = 75
            # border = 41
            
            # half_size = half_chunk_size + border
            
            # kernel_1d = cv2.getGaussianKernel( 2*half_chunk_size+1,  1.5*(0.3*((2*half_chunk_size+1-1)*0.5 - 1) + 0.8) , cv2.CV_32F )
            # kernel_1d/=kernel_1d[ half_chunk_size ]
            
            # half_kernel_1d = kernel_1d[ half_chunk_size : 2*half_chunk_size ]
            
            # alpha = np.zeros( ( 2*half_chunk_size, 2*half_chunk_size, 3 ), dtype = np.float32 ) #np.float32 uint8)
            
            # for y in range(0,2*half_chunk_size):
                # for x in range(0,2*half_chunk_size):
                    # yt = y - half_chunk_size
                    # xt = x - half_chunk_size
                    
                    # r = int( math.sqrt( yt*yt + xt*xt ) )
                    
                    # if r > half_chunk_size-1:
                        # r = half_chunk_size-1
                    
                    # kv = half_kernel_1d[r]
                    
                    # alpha[ y, x, 0] = alpha[ y, x, 1] = alpha[ y, x, 2] = kv
            
            
            # # Grab the indices of pixels along the projected image border and blend into the
            # # background with a gaussian blur and alpha map.
            
            # dst_mask_y_border = np.concatenate( [ dst_mask_y[ 0:,0 ], 
                                                  # dst_mask_y[ 0:, -1 ],
                                                  # dst_mask_y[ 0, 0: ],
                                                  # dst_mask_y[-1, 0: ] ] )
                            
            # dst_mask_x_border = np.concatenate( [ dst_mask_x[ 0:,0 ], 
                                                  # dst_mask_x[ 0:, -1 ],
                                                  # dst_mask_x[ 0, 0: ],
                                                  # dst_mask_x[-1, 0: ] ] )
            
            # dst_mask_border = np.column_stack( [ dst_mask_y_border, dst_mask_x_border ] )
            
            # #for y_ind in np.arange( 0, dst_mask_y.shape[0], 75 ):
            
            # for border_pix in dst_mask_border[::75]:
                
                # border_y = border_pix[0] #dst_mask_y[y_ind,0]
                # border_x = border_pix[1] #dst_mask_x[y_ind,0]
            
                # sub_img = mosaic_img[ border_y - half_size : border_y + half_size, border_x - half_size : border_x + half_size ].copy()
            
                # sub_rng = border + 2*half_chunk_size
            
                # sub_img[border:sub_rng,border:sub_rng] = cv2.GaussianBlur( sub_img[border:sub_rng,border:sub_rng], (81,81), 0 )
            
                # mosaic_img[ border_y - half_chunk_size : border_y + half_chunk_size, border_x - half_chunk_size : border_x + half_chunk_size ] = \
                    # np.multiply( sub_img[border:sub_rng,border:sub_rng].astype( np.float32 ), alpha ) + \
                    # np.multiply( mosaic_img[ border_y - half_chunk_size : border_y + half_chunk_size, border_x - half_chunk_size : border_x + half_chunk_size ].astype( np.float32 ), 1 - alpha )
                
            #cv2.imwrite('c:\\alpha.png', alpha)
            
            #mosaic_img[ border_y - half_chunk_size : border_y + half_chunk_size, border_x - half_chunk_size : border_x + half_chunk_size ] = alpha
            
            #mosaic_img[ border_y - half_chunk_size : border_y + half_chunk_size, border_x - half_chunk_size : border_x + half_chunk_size ] = sub_img[border:sub_rng,border:sub_rng]
            
        elif interp_mode == 'nearest':

            # Implementing nearest this way rather than just changing the interpolation function of cv2.remap above
            # will be more efficient because we'll avoid the reprojection back to the source image and sample it directly
            # using our index mask.

            dst_pixels_denormalized = features.denormalized_image_coordinates( dst_pixels, camera.width, camera.height )

            # Create a full equirectangular index image with all zero indices for x and y

            fdst_y, fdst_x = np.zeros( ( 2, camera.height, camera.width ) ).astype( np.float32 )
            
            # Use the projected indices to swap in the source image indices.
            
            x = dst_pixels_denormalized[..., 0].astype(np.int32)
            y = dst_pixels_denormalized[..., 1].astype(np.int32)
            
            fdst_x[ y, x ] = src_pixels_denormalized[...,0]
            fdst_y[ y, x ] = src_pixels_denormalized[...,1]
            
            r_map_x = fdst_x
            r_map_y = fdst_y
            
            mosaic_img = cv2.remap( img, fdst_x, fdst_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT )
        
        else:
            raise NotImplementedError( 'Interpolation type not supported: {}'.format( interp_mode ) )
            
    data.save_full_mosaic_image( os.path.splitext( shot.id )[0], mosaic_img )

    end = timer()
    
    report = {
        "image": shot.id,
        "wall_time": end - start,
    }
    
    data.save_report( io.json_dumps(report),
                      'full_mosaic_reprojection/{}.json'.format( shot.id ) )
                      
    return ( r_map_x, r_map_y, dst_mask_x, dst_mask_y )
    
