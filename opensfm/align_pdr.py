"""affine transform pdr predictions to align with GPS points or SfM output."""

import operator
import logging
import math
import cv2
import numpy as np
from cmath import rect, phase

from itertools import combinations

from opensfm import csfm
from opensfm import geo
from opensfm import multiview
from opensfm import types
from opensfm import transformations as tf

logger = logging.getLogger(__name__)

# pruning parameters
ABS_HEIGHT_THRESH = 2.5  # in meters
REL_HEIGHT_THRESH = 1.0  # in meters
DISTANCE_THRESH = 1.25  # in meters
MOVEMENT_THRESH = 0.25  # in meters
SIG_MOVEMENT_THRESH = 0.50  # in meters
HEADING_THRESH = 10.0  # in degrees
ANGLE_THRESH = 120  # in degrees

# eliminate NEIGHBOR_RADIUS frames around each side of suspicious frame
NEIGHBOR_RADIUS = 2

# max size of holes. must be smaller than (2*NEIGHBOR_RADIUS+1). set to 1 to disallow holes
MAX_HOLE_SIZE = 3

# number of predictions using PDR
PDR_TRUST_SIZE = 20

# form new recons with more than MIN_RECON_SIZE consecutive frames (with possible hole in the middle)
# this should be twice the PDR_TRUST_SIZE
MIN_RECON_SIZE = 2*PDR_TRUST_SIZE


def distance_2d_no_numpy(p1, p2, expected_scale):
    x = p1[0] - p2[0]
    y = p1[1] - p2[1]
    return math.sqrt(x*x + y*y) / expected_scale


def filter_gps_points_no_numpy(gps_points_dict, expected_scale):
    shot_ids = sorted(gps_points_dict.keys())
    keep_shot_ids = [shot_ids[-1]]

    for i in range(len(shot_ids) - 2, 0, -1):
        if distance_2d_no_numpy(gps_points_dict[shot_ids[i]], gps_points_dict[keep_shot_ids[-1]], expected_scale) > 3.0:
            keep_shot_ids.append(shot_ids[i])

    keep_shot_ids.append(shot_ids[0])

    return {k: v for k, v in gps_points_dict.items() if k in keep_shot_ids}


def update_pdr_global_2d(gps_points_dict, pdr_shots_dict, scale_factor, skip_bad=True):
    """
    *globally* align pdr predictions to GPS points. used with direct_align

    use 2 gps points at a time to align pdr predictions

    :param gps_points_dict: gps points in topocentric coordinates
    :param pdr_shots_dict: position of each shot as predicted by pdr
    :param scale_factor: reconstruction_scale_factor
    :param skip_bad: avoid bad alignment sections
    :return: aligned pdr shot predictions
    """
    if len(gps_points_dict) < 2 or len(pdr_shots_dict) < 2:
        return {}

    # reconstruction_scale_factor is from oiq_config.yaml, and it's feet per pixel.
    # 0.3048 is meter per foot. 1.0 / (reconstruction_scale_factor * 0.3048) is
    # therefore pixels/meter, and since pdr output is in meters, it's the
    # expected scale
    expected_scale = 1.0 / (scale_factor * 0.3048)

    pdr_predictions_dict = {}

    all_gps_shot_ids = sorted(gps_points_dict.keys())
    for i in range(len(all_gps_shot_ids) - 1):
        gps_coords = []
        pdr_coords = []

        for j in range(2):
            shot_id = all_gps_shot_ids[i+j]
            gps_coords.append(gps_points_dict[shot_id])
            pdr_coords.append([pdr_shots_dict[shot_id][0], pdr_shots_dict[shot_id][1], 0])

        # the last few pdr coords maybe identical if no steps were detected. treat this as a special case
        # to avoid divide by zero later on
        if pdr_coords[0] == pdr_coords[1]:
            pdr_coords[1][0] = pdr_coords[0][0] + (gps_coords[1][0] - gps_coords[0][0])/expected_scale
            pdr_coords[1][1] = pdr_coords[0][1] + (gps_coords[1][1] - gps_coords[0][1])/expected_scale

        #s, A, b = get_affine_transform_2d(gps_coords, pdr_coords)
        s, A, b = get_affine_transform_2d_no_numpy(gps_coords, pdr_coords)

        # the closer s is to expected_scale, the better the fit, and the less the deviation
        deviation = math.fabs(1.0 - s/expected_scale)

        # debugging
        #[x, y, z] = _rotation_matrix_to_euler_angles(A)
        #logger.debug("update_pdr_global_2d: deviation=%f, rotation=%f, %f, %f", deviation, np.degrees(x), np.degrees(y), np.degrees(z))

        if skip_bad and not ((0.50 * expected_scale) < s < (2.0 * expected_scale)):
            logger.debug("s/expected_scale={}, discard".format(s/expected_scale))
            continue

        start_shot_id = all_gps_shot_ids[i]
        end_shot_id = all_gps_shot_ids[i+1]

        # in first iteration, we transform pdr from first shot
        # in last iteration, we transform pdr until last shot
        if i == 0:
            start_shot_id = _int_to_shot_id(0)

        if i == len(gps_points_dict)-2:
            end_shot_id = _int_to_shot_id(len(pdr_shots_dict)-1)

        #new_dict = apply_affine_transform(pdr_shots_dict, start_shot_id, end_shot_id,
        #s, A, b,
        #deviation, [all_gps_shot_ids[i], all_gps_shot_ids[i+1]])
        new_dict = apply_affine_transform_no_numpy(pdr_shots_dict, start_shot_id, end_shot_id,
                                                   s, A, b,
                                                   deviation, [all_gps_shot_ids[i], all_gps_shot_ids[i+1]])

        pdr_predictions_dict.update(new_dict)

    # special case: if last two points are very close, alignment to them can be quite off, and the
    # extrapolations beyond the last point can contain large errors. therefore we seek a second
    # point that's sufficiently far away from the last point, which should make the alignment
    # (hence the extrapolations) more stable. if no such point was found, we would just fall back to
    # use whatever was calculated above
    p_last = gps_points_dict[all_gps_shot_ids[-1]]
    for i in range(len(all_gps_shot_ids) - 2, 0, -1):
        p_curr = gps_points_dict[all_gps_shot_ids[i]]
        x = p_last[0] - p_curr[0]
        y = p_last[1] - p_curr[1]

        if math.sqrt(x * x + y * y) / expected_scale > 5.0: # 5 meters
            gps_coords = [p_curr, p_last]
            pdr_coords = [[pdr_shots_dict[all_gps_shot_ids[i]][0], pdr_shots_dict[all_gps_shot_ids[i]][1], 0],
                          [pdr_shots_dict[all_gps_shot_ids[-1]][0], pdr_shots_dict[all_gps_shot_ids[-1]][1], 0]]

            s, A, b = get_affine_transform_2d_no_numpy(gps_coords, pdr_coords)
            deviation = math.fabs(1.0 - s / expected_scale)

            # this transformation is applied to shots after the last gps point only
            new_dict = apply_affine_transform_no_numpy(pdr_shots_dict, all_gps_shot_ids[-1], _int_to_shot_id(len(pdr_shots_dict)-1),
                                                       s, A, b,
                                                       deviation, [all_gps_shot_ids[i], all_gps_shot_ids[-1]])
            pdr_predictions_dict.update(new_dict)
            break

    return pdr_predictions_dict


def update_pdr_global(gps_points_dict, pdr_shots_dict, scale_factor, skip_bad=True, stride_len=3):
    """
    *globally* align pdr predictions to GPS points. used with direct_align

    Move a sliding window through the gps points and get stride_len neighboring points at a time;
    use them to piece-wise affine transform pdr predictions to align with GPS points

    :param gps_points_dict: gps points in topocentric coordinates
    :param pdr_shots_dict: position of each shot as predicted by pdr
    :param scale_factor: reconstruction_scale_factor - scale factor feet per pixel
    :param skip_bad: avoid bad alignment sections
    :param stride_len: how many gps points are used for each section
    :return: aligned pdr shot predictions - [x, y, z, dop]
    """
    if len(gps_points_dict) < stride_len or len(pdr_shots_dict) < stride_len:
        logger.info("update_pdr_global: need more gps points. supplied only {}", len(gps_points_dict))
        return {}

    pdr_predictions_dict = {}

    # reconstruction_scale_factor is from oiq_config.yaml, and it's feet per pixel.
    # 0.3048 is meter per foot. 1.0 / (reconstruction_scale_factor * 0.3048) is
    # therefore pixels/meter, and since pdr output is in meters, it's the
    # expected scale
    expected_scale = 1.0 / (scale_factor * 0.3048)

    last_deviation = 1.0

    all_gps_shot_ids = sorted(gps_points_dict.keys())
    first_iteration = True
    for i in range(len(all_gps_shot_ids) - stride_len + 1):
        gps_coords = []
        pdr_coords = []
        gps_shot_ids = []

        for j in range(stride_len):
            shot_id = all_gps_shot_ids[i+j]
            gps_shot_ids.append(shot_id)
            gps_coords.append(gps_points_dict[shot_id])
            pdr_coords.append(pdr_shots_dict[shot_id][0:3])

        s, A, b = get_affine_transform(gps_coords, pdr_coords)

        # the closer s is to expected_scale, the better the fit, and the less the deviation
        ratio = s/expected_scale
        if ratio > 1.0:
            ratio = 1/ratio

        # if deviation is very large, skip it
        deviation = math.fabs(1.0 - ratio)
        if skip_bad and deviation > 0.5:
            last_deviation = 1.0
            continue

        # if x/y rotation is not close to 0, then likely it's 'flipped' and no good
        [x, y, z] = _rotation_matrix_to_euler_angles(A)
        logger.debug("update_pdr_global: deviation=%f, rotation=%f, %f, %f", deviation, np.degrees(x), np.degrees(y), np.degrees(z))
        if skip_bad and (math.fabs(x) > 1.0 or math.fabs(y) > 1.0):
            last_deviation = 1.0
            continue

        # based on deviation, we choose different starting pdr shot to transform
        if deviation < last_deviation:
            pdr_start_shot_id = gps_shot_ids[0]
        else:
            pdr_start_shot_id = gps_shot_ids[1]

        pdr_end_shot_id = _int_to_shot_id(len(pdr_shots_dict)-1)

        if first_iteration:
            # in first iteration, we transform pdr from first shot
            pdr_start_shot_id = _int_to_shot_id(0)
            first_iteration = False

        new_dict = apply_affine_transform(pdr_shots_dict, pdr_start_shot_id, pdr_end_shot_id,
                                          s, A, b,
                                          deviation, gps_shot_ids)
        pdr_predictions_dict.update(new_dict)

        last_deviation = deviation

    return pdr_predictions_dict


def init_pdr_predictions(data, use_2d=False):
    """
    globally align pdr path to gps points

    :param data:
    :param use_2d:
    """
    if not data.gps_points_exist():
        return

    if not data.pdr_shots_exist():
        return

    scale_factor = data.config['reconstruction_scale_factor']
    gps_points_dict = data.load_gps_points()
    reflla = data.load_reference_lla()
    pdr_shots_dict = data.load_pdr_shots()

    topocentric_gps_points_dict = {}

    for key, value in gps_points_dict.items():
        x, y, z = geo.topocentric_from_lla(
            value[0], value[1], value[2],
            reflla['latitude'], reflla['longitude'], reflla['altitude'])
        topocentric_gps_points_dict[key] = [x, y, z]

    if use_2d:
        pdr_predictions_dict = update_pdr_global_2d(topocentric_gps_points_dict, pdr_shots_dict, scale_factor, False)
    else:
        pdr_predictions_dict = update_pdr_global(topocentric_gps_points_dict, pdr_shots_dict, scale_factor)
        if len(pdr_predictions_dict) != len(pdr_shots_dict):
            # under degenerate configurations, update_pdr_global can fail to produce pdr predictions for
            # every shot. in that case, we revert to 2-point alignment below
            pdr_predictions_dict = update_pdr_global_2d(topocentric_gps_points_dict, pdr_shots_dict, scale_factor, False)

    data.save_topocentric_gps_points(topocentric_gps_points_dict)
    data.save_pdr_predictions(pdr_predictions_dict)

    return pdr_predictions_dict


def direct_align_pdr(data, target_images=None):
    """
    directly form a reconstruction based on pdr data
    :param data:
    :param target_images:
    :return: reconstruction
    """
    pdr_predictions_dict = init_pdr_predictions(data, True)
    pdr_shots_dict = data.load_pdr_shots()

    if not target_images:
        target_images = data.config.get('target_images', [])

    cameras = data.load_camera_models()

    reconstruction = types.Reconstruction()
    reconstruction.cameras = cameras

    for img in target_images:

        camera = cameras[data.load_exif(img)['camera']]

        shot = types.Shot()
        shot.id = img
        shot.camera = camera
        shot.pose = types.Pose()

        prev_img = _prev_shot_id(img)
        next_img = _next_shot_id(img)

        curr_coords = pdr_predictions_dict[img][:3]

        prev_heading = next_heading = heading = None
        if prev_img in pdr_predictions_dict:
            prev_coords = pdr_predictions_dict[prev_img][:3]
            prev_heading = np.arctan2(curr_coords[1] - prev_coords[1], curr_coords[0] - prev_coords[0])

        if next_img in pdr_predictions_dict:
            next_coords = pdr_predictions_dict[next_img][:3]
            next_heading = np.arctan2(next_coords[1] - curr_coords[1], next_coords[0] - curr_coords[0])

        if prev_heading and next_heading:
            heading = phase((rect(1, prev_heading) + rect(1, next_heading)) * 0.5)
        elif prev_heading:
            heading = prev_heading
        elif next_heading:
            heading = next_heading

        if not heading:
            continue

        # Our floorplan/gps coordinate system: x point right, y point back, z point down
        #
        # OpenSfM 3D viewer coordinate system: x point left, y point back, z point up (or equivalently it can be
        # viewed as x point right, y point forward, z point up)
        #
        # OpenSfM camera coordinate system: x point right of its body, y point down, z point forward (look-at dir)
        #
        # Since our floorplan/gps uses a different coordinate system than the OpenSfM 3D viewer, reconstructions
        # are upside down in the 3D viewer.
        #
        # We can fix in one of two ways: 1) assume the origin of the floor plan to be bottom-left, rather than top-
        # left; or 2) we can hack the OpenSfM 3D viewer for it to follow our coordinate system. The first option is
        # better, however it will probably require changes in both the current gps picker and our own viewer.
        #
        # If camera has 0 rotation on all axes relative to OpenSfM 3D viewer coordinate system, then in the
        # viewer, its lens points up towards the sky. If camera has 0 rotation relative to our floorplan/gps
        # coordinate system, its lens points down towards the ground.
        #
        # What *should* the camera rotation be, when the camera is placed on a table (i.e. there is no roll or
        # pitch) and have a heading of exactly zero degrees? In this case, the camera lens (z) would be horizontal
        # looking at the positive x axis. Therefore, relative to our floorplan/gps coordinate system, its rotation
        # expressed in euler angles in xyz order should be (pi/2, 0, pi/2). This should be considered as the
        # 'canonical' rotation of the camera in our floorplan/gps coordinate system.
        #
        # NC Tech camera imu sensor coordinate system: x point right of body, y point forward, z point up. Roll,
        # pitch and heading in pdr_shots.txt are rotations of this coordinate system relative to the ground reference
        # frame which is assumed to be ENU (east-north-up). However, because the magnetometer is uncalibrated and
        # can't be trusted, the heading is relative to a rather random starting point and is not absolute.
        #
        # The 'heading' calculated above however, is relative to floorplan/gps coordinate system and is the
        # rotation around its z axis. It will be used to replace the heading in pdr_shots.txt.
        #
        # In the 'canonical' configuration, our floorplan/gps has: x point right, y point back, z point down;
        # camera has x point back, y point down, z point right; imu has x point back, y point right, z point up.
        # Now we need to convert the roll/pitch that's relative to the imu coordinate system to floorplan/gps
        # coordinate system, which means we swap roll/pitch. In matrix form this transformation is:
        #     [0  1  0]
        #     [1  0  0]
        #     [0  0 -1]
        # Again, we will use the 'heading' from calculation above, which is based on alignment with annotated
        # gps points, and only swap roll/pitch. Finally we concatenate this matrix with the 'canonical'
        # transformation to obtain the final rotation matrix.
        R1 = _euler_angles_to_rotation_matrix([np.pi*0.5, 0, np.pi*0.5])
        R2 = _euler_angles_to_rotation_matrix([np.radians(pdr_shots_dict[img][4]), np.radians(pdr_shots_dict[img][3]), heading])
        R = R2.dot(R1)

        t_shot = np.array(pdr_predictions_dict[img][:3])
        tp = -R.T.dot(t_shot)

        shot.pose.set_rotation_matrix(R.T)
        shot.pose.translation = list(tp)

        reconstruction.add_shot(shot)

    reconstruction.alignment.aligned = True
    reconstruction.alignment.num_correspondences = len(target_images)

    return reconstruction


def hybrid_align_pdr(data, target_images=None):
    """
    this routine is intended to be run after gps picking is complete

    after data processor is done gps picking, this routine should be invoked, which will trigger
    update_gps_picker_hybrid first. for all shots not in an aligned recon, direct alignment will
    be performed on them and they will be grouped into one 'aligned' recon.

    """
    # load gps points and convert them to topocentric
    gps_points_dict = data.load_gps_points()
    reflla = data.load_reference_lla()

    curr_gps_points_dict = {}
    for key, value in gps_points_dict.items():
        x, y, z = geo.topocentric_from_lla(
            value[0], value[1], value[2],
            reflla['latitude'], reflla['longitude'], reflla['altitude'])
        curr_gps_points_dict[key] = [x, y, z]

    # now align recons that has 2 or more gps points (and trusted shots if any). we run through the same
    # procedure as in hybrid gps picking process, so that the output will be identical. only difference
    # is that align_reconstruction_segments below calculates the full camera pose, as opposed to position
    # only in align_reconstruction_no_numpy.
    scale_factor = data.config['reconstruction_scale_factor']
    pdr_shots_dict = data.load_pdr_shots()
    reconstructions = data.load_reconstruction()

    aligned_recons = []
    aligned_shots_dict = curr_gps_points_dict.copy()

    graph = None

    # init pdr predictions
    pdr_predictions_dict = update_pdr_global_2d(curr_gps_points_dict, pdr_shots_dict, scale_factor, False)

    # align recons to gps points and/or trusted shots
    while True:
        can_align = False
        for recon in reconstructions:
            if recon.alignment.aligned or len(recon.shots) < MIN_RECON_SIZE:
                continue

            recon_gps_points = {}
            recon_trusted_shots = {}

            # match gps points to this recon
            for shot_id in recon.shots:
                if shot_id in curr_gps_points_dict:
                    recon_gps_points[shot_id] = curr_gps_points_dict[shot_id]

            # find trusted shots on this recon if not enough gps points
            if len(recon_gps_points) < 2:
                recon_shot_ids = sorted(recon.shots)

                if recon_shot_ids[0] not in curr_gps_points_dict and \
                        _prev_shot_id(recon_shot_ids[0]) in aligned_shots_dict:
                    recon_trusted_shots[recon_shot_ids[0]] = pdr_predictions_dict[recon_shot_ids[0]][:3]

                if recon_shot_ids[-1] not in curr_gps_points_dict and \
                        _next_shot_id(recon_shot_ids[-1]) in aligned_shots_dict:
                    recon_trusted_shots[recon_shot_ids[-1]] = pdr_predictions_dict[recon_shot_ids[-1]][:3]

            if len(recon_gps_points) + len(recon_trusted_shots) >= 2:
                # combine trusted shots with gps points
                recon_trusted_shots.update(recon_gps_points)

                # only need to load graph if it hasn't been loaded before AND there are more than
                # 2 trusted points on this recon (hence the need to break it into segments)
                if graph is None and len(recon_trusted_shots) > 2:
                    graph = data.load_tracks_graph()

                # below, each 'segment' is a Reconstruction object
                segments = align_reconstruction_segments(data, graph, recon, recon_trusted_shots)
                aligned_recons.extend(segments)

                # the 'shot' objects in segments are the same as those in recon
                for shot_id in recon.shots:
                    aligned_shots_dict[shot_id] = recon.shots[shot_id].pose.get_origin()

                # update pdr predictions based on aligned shots so far
                pdr_predictions_dict = update_pdr_global_2d(aligned_shots_dict, pdr_shots_dict, scale_factor, False)

                recon.alignment.aligned = True
                can_align = True
                break

        if not can_align:
            break

    # for shots that are not in aligned recons at this point, we throw them in a new recon. the
    # camera poses are calculated using the same method as direct align
    pdr_predictions_dict = update_pdr_global_2d(aligned_shots_dict, pdr_shots_dict, scale_factor, False)

    if not target_images:
        target_images = data.config.get('target_images', [])

    cameras = data.load_camera_models()

    direct_align_recon = types.Reconstruction()
    direct_align_recon.cameras = cameras

    for img in target_images:

        if img in aligned_shots_dict:
            continue

        camera = cameras[data.load_exif(img)['camera']]

        shot = types.Shot()
        shot.id = img
        shot.camera = camera
        shot.pose = types.Pose()

        prev_img = _prev_shot_id(img)
        next_img = _next_shot_id(img)

        curr_coords = pdr_predictions_dict[img][:3]

        prev_heading = next_heading = heading = None
        if prev_img in pdr_predictions_dict:
            prev_coords = pdr_predictions_dict[prev_img][:3]
            prev_heading = np.arctan2(curr_coords[1] - prev_coords[1], curr_coords[0] - prev_coords[0])

        if next_img in pdr_predictions_dict:
            next_coords = pdr_predictions_dict[next_img][:3]
            next_heading = np.arctan2(next_coords[1] - curr_coords[1], next_coords[0] - curr_coords[0])

        if prev_heading is not None and next_heading is not None:
            heading = phase((rect(1, prev_heading) + rect(1, next_heading)) * 0.5)
        elif prev_heading is not None:
            heading = prev_heading
        elif next_heading is not None:
            heading = next_heading

        if heading is None:
            continue

        R1 = _euler_angles_to_rotation_matrix([np.pi*0.5, 0, np.pi*0.5])
        R2 = _euler_angles_to_rotation_matrix([np.radians(pdr_shots_dict[img][4]), np.radians(pdr_shots_dict[img][3]), heading])
        R = R2.dot(R1)

        t_shot = np.array(pdr_predictions_dict[img][:3])
        tp = -R.T.dot(t_shot)

        shot.pose.set_rotation_matrix(R.T)
        shot.pose.translation = list(tp)

        direct_align_recon.add_shot(shot)

    if len(direct_align_recon.shots) > 0:
        direct_align_recon.alignment.aligned = True
        direct_align_recon.alignment.num_correspondences = len(direct_align_recon.shots)

        aligned_recons.append(direct_align_recon)

    return aligned_recons


def position_extrapolate(dist_1_coords, dist_2_coords, delta_heading, delta_distance):
    """
    update pdr predictions based on extrapolating last SfM position and direction

    :param dist_1_coords: sfm point immediately preceding or following (distance = 1)
    :param dist_2_coords: sfm point next to dist_1_coords (distance = 2)
    :param delta_heading: delta heading
    :param delta_distance: delta distance
    :return: updated pdr prediction
    """
    ref_coord = dist_1_coords
    ref_dir = np.arctan2(dist_1_coords[1] - dist_2_coords[1], dist_1_coords[0] - dist_2_coords[0])

    curr_dir = ref_dir + delta_heading
    x = ref_coord[0] + delta_distance*np.cos(curr_dir)
    y = ref_coord[1] + delta_distance*np.sin(curr_dir)
    z = ref_coord[2]

    return [x, y, z]


def rotation_extrapolate(shot_id, base_shot_id, reconstruction, data):
    """
    based on pdr rotations of base shot and current shot, calculate a delta rotation,
    then apply this delta rotation to sfm rotation of base to obtain a prediction/prior
    for sfm rotation of current shot
    :param shot_id:
    :param base_shot_id:
    :param reconstruction:
    :param data:
    :return: prediction for sfm rotation of current shot
    """
    pdr_shots_dict = data.load_pdr_shots()

    # calculate delta rotation
    base_pdr_rotation = _euler_angles_to_rotation_matrix(np.radians(pdr_shots_dict[base_shot_id][3:6]))
    pdr_rotation = _euler_angles_to_rotation_matrix(np.radians(pdr_shots_dict[shot_id][3:6]))
    delta_rotation = pdr_rotation.dot(base_pdr_rotation.T)

    # get sfm rotation of base shot
    base_sfm_rotation = reconstruction.shots[base_shot_id].pose.get_rotation_matrix().T

    # return prediction
    return _rotation_matrix_to_euler_angles((delta_rotation.dot(base_sfm_rotation)).T)


def update_pdr_local(shot_id, sfm_points_dict, pdr_shots_dict, scale_factor):
    """
    *locally* align pdr predictions to SfM output. the SfM points have been aligned with
    GPS points. used with bundle_use_pdr

    :param shot_id: update pdr prediction for this shot
    :param sfm_points_dict: sfm point coordinates
    :param pdr_shots_dict: original predictions
    :param scale_factor: reconstruction_scale_factor - scale factor feet per pixel
    :return: updated pdr prediction for shot_id
    """
    prev1 = _prev_shot_id(shot_id)
    prev2 = _prev_shot_id(prev1)
    next1 = _next_shot_id(shot_id)
    next2 = _next_shot_id(next1)

    dist_1_id = dist_2_id = None
    if prev1 in sfm_points_dict and prev2 in sfm_points_dict:
        dist_1_id = prev1
        dist_2_id = prev2
    elif next1 in sfm_points_dict and next2 in sfm_points_dict:
        dist_1_id = next1
        dist_2_id = next2

    #logger.info("update_pdr_local: update {} based on {} {}".format(shot_id, dist_1_id, dist_2_id))

    if dist_1_id and dist_2_id:
        dist_1_coords = sfm_points_dict[dist_1_id]
        dist_2_coords = sfm_points_dict[dist_2_id]

        pdr_info_dist_1 = pdr_shots_dict[dist_1_id]
        pdr_info = pdr_shots_dict[shot_id]

        delta_heading = tf.delta_heading(np.radians(pdr_info_dist_1[3:6]), np.radians(pdr_info[3:6]))

        # scale pdr delta distance according to sfm estimate for last step. we don't use
        # pdr delta distance directly because stride length is not very accurate in tight
        # spaces, which is often the case for us. also make sure we don't get wild values
        # when pdr delta distance for last step is very small
        if pdr_info_dist_1[9] > 1e-1:
            delta_distance = pdr_info[9] / pdr_info_dist_1[9] * np.linalg.norm(dist_1_coords-dist_2_coords)
        else:
            delta_distance = pdr_info_dist_1[9]

        # TODO: put 200 in config
        return position_extrapolate(dist_1_coords, dist_2_coords, delta_heading, delta_distance), 200
    else:
        return [0, 0, 0], 999999.0


def update_pdr_local_affine(shot_id, sfm_points_dict, pdr_shots_dict, scale_factor, sorted_shot_ids):
    """
    estimating the affine transform between a set of SfM point coordinates and a set of
    original pdr predictions. Then affine transform the pdr predictions

    :param shot_id: the shot to update
    :param sfm_points_dict: sfm point coordinates
    :param pdr_shots_dict: original predictions
    :param scale_factor: reconstruction_scale_factor - scale factor feet per pixel
    :param sorted_shot_ids: sfm_shot_ids sorted by their closeness to shot_id
    :return: updated pdr predictions for shot_id
    """
    if len(sorted_shot_ids) < 3:
        return [0, 0, 0], 999999

    # reconstruction_scale_factor is from oiq_config.yaml, and it's feet per pixel.
    # 0.3048 is meter per foot. 1.0 / (reconstruction_scale_factor * 0.3048) is
    # therefore pixels/meter, and since pdr output is in meters, it's the
    # expected scale
    expected_scale = 1.0 / (scale_factor * 0.3048)

    sfm_coords = []
    pdr_coords = []

    for i in range(3):
        a_id = sorted_shot_ids[i]
        sfm_coords.append(sfm_points_dict[a_id])
        pdr_coords.append(pdr_shots_dict[a_id][0:3])

    s, A, b = get_affine_transform(sfm_coords, pdr_coords)

    # the closer s is to expected_scale, the better the fit, and the less the deviation
    ratio = s/expected_scale
    if ratio > 1.0:
        ratio = 1/ratio

    # if deviation is very large, skip it
    deviation = math.fabs(1.0 - ratio)
    if deviation > 0.5:
        return [0, 0, 0], 999999

    # if x/y rotation is not close to 0, then likely it's 'flipped' and no good
    [x, y, z] = _rotation_matrix_to_euler_angles(A)
    if math.fabs(x) > 1.0 or math.fabs(y) > 1.0:
        return [0, 0, 0], 999999

    update = apply_affine_transform(pdr_shots_dict, shot_id, shot_id, s, A, b, deviation)

    #logger.info("update_pdr_local_affine: shot_id {}, new prediction {}".format(shot_id, update[shot_id]))
    return update[shot_id][:3], update[shot_id][3]


def update_pdr_prediction_position(shot_id, reconstruction, data):
    if data.pdr_shots_exist():
        if len(reconstruction.shots) < 3:
            return [0, 0, 0], 999999.0

        # get updated predictions
        sfm_points_dict = {}

        for shot in reconstruction.shots.values():
            sfm_points_dict[shot.id] = shot.pose.get_origin()

        pdr_shots_dict = data.load_pdr_shots()
        scale_factor = data.config['reconstruction_scale_factor']

        return update_pdr_local(shot_id, sfm_points_dict, pdr_shots_dict, scale_factor)

    return [0, 0, 0], 999999.0


def update_pdr_prediction_rotation(shot_id, reconstruction, data):
    """
    get rotation prior of shot_id based on the closest shots in recon.
    :param shot_id:
    :param reconstruction:
    :param data:
    :return:
    """
    if data.pdr_shots_exist():
        if len(reconstruction.shots) < 3:
            return [0, 0, 0], 999999.0

        # get sorted shot ids that are closest to shot_id (in sequence number)
        sorted_shot_ids = get_closest_shots(shot_id, reconstruction.shots.keys())

        base_shot_id_0 = sorted_shot_ids[0]
        base_shot_id_1 = sorted_shot_ids[1]

        prediction_0 = rotation_extrapolate(shot_id, base_shot_id_0, reconstruction, data)
        prediction_1 = rotation_extrapolate(shot_id, base_shot_id_1, reconstruction, data)

        # 0.1 radians is roughly 6 degrees
        # TODO: put 0.1 in config
        tolerance = 0.1 * abs(_shot_id_to_int(base_shot_id_0) - _shot_id_to_int(shot_id))

        q_0 = tf.quaternion_from_euler(prediction_0[0], prediction_0[1], prediction_0[2])
        q_1 = tf.quaternion_from_euler(prediction_1[0], prediction_1[1], prediction_1[2])

        if tf.quaternion_distance(q_0, q_1) > tolerance:
            return prediction_0, 999999.0

        return prediction_0, tolerance

    return [0, 0, 0], 999999.0


def align_reconstructions_to_pdr(reconstructions, data):
    reconstructions[:] = [align_reconstruction_to_pdr(recon, data) for recon in reconstructions]


def align_reconstruction_to_pdr(reconstruction, data):
    """
    leveling and scaling the reconstructions to pdr
    """
    if reconstruction.alignment.aligned:
        return reconstruction

    if not data.pdr_shots_exist():
        return reconstruction

    pdr_shots_dict = data.load_pdr_shots()

    X, Xp = [], []
    onplane, verticals = [], []
    for shot_id in reconstruction.shots.keys():
        X.append(reconstruction.shots[shot_id].pose.get_origin())
        Xp.append(pdr_shots_dict[shot_id][0:3])
        R = reconstruction.shots[shot_id].pose.get_rotation_matrix()
        onplane.append(R[0,:])
        onplane.append(R[2,:])
        verticals.append(R[1,:])

    X = np.array(X)
    Xp = np.array(Xp)

    # Estimate ground plane.
    p = multiview.fit_plane(X - X.mean(axis=0), onplane, verticals)
    Rplane = multiview.plane_horizontalling_rotation(p)
    X = Rplane.dot(X.T).T

    # Estimate 2d similarity to align to pdr predictions
    T = tf.affine_matrix_from_points(X.T[:2], Xp.T[:2], shear=False)
    s = np.linalg.det(T[:2, :2]) ** 0.5
    A = np.eye(3)
    A[:2, :2] = T[:2, :2] / s
    A = A.dot(Rplane)
    b = np.array([
        T[0, 2],
        T[1, 2],
        Xp[:, 2].mean() - s * X[:, 2].mean()  # vertical alignment
    ])

    # Align points.
    for point in reconstruction.points.values():
        p = s * A.dot(point.coordinates) + b
        point.coordinates = p.tolist()

    # Align cameras.
    for shot in reconstruction.shots.values():
        R = shot.pose.get_rotation_matrix()
        t = np.array(shot.pose.translation)
        Rp = R.dot(A.T)
        tp = -Rp.dot(b) + s * t
        try:
            shot.pose.set_rotation_matrix(Rp)
            shot.pose.translation = list(tp)
        except:
            logger.debug("unable to transform reconstruction!")

    return reconstruction


def align_reconstructions_to_hlf(reconstructions, data):
    # 1. load list of hlf coordinates on floor plan
    hlf_list = data.load_hlf_list()
    logger.debug("hlf_list has {} entries".format(len(hlf_list)))

    # 2. load list of images detected with hlf
    hlf_det_list = data.load_hlf_det_list()
    logger.debug("hlf_det_list has {} entries".format(len(hlf_det_list)))

    # 3. for each reconstruction, attempt to auto discover gps
    for recon in reconstructions:
        logger.debug("recon has {} shots {}".format(len(recon.shots), sorted(recon.shots)))

        det_list = []
        img_list = []
        gt_list = []

        # second, get all shot ids in this recon that has hlf detection
        for shot_id in hlf_det_list:
            if shot_id in recon.shots:
                o = recon.shots[shot_id].pose.get_origin()
                det_list.append([o[0], o[1]])
                img_list.append(shot_id)

        if len(det_list) < 8:
            logger.debug("recon has {} door detections, too few to perform alignment".format(len(det_list)))
            continue

        logger.debug("det_list has {} entries".format(len(det_list)))
        #logger.debug("img_list {}".format(img_list))

        for i in range(len(det_list)):
            # change this if we add ground truth manually
            gt_list.append(-1)

        matches = csfm.run_hlf_matcher(hlf_list, det_list, gt_list,
                                       data.config['reconstruction_scale_factor'])

        for i in matches.keys():
            logger.debug("{} => {}".format(img_list[i], hlf_list[matches[i]]))


def align_reconstruction_segments(data, graph, reconstruction, recon_gps_points):
    """
    align reconstruction to gps points. if more than 2 gps points, alignment is done segment-wise,
    i.e. two 2 gps points are used at a time. each segment is returned as a separate Reconstruction
    object. 3d points seen by each shot are assigned to the segment the shot belongs to.
    """
    segments = []

    gps_shot_ids = sorted(recon_gps_points.keys())

    for i in range(len(gps_shot_ids) - 1):
        X, Xp = [], []
        onplane, verticals = [], []

        for j in range(2):
            shot_id = gps_shot_ids[i+j]
            X.append(reconstruction.shots[shot_id].pose.get_origin())
            Xp.append(recon_gps_points[shot_id])

            R = reconstruction.shots[shot_id].pose.get_rotation_matrix()
            onplane.append(R[0,:])
            onplane.append(R[2,:])
            verticals.append(R[1,:])

        X = np.array(X)
        Xp = np.array(Xp)

        # Estimate ground plane.
        p = multiview.fit_plane(X - X.mean(axis=0), onplane, verticals)
        Rplane = multiview.plane_horizontalling_rotation(p)
        X = Rplane.dot(X.T).T

        # Estimate 2d similarity to align to pdr predictions
        T = tf.affine_matrix_from_points(X.T[:2], Xp.T[:2], shear=False)
        s = np.linalg.det(T[:2, :2]) ** 0.5
        A = np.eye(3)
        A[:2, :2] = T[:2, :2] / s
        A = A.dot(Rplane)
        b = np.array([
            T[0, 2],
            T[1, 2],
            Xp[:, 2].mean() - s * X[:, 2].mean()  # vertical alignment
        ])

        shot_ids = sorted(reconstruction.shots.keys())
        if i == 0:
            # in first iteration, we transform from first shot of recon
            start_index = _shot_id_to_int(shot_ids[0])
        else:
            start_index = _shot_id_to_int(gps_shot_ids[i])

        if i == len(gps_shot_ids)-2:
            # in last iteration, we transform until last shot of recon
            end_index = _shot_id_to_int(shot_ids[-1])
        else:
            # subtract 1 at the end, since gps_shots_ids[i+1] will be transformed in the next iteration
            end_index = _shot_id_to_int(gps_shot_ids[i+1]) - 1

        segment = extract_segment(reconstruction, start_index, end_index, graph, data.load_camera_models())
        apply_similarity(segment, s, A, b)

        segment.alignment.aligned = True
        segment.alignment.num_correspondences = 2

        segments.append(segment)

    return segments


def diff_recon_preint(im1, im2, reconstruction, pdr_shots_dict):
    """
    compare rel recon rotation of two images to that of imu gyro preintegration,
    if they are not close, the recon of the images is considered to be an erroneous
    """
    # calculate relative rotation from preintegrated gyro input
    preint_im1_rot = cv2.Rodrigues(np.asarray([pdr_shots_dict[im1][6], pdr_shots_dict[im1][7], pdr_shots_dict[im1][8]]))[0]
    preint_im2_rot = cv2.Rodrigues(np.asarray([pdr_shots_dict[im2][6], pdr_shots_dict[im2][7], pdr_shots_dict[im2][8]]))[0]
    preint_rel_rot = np.dot(preint_im2_rot, preint_im1_rot.T)

    # convert this rotation from sensor frame to camera frame
    b_to_c = np.asarray([1, 0, 0, 0, 0, -1, 0, 1, 0]).reshape(3, 3)
    preint_rel_rot = cv2.Rodrigues(b_to_c.dot(cv2.Rodrigues(preint_rel_rot)[0].ravel()))[0]

    R1 = reconstruction.shots[im1].pose.get_rotation_matrix()
    R2 = reconstruction.shots[im2].pose.get_rotation_matrix()
    recon_rel_rot = np.dot(R1, R2.T)

    #logger.debug("preint {}".format(np.degrees(_rotation_matrix_to_euler_angles(preint_rel_rot))))
    #logger.debug("recon {}".format(np.degrees(_rotation_matrix_to_euler_angles(recon_rel_rot))))

    diff_rot = np.dot(preint_rel_rot, recon_rel_rot.T)

    return np.linalg.norm(cv2.Rodrigues(diff_rot)[0].ravel())

    # instead of returning norm of the axis-angle difference, we return only the Y component which is
    # more relevant to the mapping use case we have
    #return abs(_rotation_matrix_to_euler_angles(diff_rot)[1])


def point_copy(point):
    c = types.Point()
    c.id = point.id
    c.color = point.color
    c.coordinates = point.coordinates.copy()
    c.reprojection_errors = point.reprojection_errors.copy()
    return c


def extract_segment(reconstruction, start_index, end_index, graph, cameras):
    recon_points = set(reconstruction.points)
    segment_points = set()

    segment = types.Reconstruction()
    segment.cameras = cameras

    for shot in reconstruction.shots.values():
        if start_index <= _shot_id_to_int(shot.id) <= end_index:
            segment.add_shot(reconstruction.shots[shot.id])
            segment_points = segment_points | (recon_points & set(graph[shot.id]))

    # need to copy point because a point may belong to more than one segment
    for point_id in segment_points:
        segment.add_point(point_copy(reconstruction.points[point_id]))

    return segment


def prune_reconstructions_by_pdr(reconstructions, data, graph):
    segments = []

    recon_quality = 0
    avg_segment_size = 0
    ratio_shots_in_min_recon = 0
    speedup = 1.0

    total_shots_cnt = 0
    total_bad_shots_cnt = 0

    cameras = data.load_camera_models()
    pdr_shots_dict = data.load_pdr_shots()
    culling_dict = data.load_culling_dict()

    for reconstruction in reconstructions:
        total_shots_cnt += len(reconstruction.shots)
        new_segments, bad_shots_cnt = prune_reconstruction_by_pdr(reconstruction, pdr_shots_dict, culling_dict, graph, cameras)
        total_bad_shots_cnt += bad_shots_cnt

        segments.extend(new_segments)

    if len(segments) > 0:
        recon_quality = 100-int(total_bad_shots_cnt*100/total_shots_cnt)
        logger.info("Recon Quality - {}".format(recon_quality))

        segments_shots_cnt = 0
        for segment in segments:
            segments_shots_cnt += len(segment.shots)
        avg_segment_size = segments_shots_cnt/len(segments)
        logger.info("Average good segment size - {}".format(int(avg_segment_size)))

        ratio_shots_in_min_recon = (segments_shots_cnt/total_shots_cnt)
        logger.info("Percentage of shots in good segments - {}%".format(int(100*ratio_shots_in_min_recon)))

        # PDR predicts 20 frames at a time. so if average segment is 40 then we have a 40/20 = 2.0x speed up.
        # the actual speed up depends on other factors and likely lower. avg_segment_size is capped to 100
        # below, because we predicts at most 100 frames at a time
        speedup = 1.0 / (ratio_shots_in_min_recon / math.floor(min(avg_segment_size, 100.0)/20.0)
                         + (1.0 - ratio_shots_in_min_recon))
        logger.info("Estimated speedup Hybrid vs PDR - {:2.1f}x".format(speedup))

    # for gps picker tool, calculate and save a recon quality factor. pdr/hybrid will be based on it.
    data.save_recon_quality(recon_quality, avg_segment_size, ratio_shots_in_min_recon, speedup)

    return segments


def prune_reconstruction_by_pdr(reconstruction, pdr_shots_dict, culling_dict, graph, cameras):
    segments = []
    suspicious_images = []

    shot_ids = sorted(reconstruction.shots)
    for shot_id in shot_ids:
        next_shot_id = _next_shot_id(shot_id)
        prev_shot_id = _prev_shot_id(shot_id)

        origin_curr = reconstruction.shots[shot_id].pose.get_origin()

        if next_shot_id in reconstruction.shots and _is_no_culling_in_between(culling_dict, shot_id, next_shot_id):
            vector_next = reconstruction.shots[next_shot_id].pose.get_origin() - origin_curr

            # check 1: compare rotations predicted by pre-integration and that of the reconstruction
            geo_diff = diff_recon_preint(shot_id, next_shot_id, reconstruction, pdr_shots_dict)

            if geo_diff > np.radians(HEADING_THRESH): # 10 degrees
                logger.debug("{} {} rotation diff {} degrees".format(shot_id, next_shot_id, np.degrees(geo_diff)))
                suspicious_images.append(shot_id)
                continue

            # check 2: check absolute height of current frame and difference between neighboring frames
            height_diff = abs(vector_next[2])
            if abs(vector_next[2]) > REL_HEIGHT_THRESH or abs(origin_curr[2]) > ABS_HEIGHT_THRESH:
                logger.debug("{} {} height {} diff {} meters".format(shot_id, next_shot_id, origin_curr[2], height_diff))
                suspicious_images.append(shot_id)
                continue

            # check 3: check distance traveled between neighboring frames - we know frames are taken 0.5s apart,
            # and physically the camera should move around 0.5 meters at normal walking speed. but we set the
            # threshold somewhat higher to account for inaccuracies of pdr.
            distance = np.linalg.norm(vector_next)
            if distance > DISTANCE_THRESH:
                logger.debug("{} {} coords distance {} meters".format(shot_id, next_shot_id, distance))
                suspicious_images.append(shot_id)
                continue

            # check 4: we know that frames are 0.5s apart, which is not enough time for a human to turning around
            # at normal walking speed, it's unlikely to have any significant movement that's forward then backward
            if prev_shot_id in reconstruction.shots and _is_no_culling_in_between(culling_dict, shot_id, prev_shot_id):
                vector_prev = origin_curr - reconstruction.shots[prev_shot_id].pose.get_origin()

                angle = _vector_angle(vector_next, vector_prev)
                if angle > np.radians(ANGLE_THRESH):
                    d_next = np.linalg.norm(vector_next)
                    d_prev = np.linalg.norm(vector_prev)

                    if d_next > MOVEMENT_THRESH and d_prev > MOVEMENT_THRESH and \
                            (d_next > SIG_MOVEMENT_THRESH or d_prev > SIG_MOVEMENT_THRESH):
                        logger.debug("{} {} {} angle = {} d_prev = {} d_next {}".
                                     format(prev_shot_id, shot_id, next_shot_id,
                                            np.degrees(angle), d_prev, d_next))
                        suspicious_images.append(shot_id)
                        continue

    # now remove suspicious images and images around them
    remove_images = set()
    for shot_id in suspicious_images:
        index = _shot_id_to_int(shot_id)
        for i in range(index - NEIGHBOR_RADIUS, index + NEIGHBOR_RADIUS + 1):
            remove_images.add(_int_to_shot_id(i))

    # TODO: is it necessary to search for and retriangulate/remove points in removed images?
    for shot_id in sorted(remove_images):
        if shot_id in reconstruction.shots:
            #logger.info("removing image: {}".format(shot_id))
            reconstruction.shots.pop(shot_id, None)

    # find sequential frames longer than MIN_RECON_SIZE to form new recons. the sequence
    # is allowed to have holes no larger than MAX_HOLE_SIZE
    shot_ids = sorted(reconstruction.shots)

    if len(shot_ids) < MIN_RECON_SIZE/MAX_HOLE_SIZE:
        return segments, len(suspicious_images)

    first_index = _shot_id_to_int(shot_ids[0])
    last_index = _shot_id_to_int(shot_ids[-1])

    start_index = end_index = first_index
    while True:
        while True:
            found = False
            for i in range(end_index + MAX_HOLE_SIZE, end_index, -1):
                if _int_to_shot_id(i) in shot_ids:
                    end_index = i
                    found = True
                    break

            if not found:
                break

        if (end_index - start_index + 1) >= MIN_RECON_SIZE:
            logger.debug("extract new recon from {} to {}".format(start_index, end_index))
            segment = extract_segment(reconstruction, start_index, end_index, graph, cameras)
            segments.append(segment)

        found = False
        for i in range(end_index + MAX_HOLE_SIZE, last_index - MIN_RECON_SIZE + 1):
            if _int_to_shot_id(i) in shot_ids:
                start_index = end_index = i
                found = True
                break

        if not found:
            break

    return segments, len(suspicious_images)


def apply_similarity(reconstruction, s, A, b):
    """Apply a similarity (y = s A x + b) to a reconstruction.

    :param reconstruction: The reconstruction to transform.
    :param s: The scale (a scalar)
    :param A: The rotation matrix (3x3)
    :param b: The translation vector (3)
    """
    # Align points.
    for point in reconstruction.points.values():
        Xp = s * A.dot(point.coordinates) + b
        point.coordinates = Xp.tolist()

    # Align cameras.
    for shot in reconstruction.shots.values():
        R = shot.pose.get_rotation_matrix()
        t = np.array(shot.pose.translation)
        Rp = R.dot(A.T)
        tp = -Rp.dot(b) + s * t
        shot.pose.set_rotation_matrix(Rp)
        shot.pose.translation = list(tp)


def get_origin_no_numpy_opencv(rotation, translation):
    def S(n):
        return [[0, -n[2], n[1]],
                [n[2], 0, -n[0]],
                [-n[1], n[0], 0]]

    def S_sq(n):
        return [[-n[1]**2-n[2]**2, n[0]*n[1], n[0]*n[2]],
                [n[0]*n[1], -n[0]**2-n[2]**2, n[1]*n[2]],
                [n[0]*n[2], n[1]*n[2], -n[0]**2-n[1]**2]]

    def norm(r):
        return math.sqrt(r[0]**2+r[1]**2+r[2]**2)

    def eye():
        return [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]

    theta = norm(rotation)
    if theta > 1e-30:
        n = rotation/theta
        tmp1 = [[j * math.sin(theta) for j in i] for i in S(n)]
        tmp2 = [[j * (1-math.cos(theta)) for j in i] for i in S_sq(n)]
    else:
        theta2 = theta**2
        tmp1 = [[j * (1-theta2/6.) for j in i] for i in S(rotation)]
        tmp2 = [[j * (0.5-theta2/24.) for j in i] for i in S_sq(rotation)]

    eye = eye()
    tmp = [[sum(x) for x in zip(tmp1[i], tmp2[i])] for i in range(3)]
    R = [[sum(x) for x in zip(eye[i], tmp[i])] for i in range(3)]

    origin = [-R[0][0] * translation[0] - R[1][0] * translation[1] - R[2][0] * translation[2],
              -R[0][1] * translation[0] - R[1][1] * translation[1] - R[2][1] * translation[2],
              -R[0][2] * translation[0] - R[1][2] * translation[1] - R[2][2] * translation[2]]
    return origin


def align_reconstruction_no_numpy(reconstruction, anchor_points_dict):
    """
    align reconstruction to anchor points. to be ported and used in gps picker

    anchor points are either gps points/trusted shots, or pdr points. alignment is done segment-wise,
    using 2 gps points at a time

    note only the x/y coordinates are aligned, not the full camera pose (gps picker don't need that)

    :param anchor_points_dict:
    :param reconstruction:
    :return:
    """
    modified_shots_dict = {}
    all_anchor_shot_ids = sorted(anchor_points_dict.keys())
    for i in range(len(all_anchor_shot_ids) - 1):
        anchor_coords = []
        recon_coords = []

        for j in range(2):
            shot_id = all_anchor_shot_ids[i+j]
            anchor_coords.append(anchor_points_dict[shot_id])
            o = get_origin_no_numpy_opencv(reconstruction.shots[shot_id].pose.rotation,
                                           reconstruction.shots[shot_id].pose.translation)

            recon_coords.append(o)

        s, A, b = get_affine_transform_2d_no_numpy(anchor_coords, recon_coords)

        start_shot_id = all_anchor_shot_ids[i]
        end_shot_id = all_anchor_shot_ids[i+1]

        # in first iteration, we transform from first shot of recon
        # in last iteration, we transform until last shot of recon
        shot_ids = sorted(reconstruction.shots.keys())
        if i == 0:
            start_shot_id = shot_ids[0]

        if i == len(anchor_points_dict)-2:
            end_shot_id = shot_ids[-1]

        new_dict = {}

        start_index = _shot_id_to_int(start_shot_id)
        end_index = _shot_id_to_int(end_shot_id)

        # transform pdr shots
        for i in range(start_index, end_index + 1):
            shot_id = _int_to_shot_id(i)

            if shot_id in reconstruction.shots:
                X = get_origin_no_numpy_opencv(reconstruction.shots[shot_id].pose.rotation,
                                               reconstruction.shots[shot_id].pose.translation)
                A_dot_X = [A[0][0] * X[0] + A[0][1] * X[1] + A[0][2] * X[2],
                           A[1][0] * X[0] + A[1][1] * X[1] + A[1][2] * X[2],
                           A[2][0] * X[0] + A[2][1] * X[1] + A[2][2] * X[2]]
                Xp = [i * s + j for i, j in zip(A_dot_X, b)]
                new_dict[shot_id] = [Xp[0], Xp[1], Xp[2]]

        modified_shots_dict.update(new_dict)

    return modified_shots_dict


def update_gps_picker(curr_gps_points_dict, pdr_shots_dict, scale_factor, num_extrapolation):
    """
    this routine is intended to be ported and used in gps picker

    globally align pdr path to current set of gps shots. return pdr predictions for
    the first x + num_extrapolation shots, where x is the largest sequence number in
    current gps shots.

    when there is no gps point, no alignment is done, instead, this function returns
    a scaled version of the pdr path, so that gps pickers can see it and easily work
    with it on the floor plan

    when there is one gps point, no alignment is done, instead, this function simply
    shifts the pdr path so it overlaps with that single gps point

    :param curr_gps_points_dict:
    :param pdr_shots_dict:
    :param scale_factor:
    :param num_extrapolation:
    :return:
    """
    pdr_predictions_dict = {}

    scaled_pdr_shots_dict = {}
    for shot_id in pdr_shots_dict:
        scaled_pdr_shots_dict[shot_id] = (pdr_shots_dict[shot_id][0]/(scale_factor * 0.3048),
                                          pdr_shots_dict[shot_id][1]/(scale_factor * 0.3048), 0)

    if len(curr_gps_points_dict) == 0:
        shot_id = _int_to_shot_id(0)
        pdr_predictions_dict[shot_id] = (2000, 2000, 0)
    elif len(curr_gps_points_dict) == 1:
        shot_id = list(curr_gps_points_dict.keys())[0]
        offset = tuple(np.subtract(curr_gps_points_dict[shot_id],
                                   scaled_pdr_shots_dict[shot_id]))
        num = _shot_id_to_int(shot_id) + num_extrapolation

        for i in range(num):
            shot_id = _int_to_shot_id(i)
            pdr_predictions_dict[shot_id] = tuple(map(sum, zip(offset, scaled_pdr_shots_dict[shot_id])))
    else:
        all_predictions_dict = update_pdr_global_2d(curr_gps_points_dict, pdr_shots_dict, scale_factor, False)

        sorted_gps_ids = sorted(curr_gps_points_dict.keys(), reverse=True)
        num = _shot_id_to_int(sorted_gps_ids[0]) + num_extrapolation

        for i in range(num):
            shot_id = _int_to_shot_id(i)
            pdr_predictions_dict[shot_id] = all_predictions_dict[shot_id]

    return pdr_predictions_dict


def update_gps_picker_hybrid(curr_gps_points_dict, reconstructions, pdr_shots_dict, scale_factor, num_extrapolation):
    """
    this routine is intended to be ported and used in gps picker

    :param curr_gps_points_dict:
    :param reconstructions:
    :param pdr_shots_dict:
    :param scale_factor:
    :param num_extrapolation:
    :return:
    """
    aligned_shots_dict = curr_gps_points_dict.copy()

    if len(curr_gps_points_dict) == 0:
        # with no gps point, this routine shouldn't be called. we simply place shot 0 at an arbitrary
        # point for debugging.
        return {}, {_int_to_shot_id(0): (2000, 2000, 0)}
    elif len(curr_gps_points_dict) == 1:
        gps_shot_id = list(curr_gps_points_dict.keys())[0]

        scaled_shots_dict = {}
        num = min(len(pdr_shots_dict) - 1, _shot_id_to_int(gps_shot_id) + 50)
        for i in range(num):
            shot_id = _int_to_shot_id(i)
            scaled_shots_dict[shot_id] = (pdr_shots_dict[shot_id][0] / (scale_factor * 0.3048),
                                          pdr_shots_dict[shot_id][1] / (scale_factor * 0.3048), 0)

        offset = tuple(np.subtract(curr_gps_points_dict[gps_shot_id],
                                   scaled_shots_dict[gps_shot_id]))

        init_shots_dict = {}

        for shot_id in scaled_shots_dict:
            init_shots_dict[shot_id] = tuple(map(sum, zip(offset, scaled_shots_dict[shot_id])))

        return {}, init_shots_dict

    # we modify the alignment flag each time we invoke this routine.
    # so return the flag to default first.
    for recon in reconstructions:
        recon.alignment.aligned = False

    # init pdr predictions
    pdr_predictions_dict = update_pdr_global_2d(curr_gps_points_dict, pdr_shots_dict, scale_factor, False)

    # align recons to gps points and/or trusted shots
    while True:
        can_align = False
        for recon in reconstructions:
            if recon.alignment.aligned or len(recon.shots) < MIN_RECON_SIZE:
                continue

            recon_gps_points = {}
            recon_trusted_shots = {}

            # match gps points to this recon
            for shot_id in recon.shots:
                if shot_id in curr_gps_points_dict:
                    recon_gps_points[shot_id] = curr_gps_points_dict[shot_id]

            # find trusted shots on this recon if not enough gps points
            if len(recon_gps_points) < 2:
                recon_shot_ids = sorted(recon.shots)

                if recon_shot_ids[0] not in curr_gps_points_dict and \
                        _prev_shot_id(recon_shot_ids[0]) in aligned_shots_dict:
                    recon_trusted_shots[recon_shot_ids[0]] = pdr_predictions_dict[recon_shot_ids[0]][:3]

                if recon_shot_ids[-1] not in curr_gps_points_dict and \
                        _next_shot_id(recon_shot_ids[-1]) in aligned_shots_dict:
                    recon_trusted_shots[recon_shot_ids[-1]] = pdr_predictions_dict[recon_shot_ids[-1]][:3]

            if len(recon_gps_points) + len(recon_trusted_shots) >= 2:
                # combine trusted shots with gps points
                recon_trusted_shots.update(recon_gps_points)

                shots_dict = align_reconstruction_no_numpy(recon, recon_trusted_shots)
                aligned_shots_dict.update(shots_dict)

                # update pdr predictions based on aligned shots so far
                pdr_predictions_dict = update_pdr_global_2d(aligned_shots_dict, pdr_shots_dict, scale_factor, False)

                recon.alignment.aligned = True
                can_align = True
                break

        if not can_align:
            break

    # update pdr predictions based on aligned shots so far
    pdr_predictions_dict = update_pdr_global_2d(aligned_shots_dict, pdr_shots_dict, scale_factor, False)

    '''
    Code below constructs two dictionaries: before_dict, which contains shots before last gps point, and
    after_dict, which contains shot predictions after last gps point (at most num_extrapolation shots).

    before_dict consists of shots from aligned_shots_dict, or shots from pdr_predictions_dict if they are
    not aligned.

    after_dict shots are composed of 
        1) up to num_extrapolation aligned shots (may contain small holes), or
        2) up to PDR_TRUST_SIZE of pdr predictions when there is no remaining aligned shots, or
        3) in case of less than PDR_TRUST_SIZE aligned shots, pdr predictions are padded at the end
    '''

    last_gps_idx = _shot_id_to_int(sorted(curr_gps_points_dict.keys(), reverse=True)[0])
    last_aligned_idx = _shot_id_to_int(sorted(aligned_shots_dict.keys(), reverse=True)[0])

    if num_extrapolation == -1:
        num_extrapolation = 999999

    if last_aligned_idx > last_gps_idx:
        num_aligned_remaining = last_aligned_idx - last_gps_idx
        if num_extrapolation < num_aligned_remaining < 2*num_extrapolation:
            # avoid situation where only a small number of aligned shots remaining
            max_shot_idx = last_gps_idx + num_aligned_remaining/2
        elif num_aligned_remaining < PDR_TRUST_SIZE:
            # in this case we will have a few aligned shots followed by up to PDR_TRUST_SIZE predictions
            max_shot_idx = last_gps_idx + PDR_TRUST_SIZE
        else:
            # in this case we will have up to num_extrapolation aligned shots
            max_shot_idx = last_gps_idx + min(num_extrapolation, num_aligned_remaining)
    else:
        # in this case we will have PDR_TRUST_SIZE predictions
        max_shot_idx = last_gps_idx + PDR_TRUST_SIZE

    max_shot_idx = min(len(pdr_shots_dict) - 1, max_shot_idx)

    before_dict = {}
    after_dict = {}

    for idx in range(last_gps_idx+1):
        shot_id = _int_to_shot_id(idx)
        if shot_id in aligned_shots_dict:
            before_dict[shot_id] = aligned_shots_dict[shot_id]
        else:
            before_dict[shot_id] = pdr_predictions_dict[shot_id][:3]

    curr_shot_idx = last_gps_idx + 1

    while curr_shot_idx <= max_shot_idx:
        shot_id = _int_to_shot_id(curr_shot_idx)
        if shot_id in aligned_shots_dict:
            after_dict[shot_id] = aligned_shots_dict[shot_id]
        else:
            after_dict[shot_id] = pdr_predictions_dict[shot_id][:3]

        curr_shot_idx += 1

    return before_dict, after_dict


def update_gps_picker_hybrid_old(curr_gps_points_dict, reconstructions, pdr_shots_dict, scale_factor):
    """
    this routine is intended to be ported and used in gps picker

    this routine facilitates gps picking using a hybrid (sfm+pdr) approach. the goal is to reduce the
    number of gps points that need to be picked manually. to achieve this goal, pdr is used to auto
    fill small holes and gaps (see definitions of 'hole' and 'gap' below).

        def 1: a 'hole' is consecutive frames missing in the middle of a recon. e.g. if a recon contains
        images 100-150 and 161-200, then 151-160 is a 10 image 'hole'. a hole is considered small if it
        is less than PDR_TRUST_SIZE long.

        def 2: the 'gap' of an unaligned recon is the min distance of its lowest numbered shot to any
        aligned shot ahead of it. e.g. if shots 0-200 have been aligned, and an unaligned recon has shots
        231-300, then the 'gap' is 30. a gap is considered small if it is less than PDR_TRUST_SIZE long.

        def 3: an 'aligned recon" is a recon with at least two shots being either gps point or 'trusted
        shot'.

        def 4: an 'aligned shot' is any shot belonging to an aligned recon, or otherwise is itself a gps
        point.

        def 5: a 'trusted shot' is a shot very close (distance = 1) to an aligned shot.

    algorithm:

    1) recons with at least two gps points or trusted shots are aligned to become aligned recons. if
    there are more than two gps points in a recon, alignment is done segment-wise (i.e., two points at
    a time).

    2) update pdr predictions using aligned shots

    3) start_shot = current_shot = lowest numbered shot that is not aligned

    4) while true:

            a. if current shot is an aligned shot, then shots between start_shot and current_shot-1 are
            now aligned (because start_shot and current_shot-1 are both 'trusted'). repeat from step 2.

            b. if current shot is in a recon with size > MIN_RECON_SIZE, then make predictions for shots
            in this recon using pdr, plus predictions for shots between start_shot to current_shot-1,
            using pdr. break.

            c. if current_shot - start_shot >= PDR_TRUST_SIZE, then make predictions for shots between
            start_shot and current_shot-1 (PDR_TRUST_SIZE shots in total) using pdr. break.

            d. current_shot++

    5) return all aligned shots, as well as predictions from step 4. predictions from step 4 will be
    marked as such and the gps picker UI should present them differently than aligned shots.

    :param curr_gps_points_dict:
    :param reconstructions:
    :param pdr_shots_dict:
    :param scale_factor:
    :param num_extrapolation:
    :return:
    """
    aligned_shots_dict = curr_gps_points_dict.copy()
    predicted_shots_dict = {}

    if len(curr_gps_points_dict) == 0:
        # with no gps point, this routine shouldn't be called. we simply place shot 0 at an arbitrary
        # point for debugging.
        predicted_shots_dict[_int_to_shot_id(0)] = (2000, 2000, 0)
        return {}, predicted_shots_dict
    elif len(curr_gps_points_dict) == 1:
        gps_shot_id = list(curr_gps_points_dict.keys())[0]
        long_unaligned_recon = None

        for recon in reconstructions:
            if gps_shot_id in recon.shots and len(recon.shots) > MIN_RECON_SIZE:
                long_unaligned_recon = recon
                # break for loop
                break

        scaled_shots_dict = {}
        if long_unaligned_recon:
            for shot_id in long_unaligned_recon.shots:
                o = long_unaligned_recon.shots[shot_id].pose.get_origin()
                scaled_shots_dict[shot_id] = (o[0] / (scale_factor * 0.3048),
                                              o[1] / (scale_factor * 0.3048), 0)
        else:
            num = _shot_id_to_int(gps_shot_id) + PDR_TRUST_SIZE
            for i in range(num):
                shot_id = _int_to_shot_id(i)
                scaled_shots_dict[shot_id] = (pdr_shots_dict[shot_id][0] / (scale_factor * 0.3048),
                                              pdr_shots_dict[shot_id][1] / (scale_factor * 0.3048), 0)

        offset = tuple(np.subtract(curr_gps_points_dict[gps_shot_id],
                                   scaled_shots_dict[gps_shot_id]))

        for shot_id in scaled_shots_dict:
            predicted_shots_dict[shot_id] = tuple(map(sum, zip(offset, scaled_shots_dict[shot_id])))

        return {}, predicted_shots_dict

    # we modify the alignment flag each time we invoke this routine.
    # so return the flag to default first.
    for recon in reconstructions:
        recon.alignment.aligned = False

    # init pdr predictions
    pdr_predictions_dict = update_pdr_global_2d(curr_gps_points_dict, pdr_shots_dict, scale_factor, False)

    # align recons to gps points and/or trusted shots
    while True:
        can_align = False
        for recon in reconstructions:
            if recon.alignment.aligned:
                continue

            recon_gps_points = {}
            recon_trusted_shots = {}

            # match gps points to this recon
            for shot_id in recon.shots:
                if shot_id in curr_gps_points_dict:
                    recon_gps_points[shot_id] = curr_gps_points_dict[shot_id]

            # find trusted shots on this recon if not enough gps points
            if len(recon_gps_points) < 2:
                recon_shot_ids = sorted(recon.shots)

                if recon_shot_ids[0] not in curr_gps_points_dict and \
                        _prev_shot_id(recon_shot_ids[0]) in aligned_shots_dict:
                    recon_trusted_shots[recon_shot_ids[0]] = pdr_predictions_dict[recon_shot_ids[0]][:3]

                if recon_shot_ids[-1] not in curr_gps_points_dict and \
                        _next_shot_id(recon_shot_ids[-1]) in aligned_shots_dict:
                    recon_trusted_shots[recon_shot_ids[-1]] = pdr_predictions_dict[recon_shot_ids[-1]][:3]

            if len(recon_gps_points) + len(recon_trusted_shots) >= 2:
                # combine trusted shots with gps points
                recon_trusted_shots.update(recon_gps_points)

                shots_dict = align_reconstruction_no_numpy(recon, recon_trusted_shots)
                aligned_shots_dict.update(shots_dict)

                # update pdr predictions based on aligned shots so far
                pdr_predictions_dict = update_pdr_global_2d(aligned_shots_dict, pdr_shots_dict, scale_factor, False)

                recon.alignment.aligned = True
                can_align = True
                break

        if not can_align:
            break

    # find first unaligned shot
    for i in range(len(pdr_shots_dict) + 1):
        if _int_to_shot_id(i) not in aligned_shots_dict:
            # break for loop
            break

    if i == len(pdr_shots_dict):
        # all shots have been aligned
        logger.debug("all shots aligned")
        return aligned_shots_dict, {}

    logger.debug("first unaligned = {}".format(i))

    start_shot_idx = i
    current_shot_idx = start_shot_idx + 1

    # make predictions
    while True:

        if _int_to_shot_id(current_shot_idx) in aligned_shots_dict:
            # start_shot and current_shot-1 are trusted shots. pdr predictions in between are counted as align shots
            for i in range(start_shot_idx, current_shot_idx):
                aligned_shots_dict[_int_to_shot_id(i)] = pdr_predictions_dict[_int_to_shot_id(i)][:3]

            logger.debug("filled hole {}-{}".format(start_shot_idx, current_shot_idx-1))

            # update pdr predictions based on aligned shots so far
            pdr_predictions_dict = update_pdr_global_2d(aligned_shots_dict, pdr_shots_dict, scale_factor, False)

            # continue to find lowest numbered unaligned shot
            for i in range(current_shot_idx + 1, len(pdr_shots_dict) + 1):
                if _int_to_shot_id(i) not in aligned_shots_dict:
                    # break for loop
                    break

            logger.debug("first unaligned = {}".format(i))
            if i == len(pdr_shots_dict):
                # all shots have been aligned
                logger.debug("all shots aligned")
                return aligned_shots_dict, {}

            start_shot_idx = i
            current_shot_idx = start_shot_idx + 1
            continue

        long_unaligned_recon = None
        for recon in reconstructions:
            if _int_to_shot_id(current_shot_idx) in recon.shots and \
                    len(recon.shots) > MIN_RECON_SIZE and not recon.alignment.aligned:
                long_unaligned_recon = recon
                # break for loop
                break

        if long_unaligned_recon:
            logger.debug("long unaligned recon found, current_shot_idx = {}".format(current_shot_idx))
            # first use pdr to fill the gap
            for i in range(start_shot_idx, current_shot_idx):
                predicted_shots_dict[_int_to_shot_id(i)] = pdr_predictions_dict[_int_to_shot_id(i)][:3]

            logger.debug("filled gap {}-{}".format(start_shot_idx, current_shot_idx-1))
            # then align the long reconstruction to pdr and add to predicted shots
            all_recon_shot_ids = sorted(long_unaligned_recon.shots.keys())
            anchor_points = {all_recon_shot_ids[0]: pdr_predictions_dict[all_recon_shot_ids[0]][:3],
                             all_recon_shot_ids[1]: pdr_predictions_dict[all_recon_shot_ids[1]][:3]}
            new_dict = align_reconstruction_no_numpy(long_unaligned_recon, anchor_points)
            predicted_shots_dict.update(new_dict)

            # break while loop
            break

        if current_shot_idx >= len(pdr_shots_dict) or current_shot_idx - start_shot_idx >= PDR_TRUST_SIZE:
            for i in range(start_shot_idx, current_shot_idx):
                predicted_shots_dict[_int_to_shot_id(i)] = pdr_predictions_dict[_int_to_shot_id(i)][:3]

            logger.debug("pdr prediction {}-{}".format(start_shot_idx, current_shot_idx-1))

            # break while loop
            break

        current_shot_idx += 1
        logger.debug("curr shot idx={}".format(current_shot_idx))

    return aligned_shots_dict, predicted_shots_dict


def apply_affine_transform(pdr_shots_dict, start_shot_id, end_shot_id, s, A, b, deviation, gps_shot_ids=[]):
    """Apply a similarity (y = s A x + b) to a reconstruction.

    :param pdr_shots_dict: all original pdr predictions
    :param start_shot_id: start shot id to perform transform
    :param end_shot_id: end shot id to perform transform
    :param s: The scale (a scalar)
    :param A: The rotation matrix (3x3)
    :param b: The translation vector (3)
    :param gps_shot_ids: gps shot ids the affine transform is based on
    :param deviation: a measure of how closely pdr predictions match gps points
    :return: pdr shots between start and end shot id transformed by s, A, b
    """
    new_dict = {}

    start_index = _shot_id_to_int(start_shot_id)
    end_index = _shot_id_to_int(end_shot_id)

    # transform pdr shots
    for i in range(start_index, end_index + 1):
        shot_id = _int_to_shot_id(i)
        dop = get_dop(shot_id, deviation, gps_shot_ids)

        if shot_id in pdr_shots_dict:
            Xp = s * A.dot(pdr_shots_dict[shot_id][0:3]) + b
            new_dict[shot_id] = [Xp[0], Xp[1], Xp[2], dop]
            #logger.info("new_dict {} = {} {} {} {}".format(shot_id, new_dict[shot_id][0], new_dict[shot_id][1], new_dict[shot_id][2], new_dict[shot_id][3]))

    return new_dict


def apply_affine_transform_no_numpy(pdr_shots_dict, start_shot_id, end_shot_id, s, A, b, deviation, gps_shot_ids=[]):
    """Apply a similarity (y = s A x + b) to a reconstruction.

    we avoid all numpy calls, to make it easier to port to Javascript for use in gps picker

    :param pdr_shots_dict: all original pdr predictions
    :param start_shot_id: start shot id to perform transform
    :param end_shot_id: end shot id to perform transform
    :param s: The scale (a scalar)
    :param A: The rotation matrix (3x3)
    :param b: The translation vector (3)
    :param gps_shot_ids: gps shot ids the affine transform is based on
    :param deviation: a measure of how closely pdr predictions match gps points
    :return: pdr shots between start and end shot id transformed by s, A, b
    """
    new_dict = {}

    start_index = _shot_id_to_int(start_shot_id)
    end_index = _shot_id_to_int(end_shot_id)

    # transform pdr shots
    for i in range(start_index, end_index + 1):
        shot_id = _int_to_shot_id(i)
        dop = get_dop(shot_id, deviation, gps_shot_ids)

        if shot_id in pdr_shots_dict:
            X = pdr_shots_dict[shot_id]
            A_dot_X = [A[0][0]*X[0] + A[0][1]*X[1] + A[0][2]*X[2],
                          A[1][0]*X[0] + A[1][1]*X[1] + A[1][2]*X[2],
                          A[2][0]*X[0] + A[2][1]*X[1] + A[2][2]*X[2]]
            Xp = [i*s + j for i, j in zip(A_dot_X, b)]
            new_dict[shot_id] = [Xp[0], Xp[1], Xp[2], dop]
            #logger.info("new_dict {} = {} {} {} {}".format(shot_id, new_dict[shot_id][0], new_dict[shot_id][1], new_dict[shot_id][2], new_dict[shot_id][3]))

    return new_dict


def get_affine_transform_2d(gps_coords, pdr_coords):
    """
    get affine transform between pdr an GPS coordinates (dim 2)

    """
    X = np.array(pdr_coords)
    Xp = np.array(gps_coords)

    # Estimate 2d similarity to align to GPS
    T = tf.affine_matrix_from_points(X.T[:2], Xp.T[:2], shear=False)
    s = np.linalg.det(T[:2, :2]) ** 0.5
    A = np.eye(3)
    A[:2, :2] = T[:2, :2] / s
    b = np.array([
        T[0, 2],
        T[1, 2],
        Xp[:, 2].mean() - s * X[:, 2].mean()  # vertical alignment
    ])

    return s, A, b


def get_affine_transform_2d_no_numpy(gps_coords, pdr_coords):
    """
    get affine transform between 2 pdr points and 2 GPS coordinates.
    this simplification applies when we have 2 pdr points to be aligned with 2 gps points. we will avoid
    numpy functions, so as to make the porting to Javascript easier.
    """
    diff_x = [i - j for i, j in zip(pdr_coords[1], pdr_coords[0])]
    diff_xp = [i - j for i, j in zip(gps_coords[1], gps_coords[0])]

    dot = diff_x[0] * diff_xp[0] + diff_x[1] * diff_xp[1]  # dot product
    det = diff_x[0] * diff_xp[1] - diff_x[1] * diff_xp[0]  # determinant
    theta = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

    A = [[math.cos(theta), -math.sin(theta), 0],
         [math.sin(theta), math.cos(theta), 0],
         [0, 0, 1]]
    s = math.sqrt((diff_xp[0]*diff_xp[0]+diff_xp[1]*diff_xp[1]+diff_xp[2]*diff_xp[2])/
                  (diff_x[0]*diff_x[0]+diff_x[1]*diff_x[1]+diff_x[2]*diff_x[2]))

    x1 = pdr_coords[1]
    a_dot_x1 = [A[0][0]*x1[0] + A[0][1]*x1[1] + A[0][2]*x1[2],
                  A[1][0]*x1[0] + A[1][1]*x1[1] + A[1][2]*x1[2],
                  A[2][0]*x1[0] + A[2][1]*x1[1] + A[2][2]*x1[2]]
    b = [i - j*s for i, j in zip(gps_coords[1], a_dot_x1)]

    return s, A, b


def get_affine_transform(gps_coords, pdr_coords):
    """
    get affine transform between pdr an GPS coordinates (dim 3)

    """
    # Compute similarity Xp = s A X + b
    X = np.array(pdr_coords)
    Xp = np.array(gps_coords)
    T = tf.superimposition_matrix(X.T, Xp.T, scale=True)

    A, b = T[:3, :3], T[:3, 3]
    s = np.linalg.det(A)**(1. / 3)
    A /= s
    return s, A, b


def get_dop(shot_id, deviation, gps_shot_ids):
    """
    get a 'dop' of the prediction

    :param shot_id:
    :param deviation:
    :param gps_shot_ids:
    :return:
    """
    if gps_shot_ids:
        shot_id_int = _shot_id_to_int(shot_id)

        distances = []
        for gps_id in gps_shot_ids:
            distances.append(abs(_shot_id_to_int(gps_id)-shot_id_int))

        # TODO: read default dop 200 from config
        dop = 200 + min(distances)*100*(1+deviation)
    else:
        dop = 200 * (1+deviation)

    return dop


def get_farthest_shots(shots_dict):
    """get two shots that are most far apart in physical distance"""
    distances_dict = {}
    shot_ids = shots_dict.keys()
    for (i, j) in combinations(shot_ids, 2):
        distances_dict[(i, j)] = np.linalg.norm(np.array(shots_dict[i]) - np.array(shots_dict[j]))

    (shot1, shot2) = max(distances_dict.items(), key=operator.itemgetter(1))[0]

    return {shot1: shots_dict[shot1], shot2: shots_dict[shot2]}, distances_dict[(shot1, shot2)]


def get_closest_shots(shot_id, aligned_shot_ids):
    """get two shots that are closest in sequence distance"""
    distances_dict = get_distance_to_aligned_shots(shot_id, aligned_shot_ids)

    # remove self if it's in the dict
    distances_dict.pop(shot_id, None)

    # sort aligned shot ids by absolute value
    return sorted(distances_dict, key=lambda k: abs(distances_dict[k]))


def get_distance_to_aligned_shots(shot_id, aligned_shot_ids):
    """distances of shot_id to shots that are gps-aligned"""
    distances_dict = {}
    for a_id in aligned_shot_ids:
        distances_dict[a_id] = _shot_id_to_int(shot_id) - _shot_id_to_int(a_id)

    return distances_dict


def min_distance_to_aligned_shots(shot_id, aligned_shot_ids, gps_points_dict):
    """min abs distance of shot_id to any shot that is gps-aligned"""
    if shot_id in gps_points_dict:
        return 0

    distances_dict = get_distance_to_aligned_shots(shot_id, aligned_shot_ids)
    return min(distances_dict.values(), key=abs)


def _shot_id_to_int(shot_id):
    """
    Returns: shot id to integer
    """
    tokens = shot_id.split(".")
    return int(tokens[0])


def _int_to_shot_id(shot_int):
    """
    Returns: integer to shot id
    """
    return str(shot_int).zfill(10) + ".jpg"


def _prev_shot_id(curr_shot_id):
    """
    Returns: previous shot id
    """
    return _int_to_shot_id(_shot_id_to_int(curr_shot_id) - 1)


def _next_shot_id(curr_shot_id):
    """
    Returns: next shot id
    """
    return _int_to_shot_id(_shot_id_to_int(curr_shot_id) + 1)


def _vector_angle(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)


def _is_no_culling_in_between(culling_dict, shot_id_1, shot_id_2):
    if not culling_dict:
        return True

    i1 = _shot_id_to_int(culling_dict[shot_id_1])
    i2 = _shot_id_to_int(culling_dict[shot_id_2])
    if abs(i1-i2) == 1:
        return True
    else:
        return False


def _rotation_matrix_to_euler_angles(R):
    """
    The result is the same as MATLAB except the order of the euler angles ( x and z are swapped ).
    https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def _euler_angles_to_rotation_matrix(theta):
    """
    Calculates 3x3 Rotation Matrix given euler angles.

    theta[0] - x, theta[1] - y, theta[2] - z

    theta must be tait bryan sxyz order
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R
