"""Tools to align a reconstruction to GPS and GCP data."""

import operator
import logging
import math
from collections import defaultdict
from itertools import combinations

import numpy as np

from opensfm import csfm
from opensfm import multiview
from opensfm import transformations as tf

logger = logging.getLogger(__name__)


def align_reconstruction(reconstruction, gcp, config):
    """Align a reconstruction with GPS and GCP data."""
    res = align_reconstruction_similarity(reconstruction, gcp, config)
    reconstruction.alignment.aligned = False
    if res:
        s, A, b = res
        if np.isfinite(s):
            apply_similarity(reconstruction, s, A, b)
            reconstruction.alignment.aligned = True


def align_reconstruction_segments(reconstruction, gcp, config, stride_len=6):
    """
    same as align_reconstruction but one segment (stride_len gps points) at a time,
    stride_len must >= 2
    """
    gps_shots = []
    for shot in reconstruction.shots.values():
        if shot.metadata.gps_dop != 999999.0:
            gps_shots.append(shot)

    filtered_gps_shots = get_filtered_shots(gps_shots, config)

    if len(filtered_gps_shots) <= stride_len:
        align_reconstruction(reconstruction, gcp, config)
        return

    filtered_gps_shots_dict = {}
    for shot in filtered_gps_shots:
        filtered_gps_shots_dict[shot.id] = shot

    all_shot_ids = sorted(filtered_gps_shots_dict.keys())
    num_of_strides = len(all_shot_ids) // stride_len
    align_method = config['align_method']
    for i in range(num_of_strides):
        next_shots = []
        for j in range(stride_len):
            next_shots.append(filtered_gps_shots_dict[all_shot_ids[i*stride_len+j]])

        if i == num_of_strides - 1:
            left = len(all_shot_ids) - num_of_strides*stride_len
            for k in range(left):
                next_shots.append(filtered_gps_shots_dict[all_shot_ids[num_of_strides*stride_len + k]])

        res = None
        if align_method == 'orientation_prior':
            res = get_sab_2d(next_shots, config)
        elif align_method == 'naive':
            res = get_sab_3d(reconstruction, next_shots, gcp, config)
        elif align_method == 'naive_and_orientation_prior':
            res = get_sab_3d(reconstruction, next_shots, gcp, config)
            if not res:
                res = get_sab_2d(next_shots, config)

        if res:
            s, A, b = res

            if i == 0:
                start_shot_ind = 0
            else:
                start_shot_ind = _shot_id_to_int(all_shot_ids[i*stride_len-1]) + 1

            if i == num_of_strides - 1:
                end_shot_ind = 999999
            else:
                end_shot_ind = _shot_id_to_int(all_shot_ids[(i+1)*stride_len-1])

            apply_similarity_segment(reconstruction, start_shot_ind, end_shot_ind, s, A, b)


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


def apply_similarity_segment(reconstruction, start_shot_ind, end_shot_ind, s, A, b):
    """
    applies similarity transform to shots between start/end_shot_id in
    the reconstruction. affect shot pose only, not points
    """
    logger.debug("apply_similarity_segment: start/end shot index {} {}".format(start_shot_ind, end_shot_ind))
    for shot in reconstruction.shots.values():
        if start_shot_ind <= _shot_id_to_int(shot.id) <= end_shot_ind:
            R = shot.pose.get_rotation_matrix()
            t = np.array(shot.pose.translation)
            Rp = R.dot(A.T)
            tp = -Rp.dot(b) + s * t
            shot.pose.set_rotation_matrix(Rp)
            shot.pose.translation = list(tp)


def align_reconstruction_similarity(reconstruction, gcp, config):
    """Align reconstruction with GPS and GCP data.

    Config parameter `align_method` can be used to choose the alignment method.
    Accepted values are
     - navie: does a direct 3D-3D fit
     - orientation_prior: assumes a particular camera orientation
    """
    align_method = config['align_method']
    if align_method == 'orientation_prior':
        return align_reconstruction_orientation_prior_similarity( reconstruction, config )
    elif align_method == 'naive':
        return align_reconstruction_naive_similarity( reconstruction, gcp, config )
    elif align_method == 'naive_and_orientation_prior':
        res = align_reconstruction_naive_similarity( reconstruction, gcp, config )
        if not res:
            res = align_reconstruction_orientation_prior_similarity( reconstruction, config )
        return res


def align_reconstruction_naive_similarity(reconstruction, gcp, config):
    """Align with GPS and GCP data using direct 3D-3D matches."""
    gps_shots = []
    for shot in reconstruction.shots.values():
        if shot.metadata.gps_dop != 999999.0:
            gps_shots.append(shot)

    if len(gps_shots) < 3:
        reconstruction.alignment.num_correspondences = 0
        logger.debug('Similarity alignment NOT attempted ( {0} Correspondences )'.format(len(gps_shots)))
        return

    filtered_shots = gps_shots
    if len(gps_shots) > 3:
        #filtered_shots = get_filtered_shots(gps_shots, config)
        filtered_shots = get_farthest_three_shots(gps_shots)

    reconstruction.alignment.num_correspondences = len(filtered_shots)
    logger.debug('Similarity alignment attempted on shots')
    for shot in filtered_shots:
        logger.debug("{}".format(shot.id))

    return get_sab_3d(reconstruction, filtered_shots, gcp, config)


def get_sab_3d(reconstruction, filtered_shots, gcp, config):
    X, Xp = [], []

    # Get Ground Control Point correspondences
    if gcp and config['align_use_gcp']:
        triangulated, measured = triangulate_all_gcp(reconstruction, gcp)
        X.extend(triangulated)
        Xp.extend(measured)

    # Get camera center correspondences
    if config['align_use_gps']:
        for shot in filtered_shots:
            X.append(shot.pose.get_origin())
            Xp.append(shot.metadata.gps_position)

    # Compute similarity Xp = s A X + b
    X = np.array(X)
    Xp = np.array(Xp)
    T = tf.superimposition_matrix(X.T, Xp.T, scale=True)

    A, b = T[:3, :3], T[:3, 3]
    s = np.linalg.det(A)**(1. / 3)
    A /= s

    return s, A, b


def align_reconstruction_orientation_prior_similarity(reconstruction, config):
    """Align with GPS data assuming particular a camera orientation.

    In some cases, using 3D-3D matches directly fails to find proper
    orientation of the world.  That happends mainly when all cameras lie
    close to a straigh line.

    In such cases, we can impose a particular orientation of the cameras
    to improve the orientation of the alignment.  The config parameter
    `align_orientation_prior` can be used to specify such orientation.
    Accepted values are:
     - no_roll: assumes horizon is horizontal on the images
     - horizontal: assumes cameras are looking towards the horizon
     - vertical: assumes cameras are looking down towards the ground
    """
    gps_shots = []
    for shot in reconstruction.shots.values():
        if shot.metadata.gps_dop != 999999.0:
            gps_shots.append(shot)

    if len(gps_shots) < 2:
        reconstruction.alignment.num_correspondences = 0
        logger.debug('Orientation prior alignment NOT attempted')
        return

    filtered_shots = get_filtered_shots(gps_shots, config)

    reconstruction.alignment.num_correspondences = len(filtered_shots)
    logger.debug('Orientation prior alignment attempted on shots')
    for shot in filtered_shots:
        logger.debug("{}".format(shot.id))

    return get_sab_2d(filtered_shots, config)


def get_sab_2d(filtered_shots, config):
    X, Xp = [], []
    orientation_type = config['align_orientation_prior']
    onplane, verticals = [], []

    for shot in filtered_shots:
        X.append(shot.pose.get_origin())
        Xp.append(shot.metadata.gps_position)
        R = shot.pose.get_rotation_matrix()
        x, y, z = get_horizontal_and_vertical_directions(
            R, shot.metadata.orientation)
        if orientation_type == 'no_roll':
            onplane.append(x)
            verticals.append(-y)
        elif orientation_type == 'horizontal':
            onplane.append(x)
            onplane.append(z)
            verticals.append(y)
        elif orientation_type == 'vertical':
            onplane.append(x)
            onplane.append(y)
            verticals.append(-z)

    X = np.array(X)
    Xp = np.array(Xp)
    
    # Estimate ground plane.
    p = multiview.fit_plane(X - X.mean(axis=0), onplane, verticals)
    Rplane = multiview.plane_horizontalling_rotation(p)
    X = Rplane.dot(X.T).T

    # Estimate 2d similarity to align to GPS
    T = tf.affine_matrix_from_points(X.T[:2], Xp.T[:2], shear=False)
    s = np.linalg.det(T[:2, :2])**0.5
    A = np.eye(3)
    A[:2, :2] = T[:2, :2] / s
    A = A.dot(Rplane)
    b = np.array([
        T[0, 2],
        T[1, 2],
        Xp[:, 2].mean() - s * X[:, 2].mean()  # vertical alignment
    ])

    return s, A, b


def get_horizontal_and_vertical_directions(R, orientation):
    """Get orientation vectors from camera rotation matrix and orientation tag.

    Return a 3D vectors pointing to the positive XYZ directions of the image.
    X points to the right, Y to the bottom, Z to the front.
    """
    # See http://sylvana.net/jpegcrop/exif_orientation.html
    if orientation == 1:
        return R[0, :], R[1, :], R[2, :]
    if orientation == 2:
        return -R[0, :], R[1, :], -R[2, :]
    if orientation == 3:
        return -R[0, :], -R[1, :], R[2, :]
    if orientation == 4:
        return R[0, :], -R[1, :], R[2, :]
    if orientation == 5:
        return R[1, :], R[0, :], -R[2, :]
    if orientation == 6:
        return -R[1, :], R[0, :], R[2, :]
    if orientation == 7:
        return -R[1, :], -R[0, :], -R[2, :]
    if orientation == 8:
        return R[1, :], -R[0, :], R[2, :]
    logger.error('unknown orientation {0}. Using 1 instead'.format(orientation))
    return R[0, :], R[1, :], R[2, :]


def triangulate_single_gcp(reconstruction, observations):
    """Triangulate one Ground Control Point."""
    reproj_threshold = 0.014 # 0.004
    min_ray_angle_degrees = 2.0

    os, bs = [], []
    for o in observations:
        if o.shot_id in reconstruction.shots:
            shot = reconstruction.shots[o.shot_id]
            os.append(shot.pose.get_origin())
            b = shot.camera.pixel_bearing(np.asarray(o.shot_coordinates))
            r = shot.pose.get_rotation_matrix().T
            bs.append(r.dot(b))

    if len(os) >= 2:
        e, X = csfm.triangulate_bearings_midpoint(
            os, bs, reproj_threshold, np.radians(min_ray_angle_degrees))
        return X


def triangulate_all_gcp(reconstruction, gcp_observations):
    """Group and triangulate Ground Control Points seen in 2+ images."""
    groups = defaultdict(list)
    for o in gcp_observations:
        groups[tuple(o.lla)].append(o)

    triangulated, measured = [], []
    for observations in groups.values():
        x = triangulate_single_gcp(reconstruction, observations)
        if x is not None:
            triangulated.append(x)
            measured.append(observations[0].coordinates)

    return triangulated, measured


def area(a, b, c):
    return 0.5 * np.linalg.norm( np.cross( b-a, c-a ) )


def get_farthest_two_shots(gps_shots):
    """get two shots with gps that are most far apart"""
    distances = {}
    for (i, j) in combinations(gps_shots, 2):
        distances[(i, j)] = np.linalg.norm(np.array(i.metadata.gps_position) - np.array(j.metadata.gps_position))

    return max(distances.items(), key=operator.itemgetter(1))[0]


def get_farthest_three_shots(gps_shots):
    """get three shots with gps that are most far apart"""
    areas = {}
    for (i, j, k) in combinations(gps_shots, 3):
        areas[(i, j, k)] = area(np.array(i.metadata.gps_position), np.array(j.metadata.gps_position), np.array(k.metadata.gps_position))

    return max(areas.items(), key=operator.itemgetter(1))[0]


def get_filtered_shots(gps_shots, config):
    """filter out shots that are too close together"""

    # minimum distance between two gps points that we will use, in feet
    # TODO: move this constant to config
    min_gps_distance = 3

    scale_factor = config['reconstruction_scale_factor']
    min_distance_pixels = min_gps_distance / scale_factor

    distances = {}
    for (i, j) in combinations(gps_shots, 2):
        distances[(i, j)] = np.linalg.norm(np.array(i.metadata.gps_position) - np.array(j.metadata.gps_position))

    while len(gps_shots) > 2:
        (i, j) = min(distances.items(), key=operator.itemgetter(1))[0]
        if distances[(i, j)] < min_distance_pixels:
            if i in gps_shots:
                gps_shots.remove(i)
            del distances[(i, j)]
        else:
            break

    return gps_shots


def _shot_id_to_int(shot_id):
    """
    Returns: shot id to integer
    """
    tokens = shot_id.split(".")
    return int(tokens[0])
