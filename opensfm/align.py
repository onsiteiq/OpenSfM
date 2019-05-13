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


def align_reconstruction(reconstruction, gcp, config ):
    """Align a reconstruction with GPS and GCP data."""
    res = align_reconstruction_similarity(reconstruction, gcp, config )
    reconstruction.alignment.aligned = False
    if res:
        s, A, b = res
        apply_similarity(reconstruction, s, A, b)
        reconstruction.alignment.aligned = True


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

    filtered_shots = gps_shots
    if len(gps_shots) > 3:
        #filtered_shots = get_farthest_three_shots(gps_shots)
        filtered_shots = get_filtered_shots(gps_shots, config)

    if len(filtered_shots) < 3:
        reconstruction.alignment.num_correspondences = 0
        logger.debug('Similarity alignment NOT attempted ( {0} Correspondences )'.format(len(gps_shots)))
        return
    else:
        reconstruction.alignment.num_correspondences = len(filtered_shots)
        logger.debug('Similarity alignment attempted on shots')
        for shot in filtered_shots:
            logger.debug("{}".format(shot.id))

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

    # we use pdr input to guide the reconstruction (see align_reconstruction_to_pdr), so s should
    # be close to 1; x/y rotation should be close to 0, if not then likely it's 'flipped' which
    # could happen when the 3 points are close a degenerate configuration. so we check for these
    # and if any of them is true, we revert to 2-point method for alignment (orientation prior)
    [x, y, z] = _rotation_matrix_to_euler_angles(A)
    logger.debug('Similarity alignment result s={}, rot xyz={} {} {}'.format(s, x, y, z))
    if s > 1.1 or s < 0.9 or math.fabs(x) > 10.0 or math.fabs(y) > 10.0 or math.fabs(z) > 45.0:
        logger.debug('Similarity alignment result looks suspicious. Discard')
        return

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

    filtered_shots = gps_shots
    if len(gps_shots) > 2:
        #filtered_shots = get_farthest_two_shots(gps_shots)
        filtered_shots = get_filtered_shots(gps_shots, config)

    if len(filtered_shots) < 2:
        reconstruction.alignment.num_correspondences = 0
        logger.debug('Orientation prior alignment NOT attempted')
        return
    else:
        reconstruction.alignment.num_correspondences = len(filtered_shots)
        logger.debug('Orientation prior alignment attempted on shots')
        for shot in filtered_shots:
            logger.debug("{}".format(shot.id))


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

    [x, y, z] = _rotation_matrix_to_euler_angles(A)
    logger.debug('Orientation alignment result s={}, rot xyz={} {} {}'.format(s, x, y, z))
    if s > 1.1 or s < 0.9 or math.fabs(z) > 45.0:
        logger.debug('Orientation alignment result looks suspicious. Discard')
        return

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


def _rotation_matrix_to_euler_angles(R):
    """
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
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

    return np.array(np.degrees([x, y, z]))
