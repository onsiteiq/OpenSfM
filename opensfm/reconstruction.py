# -*- coding: utf-8 -*-
"""Incremental reconstruction pipeline"""

import datetime
import logging
from collections import defaultdict
from itertools import combinations

import cv2
import numpy as np
import pyopengv
import six
from timeit import default_timer as timer
from six import iteritems

from opensfm import csfm
from opensfm import geo
from opensfm import log
from opensfm import matching
from opensfm import multiview
from opensfm import types
from opensfm.align import align_reconstruction, align_reconstruction_segments, apply_similarity
from opensfm.align_pdr import init_pdr_predictions, direct_align_pdr, \
    update_pdr_prediction_position, update_pdr_prediction_rotation, \
    cull_resection_pdr, validate_resection_pdr, \
    scale_reconstruction_to_pdr, align_reconstructions_to_pdr, \
    debug_rotation_prior
from opensfm.context import parallel_map, current_memory_usage
from opensfm import transformations as tf

logger = logging.getLogger(__name__)


def _add_camera_to_bundle(ba, camera, constant):
    """Add camera to a bundle adjustment problem."""
    if camera.projection_type == 'perspective':
        ba.add_perspective_camera(
            str(camera.id), camera.focal, camera.k1, camera.k2,
            camera.focal_prior, camera.k1_prior, camera.k2_prior,
            constant)
    elif camera.projection_type == 'brown':
        c = csfm.BABrownPerspectiveCamera()
        c.id = str(camera.id)
        c.focal_x = camera.focal_x
        c.focal_y = camera.focal_y
        c.c_x = camera.c_x
        c.c_y = camera.c_y
        c.k1 = camera.k1
        c.k2 = camera.k2
        c.p1 = camera.p1
        c.p2 = camera.p2
        c.k3 = camera.k3
        c.focal_x_prior = camera.focal_x_prior
        c.focal_y_prior = camera.focal_y_prior
        c.c_x_prior = camera.c_x_prior
        c.c_y_prior = camera.c_y_prior
        c.k1_prior = camera.k1_prior
        c.k2_prior = camera.k2_prior
        c.p1_prior = camera.p1_prior
        c.p2_prior = camera.p2_prior
        c.k3_prior = camera.k3_prior
        c.constant = constant
        ba.add_brown_perspective_camera(c)
    elif camera.projection_type == 'fisheye':
        ba.add_fisheye_camera(
            str(camera.id), camera.focal, camera.k1, camera.k2,
            camera.focal_prior, camera.k1_prior, camera.k2_prior,
            constant)
    elif camera.projection_type in ['equirectangular', 'spherical']:
        ba.add_equirectangular_camera(str(camera.id))


def _get_camera_from_bundle(ba, camera):
    """Read camera parameters from a bundle adjustment problem."""
    if camera.projection_type == 'perspective':
        c = ba.get_perspective_camera(str(camera.id))
        camera.focal = c.focal
        camera.k1 = c.k1
        camera.k2 = c.k2
    elif camera.projection_type == 'brown':
        c = ba.get_brown_perspective_camera(str(camera.id))
        camera.focal_x = c.focal_x
        camera.focal_y = c.focal_y
        camera.c_x = c.c_x
        camera.c_y = c.c_y
        camera.k1 = c.k1
        camera.k2 = c.k2
        camera.p1 = c.p1
        camera.p2 = c.p2
        camera.k3 = c.k3
    elif camera.projection_type == 'fisheye':
        c = ba.get_fisheye_camera(str(camera.id))
        camera.focal = c.focal
        camera.k1 = c.k1
        camera.k2 = c.k2


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


def bundle(graph, reconstruction, gcp, config):
    """Bundle adjust a reconstruction."""
    fix_cameras = not config['optimize_camera_parameters']

    chrono = Chronometer()
    ba = csfm.BundleAdjuster()

    for camera in reconstruction.cameras.values():
        _add_camera_to_bundle(ba, camera, fix_cameras)

    for shot in reconstruction.shots.values():
        r = shot.pose.rotation
        t = shot.pose.translation
        ba.add_shot(
            str(shot.id), str(shot.camera.id),
            r[0], r[1], r[2],
            t[0], t[1], t[2],
            False
        )

    for point in reconstruction.points.values():
        x = point.coordinates
        ba.add_point(str(point.id), x[0], x[1], x[2], False)

    for shot_id in reconstruction.shots:
        if shot_id in graph:
            for track in graph[shot_id]:
                if track in reconstruction.points:
                    ba.add_observation(str(shot_id), str(track),
                                       *graph[shot_id][track]['feature'])

    if config['bundle_use_gps']:
        for shot in reconstruction.shots.values():
            g = shot.metadata.gps_position
            ba.add_position_prior(str(shot.id), g[0], g[1], g[2],
                                  shot.metadata.gps_dop)

    if config['bundle_use_gcp'] and gcp:
        for observation in gcp:
            if observation.shot_id in reconstruction.shots:
                ba.add_ground_control_point_observation(
                    str(observation.shot_id),
                    observation.coordinates[0],
                    observation.coordinates[1],
                    observation.coordinates[2],
                    observation.shot_coordinates[0],
                    observation.shot_coordinates[1])

    ba.set_loss_function(config['loss_function'],
                         config['loss_function_threshold'])
    ba.set_reprojection_error_sd(config['reprojection_error_sd'])
    ba.set_internal_parameters_prior_sd(
        config['exif_focal_sd'],
        config['principal_point_sd'],
        config['radial_distorsion_k1_sd'],
        config['radial_distorsion_k2_sd'],
        config['radial_distorsion_p1_sd'],
        config['radial_distorsion_p2_sd'],
        config['radial_distorsion_k3_sd'])
    ba.set_num_threads(config['processes'])
    ba.set_max_num_iterations(50)
    ba.set_linear_solver_type("SPARSE_SCHUR")

    chrono.lap('setup')
    ba.run()
    chrono.lap('run')

    for camera in reconstruction.cameras.values():
        _get_camera_from_bundle(ba, camera)

    for shot in reconstruction.shots.values():
        s = ba.get_shot(str(shot.id))
        shot.pose.rotation = [s.rx, s.ry, s.rz]
        shot.pose.translation = [s.tx, s.ty, s.tz]

    for point in reconstruction.points.values():
        p = ba.get_point(str(point.id))
        point.coordinates = [p.x, p.y, p.z]
        point.reprojection_error = p.reprojection_error

    chrono.lap('teardown')

    logger.debug(ba.brief_report())
    report = {
        'wall_times': dict(chrono.lap_times()),
        'brief_report': ba.brief_report(),
    }
    return report


def bundle_single_view(graph, reconstruction, shot, data):
    """Bundle adjust a single camera."""
    config = data.config
    ba = csfm.BundleAdjuster()
    camera = shot.camera

    _add_camera_to_bundle(ba, camera, constant=True)

    r = shot.pose.rotation
    t = shot.pose.translation
    ba.add_shot(
        str(shot.id), str(camera.id),
        r[0], r[1], r[2],
        t[0], t[1], t[2],
        False
    )

    for track_id in graph[shot.id]:
        if track_id in reconstruction.points:
            track = reconstruction.points[track_id]
            x = track.coordinates
            ba.add_point(str(track_id), x[0], x[1], x[2], True)
            ba.add_observation(str(shot.id), str(track_id),
                               *graph[shot.id][track_id]['feature'])

    if config['bundle_use_gps']:
        g = shot.metadata.gps_position
        ba.add_position_prior(str(shot.id), g[0], g[1], g[2],
                              shot.metadata.gps_dop)

    if config['bundle_use_pdr'] and shot.metadata.gps_dop == 999999.0:
        p, stddev1 = update_pdr_prediction_position(shot.id, reconstruction, data)
        ba.add_position_prior(str(shot.id), p[0], p[1], p[2], stddev1)

        #r, stddev2 = update_pdr_prediction_rotation(shot.id, reconstruction, data)
        #ba.add_rotation_prior(str(shot.id), r[0], r[1], r[2], stddev2)

        # debug
        if stddev1 != 999999.0:
            prev_shot_id = _prev_shot_id(shot.id)
            if prev_shot_id in reconstruction.shots:
                v = p - reconstruction.shots[prev_shot_id].pose.get_origin()
                logger.debug("pdr prior for {} positional displacement {} {} {}, dop={}".format(shot.id, v[0], v[1], v[2], stddev1))

        #if stddev2 != 999999.0:
            #prev_shot_id = _prev_shot_id(shot.id)
            #if prev_shot_id in reconstruction.shots:
                #q_0 = tf.quaternion_from_euler(r[0], r[1], r[2])
                #q_1 = tf.quaternion_from_matrix(reconstruction.shots[prev_shot_id].pose.get_rotation_matrix())

                #logger.debug("pdr prior for {} angular displacement {} dop={}".
                             #format(shot.id, tf.quaternion_distance(q_0, q_1), stddev2))

    ba.set_loss_function(config['loss_function'],
                         config['loss_function_threshold'])
    ba.set_reprojection_error_sd(config['reprojection_error_sd'])
    ba.set_internal_parameters_prior_sd(
        config['exif_focal_sd'],
        config['principal_point_sd'],
        config['radial_distorsion_k1_sd'],
        config['radial_distorsion_k2_sd'],
        config['radial_distorsion_p1_sd'],
        config['radial_distorsion_p2_sd'],
        config['radial_distorsion_k3_sd'])
    ba.set_num_threads(config['processes'])
    ba.set_max_num_iterations(10)
    ba.set_linear_solver_type("DENSE_QR")

    ba.run()

    logger.debug(ba.brief_report())

    s = ba.get_shot(str(shot.id))
    shot.pose.rotation = [s.rx, s.ry, s.rz]
    shot.pose.translation = [s.tx, s.ty, s.tz]


def bundle_local(graph, reconstruction, gcp, central_shot_id, data):
    """Bundle adjust the local neighborhood of a shot."""
    chrono = Chronometer()
    config = data.config

    interior, boundary = shot_neighborhood(
        graph, reconstruction, central_shot_id,
        config['local_bundle_radius'],
        config['local_bundle_min_common_points'],
        config['local_bundle_max_shots'])

    logger.debug(
        'Local bundle sets: interior {}  boundary {}  other {}'.format(
            len(interior), len(boundary),
            len(reconstruction.shots) - len(interior) - len(boundary)))

    point_ids = set()
    for shot_id in interior:
        if shot_id in graph:
            for track in graph[shot_id]:
                if track in reconstruction.points:
                    point_ids.add(track)

    ba = csfm.BundleAdjuster()

    for camera in reconstruction.cameras.values():
        _add_camera_to_bundle(ba, camera, constant=True)

    for shot_id in interior | boundary:
        shot = reconstruction.shots[shot_id]
        r = shot.pose.rotation
        t = shot.pose.translation
        ba.add_shot(
            str(shot.id), str(shot.camera.id),
            r[0], r[1], r[2],
            t[0], t[1], t[2],
            shot.id in boundary
        )

    for point_id in point_ids:
        point = reconstruction.points[point_id]
        x = point.coordinates
        ba.add_point(str(point.id), x[0], x[1], x[2], False)

    for shot_id in interior | boundary:
        if shot_id in graph:
            for track in graph[shot_id]:
                if track in point_ids:
                    ba.add_observation(str(shot_id), str(track),
                                       *graph[shot_id][track]['feature'])

    if config['bundle_use_gps']:
        for shot_id in interior:
            shot = reconstruction.shots[shot_id]
            g = shot.metadata.gps_position
            ba.add_position_prior(str(shot.id), g[0], g[1], g[2],
                                  shot.metadata.gps_dop)

    #if config['bundle_use_pdr']:
        #for shot_id in interior | boundary:
            #if reconstruction.shots[shot_id].metadata.gps_dop == 999999.0:
                #p, stddev1 = update_pdr_prediction_position(shot_id, reconstruction, data)
                #ba.add_position_prior(shot_id, p[0], p[1], p[2], stddev1)

                #r, stddev2 = update_pdr_prediction_rotation(shot_id, reconstruction, data)
                #ba.add_rotation_prior(shot_id, r[0], r[1], r[2], stddev2)

    if config['bundle_use_gcp'] and gcp:
        for observation in gcp:
            if observation.shot_id in interior:
                ba.add_ground_control_point_observation(
                    observation.shot_id,
                    observation.coordinates[0],
                    observation.coordinates[1],
                    observation.coordinates[2],
                    observation.shot_coordinates[0],
                    observation.shot_coordinates[1])

    ba.set_loss_function(config['loss_function'],
                         config['loss_function_threshold'])
    ba.set_reprojection_error_sd(config['reprojection_error_sd'])
    ba.set_internal_parameters_prior_sd(
        config['exif_focal_sd'],
        config['principal_point_sd'],
        config['radial_distorsion_k1_sd'],
        config['radial_distorsion_k2_sd'],
        config['radial_distorsion_p1_sd'],
        config['radial_distorsion_p2_sd'],
        config['radial_distorsion_k3_sd'])
    ba.set_num_threads(config['processes'])
    ba.set_max_num_iterations(10)
    ba.set_linear_solver_type("DENSE_SCHUR")

    chrono.lap('setup')
    ba.run()
    chrono.lap('run')

    for shot_id in interior:
        shot = reconstruction.shots[shot_id]
        s = ba.get_shot(str(shot.id))
        shot.pose.rotation = [s.rx, s.ry, s.rz]
        shot.pose.translation = [s.tx, s.ty, s.tz]

    for point in point_ids:
        point = reconstruction.points[point]
        p = ba.get_point(str(point.id))
        point.coordinates = [p.x, p.y, p.z]
        point.reprojection_error = p.reprojection_error

    chrono.lap('teardown')

    logger.debug(ba.brief_report())
    report = {
        'wall_times': dict(chrono.lap_times()),
        'brief_report': ba.brief_report(),
        'num_interior_images': len(interior),
        'num_boundary_images': len(boundary),
        'num_other_images': (len(reconstruction.shots)
                             - len(interior) - len(boundary)),
    }
    return report


def shot_neighborhood(graph, reconstruction, central_shot_id, radius,
                      min_common_points, max_interior_size):
    """Reconstructed shots near a given shot.

    Returns:
        a tuple with interior and boundary:
        - interior: the list of shots at distance smaller than radius
        - boundary: shots sharing at least on point with the interior

    Central shot is at distance 0.  Shots at distance n + 1 share at least
    min_common_points points with shots at distance n.
    """
    max_boundary_size = 1000000
    interior = set([central_shot_id])
    for distance in range(1, radius):
        remaining = max_interior_size - len(interior)
        if remaining <= 0:
            break
        neighbors = direct_shot_neighbors(
            graph, reconstruction, interior, min_common_points, remaining)
        interior.update(neighbors)
    boundary = direct_shot_neighbors(
        graph, reconstruction, interior, 1, max_boundary_size)
    return interior, boundary


def direct_shot_neighbors(graph, reconstruction, shot_ids,
                          min_common_points, max_neighbors):
    """Reconstructed shots sharing reconstructed points with a shot set."""
    points = set()
    for shot_id in shot_ids:
        for track_id in graph[shot_id]:
            if track_id in reconstruction.points:
                points.add(track_id)

    candidate_shots = set(reconstruction.shots) - set(shot_ids)
    common_points = defaultdict(int)
    for track_id in points:
        for neighbor in graph[track_id]:
            if neighbor in candidate_shots:
                common_points[neighbor] += 1

    pairs = sorted(common_points.items(), key=lambda x: -x[1])
    neighbors = set()
    for neighbor, num_points in pairs[:max_neighbors]:
        if num_points >= min_common_points:
            neighbors.add(neighbor)
        else:
            break
    return neighbors


def pairwise_reconstructability(common_tracks, rotation_inliers):
    """Likeliness of an image pair giving a good initial reconstruction."""
    outliers = common_tracks - rotation_inliers
    outlier_ratio = float(outliers) / common_tracks
    if outlier_ratio >= 0.3:
        return outliers
    else:
        return 0


def compute_image_pairs(track_dict, data):
    """All matched image pairs sorted by reconstructability."""
    args = _pair_reconstructability_arguments(track_dict, data)
    processes = data.config['processes']
    result = parallel_map(_compute_pair_reconstructability, args, processes)
    result = list(result)
    pairs = [(im1, im2) for im1, im2, r in result if r > 0]
    score = [r for im1, im2, r in result if r > 0]
    order = np.argsort(-np.array(score))
    return [pairs[o] for o in order]


def _pair_reconstructability_arguments(track_dict, data):
    threshold = 4 * data.config['five_point_algo_threshold']
    cameras = data.load_camera_models()
    args = []
    for (im1, im2), (tracks, p1, p2) in iteritems(track_dict):
        camera1 = cameras[data.load_exif(im1)['camera']]
        camera2 = cameras[data.load_exif(im2)['camera']]
        args.append((im1, im2, p1, p2, camera1, camera2, threshold))
    return args


def _compute_pair_reconstructability(args):
    log.setup()
    im1, im2, p1, p2, camera1, camera2, threshold = args
    R, inliers = two_view_reconstruction_rotation_only(
        p1, p2, camera1, camera2, threshold)
    r = pairwise_reconstructability(len(p1), len(inliers))
    return (im1, im2, r)


def get_image_metadata(data, image):
    """Get image metadata as a ShotMetadata object."""
    metadata = types.ShotMetadata()
    exif = data.load_exif(image)
    reflla = data.load_reference_lla()
    if ('gps' in exif and
            'latitude' in exif['gps'] and
            'longitude' in exif['gps']):
        lat = exif['gps']['latitude']
        lon = exif['gps']['longitude']
        if data.config['use_altitude_tag']:
            alt = exif['gps'].get('altitude', 2.0)
        else:
            alt = 2.0  # Arbitrary value used to align the reconstruction
        x, y, z = geo.topocentric_from_lla(
            lat, lon, alt,
            reflla['latitude'], reflla['longitude'], reflla['altitude'])
        metadata.gps_position = [x, y, z]
        metadata.gps_dop = exif['gps'].get('dop', 15.0)
    else:
        metadata.gps_position = [0.0, 0.0, 0.0]
        metadata.gps_dop = 999999.0

    metadata.orientation = exif.get('orientation', 1)

    if 'accelerometer' in exif:
        metadata.accelerometer = exif['accelerometer']

    if 'compass' in exif:
        metadata.compass = exif['compass']

    if 'capture_time' in exif:
        metadata.capture_time = exif['capture_time']

    if 'skey' in exif:
        metadata.skey = exif['skey']

    return metadata


def _two_view_reconstruction_inliers(b1, b2, R, t, threshold):
    """Compute number of points that can be triangulated.

    Args:
        b1, b2: Bearings in the two images.
        R, t: Rotation and translation from the second image to the first.
              That is the opengv's convention and the opposite of many
              functions in this module.
        threshold: max reprojection error in radians.
    Returns:
        array: Inlier indices.
    """
    p = pyopengv.triangulation_triangulate(b1, b2, t, R)

    br1 = p.copy()
    br1 /= np.linalg.norm(br1, axis=1)[:, np.newaxis]

    br2 = R.T.dot((p - t).T).T
    br2 /= np.linalg.norm(br2, axis=1)[:, np.newaxis]

    ok1 = np.linalg.norm(br1 - b1, axis=1) < threshold
    ok2 = np.linalg.norm(br2 - b2, axis=1) < threshold
    return np.nonzero(ok1 * ok2)[0]


def run_absolute_pose_ransac(bs, Xs, method, threshold,
                             iterations, probabilty):
    try:
        return pyopengv.absolute_pose_ransac(
            bs, Xs, method, threshold,
            iterations=iterations,
            probabilty=probabilty)
    except:
        # Older versions of pyopengv do not accept the probability argument.
        return pyopengv.absolute_pose_ransac(
            bs, Xs, method, threshold, iterations)


def run_relative_pose_ransac(b1, b2, method, threshold,
                             iterations, probability):
    try:
        return pyopengv.relative_pose_ransac(b1, b2, method, threshold,
                                             iterations=iterations,
                                             probability=probability)
    except:
        # Older versions of pyopengv do not accept the probability argument.
        return pyopengv.relative_pose_ransac(b1, b2, method, threshold,
                                             iterations)


def run_relative_pose_ransac_rotation_only(b1, b2, threshold,
                                           iterations, probability):
    try:
        return pyopengv.relative_pose_ransac_rotation_only(
            b1, b2, threshold,
            iterations=iterations,
            probability=probability)
    except:
        # Older versions of pyopengv do not accept the probability argument.
        return pyopengv.relative_pose_ransac_rotation_only(
            b1, b2, threshold, iterations)


def run_relative_pose_optimize_nonlinear(b1, b2, t, R):
    return pyopengv.relative_pose_optimize_nonlinear(b1, b2, t, R)


def two_view_reconstruction_plane_based(p1, p2, camera1, camera2, threshold):
    """Reconstruct two views from point correspondences lying on a plane.

    Args:
        p1, p2: lists points in the images
        camera1, camera2: Camera models
        threshold: reprojection error threshold

    Returns:
        rotation, translation and inlier list
    """
    b1 = camera1.pixel_bearing_many(p1)
    b2 = camera2.pixel_bearing_many(p2)
    x1 = multiview.euclidean(b1)
    x2 = multiview.euclidean(b2)

    H, inliers = cv2.findHomography(x1, x2, cv2.RANSAC, threshold)
    motions = multiview.motion_from_plane_homography(H)

    if not motions:
        return [], [], []

    motion_inliers = []
    for R, t, n, d in motions:
        inliers = _two_view_reconstruction_inliers(
            b1, b2, R.T, -R.T.dot(t), threshold)
        motion_inliers.append(inliers)

    best = np.argmax(map(len, motion_inliers))
    R, t, n, d = motions[best]
    inliers = motion_inliers[best]
    return cv2.Rodrigues(R)[0].ravel(), t, inliers


def two_view_reconstruction(p1, p2, camera1, camera2, threshold):
    """Reconstruct two views using the 5-point method.

    Args:
        p1, p2: lists points in the images
        camera1, camera2: Camera models
        threshold: reprojection error threshold

    Returns:
        rotation, translation and inlier list
    """
    b1 = camera1.pixel_bearing_many(p1)
    b2 = camera2.pixel_bearing_many(p2)

    # Note on threshold:
    # See opengv doc on thresholds here:
    #   http://laurentkneip.github.io/opengv/page_how_to_use.html
    # Here we arbitrarily assume that the threshold is given for a camera of
    # focal length 1.  Also, arctan(threshold) \approx threshold since
    # threshold is small
    T = run_relative_pose_ransac(
        b1, b2, "STEWENIUS", 1 - np.cos(threshold), 1000, 0.999)
    R = T[:, :3]
    t = T[:, 3]
    inliers = _two_view_reconstruction_inliers(b1, b2, R, t, threshold)

    if inliers.sum() > 5:
        T = run_relative_pose_optimize_nonlinear(b1[inliers], b2[inliers], t, R)
        R = T[:, :3]
        t = T[:, 3]
        inliers = _two_view_reconstruction_inliers(b1, b2, R, t, threshold)

    return cv2.Rodrigues(R.T)[0].ravel(), -R.T.dot(t), inliers


def _two_view_rotation_inliers(b1, b2, R, threshold):
    br2 = R.dot(b2.T).T
    ok = np.linalg.norm(br2 - b1, axis=1) < threshold
    return np.nonzero(ok)[0]


def two_view_reconstruction_rotation_only(p1, p2, camera1, camera2, threshold):
    """Find rotation between two views from point correspondences.

    Args:
        p1, p2: lists points in the images
        camera1, camera2: Camera models
        threshold: reprojection error threshold

    Returns:
        rotation and inlier list
    """
    b1 = camera1.pixel_bearing_many(p1)
    b2 = camera2.pixel_bearing_many(p2)

    R = run_relative_pose_ransac_rotation_only(
        b1, b2, 1 - np.cos(threshold), 1000, 0.999)
    inliers = _two_view_rotation_inliers(b1, b2, R, threshold)

    return cv2.Rodrigues(R.T)[0].ravel(), inliers


def two_view_reconstruction_general(p1, p2, camera1, camera2, threshold):
    """Reconstruct two views from point correspondences.

    These will try different reconstruction methods and return the
    results of the one with most inliers.

    Args:
        p1, p2: lists points in the images
        camera1, camera2: Camera models
        threshold: reprojection error threshold

    Returns:
        rotation, translation and inlier list
    """
    R_5p, t_5p, inliers_5p = two_view_reconstruction(
        p1, p2, camera1, camera2, threshold)

    R_plane, t_plane, inliers_plane = two_view_reconstruction_plane_based(
        p1, p2, camera1, camera2, threshold)

    report = {
        '5_point_inliers': len(inliers_5p),
        'plane_based_inliers': len(inliers_plane),
    }

    if len(inliers_5p) > len(inliers_plane):
        report['method'] = '5_point'
        return R_5p, t_5p, inliers_5p, report
    else:
        report['method'] = 'plane_based'
        return R_plane, t_plane, inliers_plane, report


def bootstrap_reconstruction(data, graph, im1, im2, p1, p2):
    """Start a reconstruction using two shots."""
    logger.info("Starting reconstruction with {} and {}".format(im1, im2))
    report = {
        'image_pair': (im1, im2),
        'common_tracks': len(p1),
    }

    cameras = data.load_camera_models()
    camera1 = cameras[data.load_exif(im1)['camera']]
    camera2 = cameras[data.load_exif(im2)['camera']]

    threshold = data.config['five_point_algo_threshold']
    min_inliers = data.config['five_point_algo_min_inliers']
    R, t, inliers, report['two_view_reconstruction'] = \
        two_view_reconstruction_general(p1, p2, camera1, camera2, threshold)

    logger.info("Two-view reconstruction inliers: {} / {}".format(
        len(inliers), len(p1)))

    if len(inliers) <= 5:
        report['decision'] = "Could not find initial motion"
        logger.info(report['decision'])
        return None, report

    reconstruction = types.Reconstruction()
    reconstruction.cameras = cameras

    shot1 = types.Shot()
    shot1.id = im1
    shot1.camera = camera1
    shot1.pose = types.Pose()
    shot1.metadata = get_image_metadata(data, im1)
    reconstruction.add_shot(shot1)

    shot2 = types.Shot()
    shot2.id = im2
    shot2.camera = camera2
    shot2.pose = types.Pose(R, t)
    shot2.metadata = get_image_metadata(data, im2)
    reconstruction.add_shot(shot2)

    triangulate_shot_features(graph, reconstruction, im1, data.config)

    logger.info("Triangulated: {}".format(len(reconstruction.points)))
    report['triangulated_points'] = len(reconstruction.points)

    if len(reconstruction.points) < min_inliers:
        report['decision'] = "Initial motion did not generate enough points"
        logger.info(report['decision'])
        return None, report

    bundle_single_view(graph, reconstruction, shot2, data)
    retriangulate(graph, reconstruction, data.config)
    bundle_single_view(graph, reconstruction, shot2, data)

    report['decision'] = 'Success'
    report['memory_usage'] = current_memory_usage()
    return reconstruction, report


def reconstructed_points_for_images(graph, reconstruction, images):
    """Number of reconstructed points visible on each image.

    Returns:
        A list of (image, num_point) pairs sorted by decreasing number
        of points.
    """
    res = []
    for image in images:
        if image not in reconstruction.shots:
            common_tracks = 0
            for track in graph[image]:
                if track in reconstruction.points:
                    common_tracks += 1
            res.append((image, common_tracks))
    return sorted(res, key=lambda x: -x[1])


def resect( data, graph, reconstruction, shot_id ):
    """Try resecting and adding a shot to the reconstruction.

    Return:
        True on success.
    """
    exif = data.load_exif(shot_id)
    camera = reconstruction.cameras[exif['camera']]

    bs = []
    Xs = []
    track_ids = []
    for track in graph[shot_id]:
        if track in reconstruction.points:
            x = graph[track][shot_id]['feature']
            b = camera.pixel_bearing(x)
            bs.append(b)
            Xs.append(reconstruction.points[track].coordinates)
            track_ids.append(track)
    bs = np.array(bs)
    Xs = np.array(Xs)

    if len(bs) < data.config['resection_min_inliers']:
        return False, {'num_common_points': len(bs)}

    # remove features that are obviously wrong, according to pdr
    #bs, Xs, track_ids = cull_resection_pdr(shot_id, reconstruction, data, bs, Xs, track_ids)

    threshold = data.config['resection_threshold']

    T = run_absolute_pose_ransac(
        bs, Xs, "KNEIP", 1 - np.cos(threshold), 1000, 0.999)

    R = T[:, :3]
    t = T[:, 3]

    reprojected_bs = R.T.dot((Xs - t).T).T
    reprojected_bs /= np.linalg.norm(reprojected_bs, axis=1)[:, np.newaxis]

    inliers = np.linalg.norm(reprojected_bs - bs, axis=1) < threshold
    ninliers = int(sum(inliers))

    min_inliers = get_resection_min_inliers(data, graph, reconstruction, shot_id, track_ids, inliers, ninliers)

    #logger.info("{} resection inliers: {} / {}, threshod {}".format(
        #shot_id, ninliers, len(bs), min_inliers))
    report = {
        'num_common_points': len(bs),
        'num_inliers': ninliers,
    }

    if ninliers >= min_inliers:
        try:
            R = T[:, :3].T
            
            print( "Resect R: " + str(R) )
            print( "Resect rod: " + str(cv2.Rodrigues(R)[0].ravel()) )
            
            t = -R.dot(T[:, 3])
            shot = types.Shot()
            shot.id = shot_id
            shot.camera = camera
            shot.pose = types.Pose()
            shot.pose.set_rotation_matrix(R)
            shot.pose.translation = t
            
            print( "Resect Origin: " + str(shot.pose.get_origin()) )
            
            shot.metadata = get_image_metadata(data, shot_id)
        except Exception as e:
            print( e )
            return False, report

        bundle_single_view(graph, reconstruction, shot, data)
        reconstruction.add_shot(shot)
        return True, report

        # check if rotation of this shot after bundle adjustment is close to what we are expecting
        #is_pos_ok, is_rot_ok = validate_resection_pdr(reconstruction, data, shot)
        #if is_pos_ok and is_rot_ok:
            #reconstruction.add_shot(shot)
            #return True, report
        #else:
            #logger.info("resection of {} failed. pos_ok {} rot_ok {}".format(shot_id, is_pos_ok, is_rot_ok))
            #return False, report
    else:
        return False, report


def resect_structureless( data, graph, common_tracks, reconstruction, shot_id ):
    """Try resecting using structureless resection and adding a shot to the reconstruction.

    Return:
        True on success.
    """
    
    # Find a the image in the reconstruction with the most shared
    # tracks (assumed minimum 2 images to create a track) to select
    # as our image one for Structureless Resection
    
    image_one = None
    # Image correspondences for relative motion computation betweein image and image_one
    pr = p1r = None
    max_common_tracks = 0
    
    im_ind = _shot_id_to_int(shot_id)
    
    for image_a in reconstruction.shots.keys():                   
        if image_a != shot_id:
            
            tracks, pcur, pa = get_common_tracks( shot_id, image_a, common_tracks )
                
            num_tracks = len(tracks)
            if num_tracks > max_common_tracks:
                max_common_tracks = num_tracks
                image_one = image_a
                pr = pcur
                p1r = pa
    
    logger.info( 'image: ' + shot_id )
    logger.info( 'image_one: ' + str(image_one) )
    
    image_two = None
    
    if image_one is not None:
        # If we've found a suitable image one then choose an image two.
        # The image within the reconstruction that has the most common
        # tracks with the candidate image and image one
        
        # Image correspondences for absolute motion computation for image translation using 
        # image_one and image_two known (R,t) and the relative motion between image and
        # image_one
        pa = p2a = None
        max_common_to_both = 0
        for image_b in reconstruction.shots.keys():
            if image_b != shot_id and image_b != image_one:
                
                tracks02, pcur02, pb02 = get_common_tracks( shot_id, image_b, common_tracks )
                tracks01, pcur01, pb01 = get_common_tracks( image_one, image_b, common_tracks )
                
                num_common_to_both = len( set(tracks01).intersection( set(tracks02) ) )
                
                if num_common_to_both > max_common_to_both:
                    max_common_to_both = num_common_to_both
                    image_two = image_b
                    pa = pcur02
                    p2a = pb02
    
    logger.info( 'image_two: ' + str(image_two) )
    
    if image_one is None:
        return False, {'num_common_points_one': 0, 'num_common_points_two': 0, 'relative_result': "Failure" }
    elif image_two is None:
        return False, {'num_common_points_one': len(pr), 'num_common_points_two': 0, 'relative_result': "Failure" }
    
    logger.info( "Computing relative motion with {} and {}".format( shot_id, image_one ) )
    
    report = {
        'image_pair_relative': ( shot_id, image_one ),
        'num_common_points_one': len(pr),
        'num_common_points_two': len(p2a),
    }

    cameras = data.load_camera_models()
    camera = cameras[data.load_exif(shot_id)['camera']]
    camera1 = cameras[data.load_exif(image_one)['camera']]

    threshold = data.config['five_point_algo_threshold']
    min_inliers = data.config['five_point_algo_min_inliers']
    
    R_rel, t, inliers, report['image_one_relative'] = \
        two_view_reconstruction_general( pr, p1r, camera, camera1, threshold )

    R_rel = cv2.Rodrigues(R_rel)[0]

    logger.info( "Two-view relative inliers: {} / {}".format( len(inliers), len(pr) ) )

    if len(inliers) <= 5:
        report['relative_result'] = "Could not find relative motion"
        logger.info(report['relative_result'])
        return False, report
       
    report['relative_result'] = "Success"
    
    logger.info( "Computing absolute motion with {}, {} and {}".format( shot_id, image_one, image_two ) )
    
    shot1 = reconstruction.shots[image_one]
    
    R1 = shot1.pose.get_rotation_matrix()
    o1 = shot1.pose.get_origin()
    
    b0_cam = camera.pixel_bearing_many(pa)
    
    camera2 = cameras[data.load_exif(image_two)['camera']]
    shot2 = reconstruction.shots[image_two]
    
    R2 = shot2.pose.get_rotation_matrix()
    o2 = shot2.pose.get_origin()
    b2_cam = camera2.pixel_bearing_many(p2a)
    
    b2_wrld = np.dot( b2_cam, R2.T )
    
    # The combination of shot one rotation and the relative rotation to the shot being considered.
    R0 = np.dot( R1, R_rel )
    
    # Translation is an unscaled unit vector in world coordinates.
    t = R1.T.dot(t)
    
    t /= np.linalg.norm(t)
    
    #print( "Resect Structureless R0: " + str(R0) )
    #print( "Resect Structureless t (in): " + str(t) )
    #print( "Resect Structureless rod: " +str( cv2.Rodrigues(R0)[0].ravel() ) )
    
    threshold = data.config['resection_threshold']
    
    # Due to row major (OpenCV) to column major (Eigen) take the transpose of R0. t is a direction in world coordinates
    # so it should be uneffected.
    
    T = pyopengv.absolute_pose_onept_ransac( b0_cam, b2_wrld, o1, o2, R0.T, t, 1 - np.cos(threshold), 1000, 0.999 )
    
    #print( "Resect Structureless t (out): " + str(T[:, 3]) )
    
    # Failure is indicated by a translation value of the zero vector
    if np.array_equal( T[:, 3], np.zeros(3) ):
        report['absolute_result'] = "Could not find absolute translation"
        logger.info(report['absolute_result'])
        return False, report
    
    report['absolute_result'] = "Success"
    
    try:
        R = T[:, :3].T
        t = -R.dot(T[:, 3])
        shot = types.Shot()
        shot.id = shot_id
        shot.camera = camera
        shot.pose = types.Pose()
        shot.pose.set_rotation_matrix(R)
        shot.pose.translation = t
        shot.metadata = get_image_metadata(data, shot_id)
    except:
        return False, report

    bundle_single_view(graph, reconstruction, shot, data)
    reconstruction.add_shot(shot)
    return True, report


def get_resection_min_inliers(data, graph, reconstruction, shot_id, track_ids, inliers, ninliers):
    """
    if inliers are mostly not from direct neighbors (shot number +/- 5), use a higher min_inlier
    threshold, to prevent spurious features getting into resection
    """
    ninliers_neighbors = 0
    for i in range(len(track_ids)):
        if inliers[i]:
            for a_id in graph[track_ids[i]]:
                if a_id in reconstruction.shots:
                    if abs(_shot_id_to_int(shot_id) - _shot_id_to_int(a_id)) < 5:
                        ninliers_neighbors += 1
                        break

    if ninliers_neighbors > ninliers/2:
        return data.config['resection_min_inliers']
    else:
        return data.config['resection_min_inliers_conservative']


class TrackTriangulator:
    """Triangulate tracks in a reconstruction.

    Caches shot origin and rotation matrix
    """

    def __init__(self, graph, reconstruction):
        """Build a triangulator for a specific reconstruction."""
        self.graph = graph
        self.reconstruction = reconstruction
        self.origins = {}
        self.rotation_inverses = {}
        self.Rts = {}
        self.max_dist_unaligned = self._max_triangulation_dist()

    def _max_triangulation_dist( self ):
        
        r = self.reconstruction
    
        #if r.alignment.aligned:
            
            # If we are aligned compute the smallest dimension of x,y axis 
            # aligned bounding box. The units are pixels.
            
            #origins = []
            #for shot in r.shots.values():
           #     o = shot.pose.get_origin()
            #    origins.append( o )
        
           # np_origins = np.stack( origins )
            
           # min_xy = np.amin( np_origins, axis = 0 )[0:2]
           # max_xy = np.amax( np_origins, axis = 0 )[0:2]
            
           # return 1630 #np.amin( max_xy - min_xy )
            
       # else:
        
            #return None
            
            # Estimate from the step size which is known to be ~3 ft.
            
        shots = list(r.shots.values())
        shots.sort( key = lambda x: x.id )
        
        index_dists = []
        for ind, shot in enumerate( shots ):
            cur_o = shot.pose.get_origin()
            if ind + 1 < len(shots):
                next_shot = shots[ind+1]
                next_o = next_shot.pose.get_origin()
                
                index_dist = np.linalg.norm( next_o - cur_o )
                index_dists.append( index_dist )
        
        index_dists.sort()
        
        #print( "========================================================================" )
        #print( np.median( index_dists ) )
        
        #exit()
        
        return np.median( index_dists )*20.0

    def triangulate(self, track, reproj_threshold, min_ray_angle_degrees,
                    min_distance=0, max_distance=999999, max_z_diff=999999):
        """Triangulate track and add point to reconstruction."""
        os, bs = [], []
        for shot_id in self.graph[track]:
            if shot_id in self.reconstruction.shots:
                shot = self.reconstruction.shots[shot_id]
                os.append(self._shot_origin(shot))
                x = self.graph[track][shot_id]['feature']
                b = shot.camera.pixel_bearing(np.array(x))
                r = self._shot_rotation_inverse(shot)
                bs.append(r.dot(b))

        if len(os) >= 2:
            thresholds = len(os) * [reproj_threshold]
            e, X = csfm.triangulate_bearings_midpoint(
                os, bs, thresholds, np.radians(min_ray_angle_degrees))
            if X is not None:
                within_range = True
                if self.reconstruction.alignment.aligned:
                    for shot_origin in os:
                        v = X - np.array(shot_origin)
                        distance = np.linalg.norm(v)

                        if distance < min_distance or distance > max_distance:
                            within_range = False
                            break

                        if np.fabs(v[2]) > max_z_diff:
                            within_range = False
                            break

                if within_range:
                    point = types.Point()
                    point.id = track
                    point.coordinates = X.tolist()

                    self.reconstruction.add_point(point)
                else:
                    if track in self.reconstruction.points:
                        del self.reconstruction.points[track]

    def triangulate_dlt(self, track, reproj_threshold, min_ray_angle_degrees):
        """Triangulate track using DLT and add point to reconstruction."""
        Rts, bs = [], []
        for shot_id in self.graph[track]:
            if shot_id in self.reconstruction.shots:
                shot = self.reconstruction.shots[shot_id]
                Rts.append(self._shot_Rt(shot))
                x = self.graph[track][shot_id]['feature']
                b = shot.camera.pixel_bearing(np.array(x))
                bs.append(b)

        if len(Rts) >= 2:
            e, X = csfm.triangulate_bearings_dlt(
                Rts, bs, reproj_threshold, np.radians(min_ray_angle_degrees))
            if X is not None:
                point = types.Point()
                point.id = track
                point.coordinates = X.tolist()
                self.reconstruction.add_point(point)

    def _shot_origin(self, shot):
        if shot.id in self.origins:
            return self.origins[shot.id]
        else:
            o = shot.pose.get_origin()
            self.origins[shot.id] = o
            return o

    def _shot_rotation_inverse(self, shot):
        if shot.id in self.rotation_inverses:
            return self.rotation_inverses[shot.id]
        else:
            r = shot.pose.get_rotation_matrix().T
            self.rotation_inverses[shot.id] = r
            return r

    def _shot_Rt(self, shot):
        if shot.id in self.Rts:
            return self.Rts[shot.id]
        else:
            r = shot.pose.get_Rt()
            self.Rts[shot.id] = r
            return r


def triangulate_shot_features(graph, reconstruction, shot_id, config):
    """Reconstruct as many tracks seen in shot_id as possible."""
    reproj_threshold = config['triangulation_threshold']
    min_ray_angle = config['triangulation_min_ray_angle']

    triangulator = TrackTriangulator(graph, reconstruction)

    max_distance = config['max_triangulation_distance']/config['reconstruction_scale_factor']
    min_distance = config['min_triangulation_distance']/config['reconstruction_scale_factor']
    max_z_diff = config['max_triangulation_height_diff']/config['reconstruction_scale_factor']

    for track in graph[shot_id]:
        if track not in reconstruction.points:
            triangulator.triangulate(track, reproj_threshold, min_ray_angle, min_distance, max_distance, max_z_diff)


def retriangulate(graph, reconstruction, config):
    """Retrianguate all points"""
    chrono = Chronometer()
    report = {}
    report['num_points_before'] = len(reconstruction.points)
    threshold = config['triangulation_threshold']
    min_ray_angle = config['triangulation_min_ray_angle']
    triangulator = TrackTriangulator(graph, reconstruction)

    max_distance = config['max_triangulation_distance']/config['reconstruction_scale_factor']
    min_distance = config['min_triangulation_distance']/config['reconstruction_scale_factor']
    max_z_diff = config['max_triangulation_height_diff']/config['reconstruction_scale_factor']

    tracks, images = matching.tracks_and_images(graph)
    for track in tracks:
        triangulator.triangulate(track, threshold, min_ray_angle, min_distance, max_distance, max_z_diff)
    report['num_points_after'] = len(reconstruction.points)
    chrono.lap('retriangulate')
    report['wall_time'] = chrono.total_time()
    return report


def remove_outliers(graph, reconstruction, config):
    """Remove points with large reprojection error."""
    threshold = config['bundle_outlier_threshold']
    if threshold > 0:
        outliers = []
        for track in reconstruction.points:
            error = reconstruction.points[track].reprojection_error
            if error > threshold:
                outliers.append(track)
        for track in outliers:
            del reconstruction.points[track]
        logger.info("Removed outliers: {}".format(len(outliers)))


def shot_lla_and_compass(shot, reference):
    """Lat, lon, alt and compass of the reconstructed shot position."""
    topo = shot.pose.get_origin()
    lat, lon, alt = geo.lla_from_topocentric(
        topo[0], topo[1], topo[2],
        reference['latitude'], reference['longitude'], reference['altitude'])

    dz = shot.viewing_direction()
    angle = np.rad2deg(np.arctan2(dz[0], dz[1]))
    angle = (angle + 360) % 360
    return lat, lon, alt, angle


def merge_two_reconstructions(r1, r2, config, threshold=1):
    """Merge two reconstructions with common tracks."""
    t1, t2 = r1.points, r2.points
    common_tracks = list(set(t1) & set(t2))

    if len(common_tracks) > 6:

        # Estimate similarity transform
        p1 = np.array([t1[t].coordinates for t in common_tracks])
        p2 = np.array([t2[t].coordinates for t in common_tracks])

        T, inliers = multiview.fit_similarity_transform(
            p1, p2, max_iterations=1000, threshold=threshold)

        if len(inliers) >= 10:
            s, A, b = multiview.decompose_similarity_transform(T)
            r1p = r1
            apply_similarity(r1p, s, A, b)
            r = r2
            r.shots.update(r1p.shots)
            r.points.update(r1p.points)
            align_reconstruction(r, None, config)
            return [r]
        else:
            return [r1, r2]
    else:
        return [r1, r2]


def merge_reconstructions(reconstructions, config):
    """Greedily merge reconstructions with common tracks."""
    num_reconstruction = len(reconstructions)
    ids_reconstructions = np.arange(num_reconstruction)
    remaining_reconstruction = ids_reconstructions
    reconstructions_merged = []
    num_merge = 0

    for (i, j) in combinations(ids_reconstructions, 2):
        if (i in remaining_reconstruction) and (j in remaining_reconstruction):
            r = merge_two_reconstructions(
                reconstructions[i], reconstructions[j], config)
            if len(r) == 1:
                remaining_reconstruction = list(set(
                    remaining_reconstruction) - set([i, j]))
                for k in remaining_reconstruction:
                    rr = merge_two_reconstructions(r[0], reconstructions[k],
                                                   config)
                    if len(r) == 2:
                        break
                    else:
                        r = rr
                        remaining_reconstruction = list(set(
                            remaining_reconstruction) - set([k]))
                reconstructions_merged.append(r[0])
                num_merge += 1

    for k in remaining_reconstruction:
        reconstructions_merged.append(reconstructions[k])

    logger.info("Merged {0} reconstructions".format(num_merge))

    return reconstructions_merged


def paint_reconstruction(data, graph, reconstruction):
    """Set the color of the points from the color of the tracks."""
    for k, point in iteritems(reconstruction.points):
        point.color = six.next(six.itervalues(graph[k]))['feature_color']


class ShouldBundle:
    """Helper to keep track of when to run bundle."""

    def __init__(self, data, reconstruction):
        self.interval = data.config['bundle_interval']
        self.new_points_ratio = data.config['bundle_new_points_ratio']
        self.reconstruction = reconstruction
        self.done()

    def should(self):
        max_points = self.num_points_last * self.new_points_ratio
        max_shots = self.num_shots_last + self.interval
        return (len(self.reconstruction.points) >= max_points or
                len(self.reconstruction.shots) >= max_shots)

    def done(self):
        self.num_points_last = len(self.reconstruction.points)
        self.num_shots_last = len(self.reconstruction.shots)


class ShouldRetriangulate:
    """Helper to keep track of when to re-triangulate."""

    def __init__(self, data, reconstruction):
        self.active = data.config['retriangulation']
        self.ratio = data.config['retriangulation_ratio']
        self.reconstruction = reconstruction
        self.done()

    def should(self):
        max_points = self.num_points_last * self.ratio
        return self.active and len(self.reconstruction.points) > max_points

    def done(self):
        self.num_points_last = len(self.reconstruction.points)


def get_common_tracks( image_a, image_b, common_tracks ):

    #ima_ind = _shot_id_to_int( image_a )
    #imb_ind = _shot_id_to_int( image_b )
                        
    return common_tracks.get( tuple( sorted([image_a, image_b]) ), ( [], [], [] ) )
                        
    #if ima_ind < imb_ind:
    #    tracks, pa, pb = common_tracks[ image_a, image_b ]
    #else:
    #    tracks, pb, pa = common_tracks[ image_b, image_a ]
        
    #return (tracks, pa, pb)


def grow_reconstruction_sequential(data, graph, reconstruction, common_tracks, images, gcp):
    """Incrementally add shots to an initial reconstruction."""
    config = data.config
    report = {'steps': []}

    bundle(graph, reconstruction, None, config)
    remove_outliers(graph, reconstruction, config)
    align_reconstruction(reconstruction, gcp, config)

    should_bundle = ShouldBundle(data, reconstruction)
    should_retriangulate = ShouldRetriangulate(data, reconstruction)
    while True:
        if config['save_partial_reconstructions']:
            paint_reconstruction(data, graph, reconstruction)
            data.save_reconstruction(
                [reconstruction], 'reconstruction.{}.json'.format(
                    datetime.datetime.now().isoformat().replace(':', '_')))

        logger.info("-------------------------------------------------------")
        for image in images:
        
            #ok, resrep = resect(data, graph, reconstruction, image)
            #if False: #not ok:
            
            ok, resrep = resect_structureless( data, graph, common_tracks, reconstruction, image )
                
            #Rst = resect_structureless( data, graph, common_tracks, reconstruction, image )
            
            #ok, resrep = resect( data, graph, reconstruction, image )
            
            #exit()
                
            if not ok:
                continue

            logger.info("Adding {0} to the reconstruction".format(image))
            step = {
                'image': image,
                'resection': resrep,
                'memory_usage': current_memory_usage()
            }
            report['steps'].append(step)
            images.remove(image)

            np_before = len(reconstruction.points)
            triangulate_shot_features(graph, reconstruction, image, config)
            np_after = len(reconstruction.points)
            step['triangulated_points'] = np_after - np_before
            logger.info("grow_reconstruction_sequential: {} points in the reconstruction".format(np_after))

            if should_retriangulate.should():
                logger.info("Re-triangulating")
                b1rep = bundle(graph, reconstruction, None, config)
                rrep = retriangulate(graph, reconstruction, config)
                b2rep = bundle(graph, reconstruction, None, config)
                remove_outliers(graph, reconstruction, config)
                align_reconstruction(reconstruction, gcp, config)
                step['bundle'] = b1rep
                step['retriangulation'] = rrep
                step['bundle_after_retriangulation'] = b2rep
                should_retriangulate.done()
                should_bundle.done()
            elif should_bundle.should():
                logger.info("Global bundle adjustment")
                brep = bundle(graph, reconstruction, None, config)
                remove_outliers(graph, reconstruction, config)
                align_reconstruction(reconstruction, gcp, config)
                step['bundle'] = brep
                should_bundle.done()
            elif config['local_bundle_radius'] > 0:
                brep = bundle_local(graph, reconstruction, None, image, data)
                remove_outliers(graph, reconstruction, config)
                step['local_bundle'] = brep

            if not reconstruction.alignment.scaled:
                scale_reconstruction_to_pdr(reconstruction, data)

            break
        else:
            break

        max_recon_size = config.get( 'reconstruction_max_images', -1 )
        if max_recon_size != -1:
            if len( reconstruction.shots ) >= max_recon_size:
                break

    logger.info("-------------------------------------------------------")

    bundle(graph, reconstruction, gcp, config)
    remove_outliers(graph, reconstruction, config)
    align_reconstruction_segments(reconstruction, gcp, config)
    paint_reconstruction(data, graph, reconstruction)

    if len(images) > 0:
        # remaining images may be ordered quite randomly, so sort them
        images.sort()
        logger.info("{} images can not be added {}".format(len(images), images))

    return reconstruction, report


def grow_reconstruction(data, graph, reconstruction, images, gcp):
    """Incrementally add shots to an initial reconstruction."""
    config = data.config
    report = {'steps': []}

    bundle(graph, reconstruction, None, config)
    remove_outliers(graph, reconstruction, config)
    align_reconstruction(reconstruction, gcp, config)

    should_bundle = ShouldBundle(data, reconstruction)
    should_retriangulate = ShouldRetriangulate(data, reconstruction)
    while True:
        if config['save_partial_reconstructions']:
            paint_reconstruction(data, graph, reconstruction)
            data.save_reconstruction(
                [reconstruction], 'reconstruction.{}.json'.format(
                    datetime.datetime.now().isoformat().replace(':', '_')))

        candidates = reconstructed_points_for_images(graph, reconstruction, images)
        if not candidates:
            break

        logger.info("-------------------------------------------------------")
        for image, num_tracks in candidates:
            ok, resrep = resect(data, graph, reconstruction, image)
            if not ok:
                continue

            logger.info("Adding {0} to the reconstruction".format(image))
            step = {
                'image': image,
                'resection': resrep,
                'memory_usage': current_memory_usage()
            }
            report['steps'].append(step)
            images.remove(image)

            np_before = len(reconstruction.points)
            triangulate_shot_features(graph, reconstruction, image, config)
            np_after = len(reconstruction.points)
            step['triangulated_points'] = np_after - np_before

            if should_retriangulate.should():
                logger.info("Re-triangulating")
                b1rep = bundle(graph, reconstruction, None, config)
                rrep = retriangulate(graph, reconstruction, config)
                b2rep = bundle(graph, reconstruction, None, config)
                remove_outliers(graph, reconstruction, config)
                align_reconstruction(reconstruction, gcp, config)
                step['bundle'] = b1rep
                step['retriangulation'] = rrep
                step['bundle_after_retriangulation'] = b2rep
                should_retriangulate.done()
                should_bundle.done()
            elif should_bundle.should():
                brep = bundle(graph, reconstruction, None, config)
                remove_outliers(graph, reconstruction, config)
                align_reconstruction(reconstruction, gcp, config)
                step['bundle'] = brep
                should_bundle.done()
            elif config['local_bundle_radius'] > 0:
                brep = bundle_local(graph, reconstruction, None, image, data)
                remove_outliers(graph, reconstruction, config)
                step['local_bundle'] = brep

            break
        else:
            logger.info("Some images can not be added")
            break

        max_recon_size = config.get( 'reconstruction_max_images', -1 )
        if max_recon_size != -1:
            if len( reconstruction.shots ) >= max_recon_size:
                break

    logger.info("-------------------------------------------------------")

    bundle(graph, reconstruction, gcp, config)
    remove_outliers(graph, reconstruction, config)
    align_reconstruction(reconstruction, gcp, config)
    paint_reconstruction(data, graph, reconstruction)
    return reconstruction, report


def direct_align_reconstruction_pdr( data ):

    target_images = data.config.get('target_images', [])

    report = {}
    report['reconstructions'] = []
    rec_report = {}
    report['reconstructions'].append(rec_report)
    rec_report['subset'] = target_images

    data.save_reconstruction([direct_align_pdr(data)])

    return report


def direct_align_reconstruction( data ):

    target_images = data.config.get( 'target_images', [] )
    
    report = {}
    report['reconstructions'] = []
    rec_report = {}
    report['reconstructions'].append(rec_report)
    rec_report['subset'] = target_images
    
    gps_points_dict = {}
    if data.gps_points_exist():
        gps_points_dict = data.load_gps_points()
    
    cameras = data.load_camera_models()
    
    reconstruction = types.Reconstruction()
    reconstruction.cameras = cameras
    
    for img in target_images:
    
        camera = cameras[data.load_exif(img)['camera']]
        
        shot = types.Shot()
        shot.id = img
        shot.camera = camera
        shot.pose = types.Pose()
        shot.metadata = get_image_metadata( data, img )
        
        if shot.metadata.compass is not None:

            Rc = tf.rotation_matrix( np.deg2rad( -shot.metadata.compass ), [ 0, 1, 0 ] )[:3, :3]
            
            shot.pose.set_rotation_matrix( Rc )
            
            Rplane = multiview.plane_horizontalling_rotation( [ 0, 1, 0] )
            
            t_shot = np.array( shot.metadata.gps_position )
            
            R = shot.pose.get_rotation_matrix()
            
            Rp = R.dot( Rplane.T )
            tp = -Rp.dot( t_shot )
            
            shot.pose.set_rotation_matrix(Rp)
            shot.pose.translation = list(tp)
            
            reconstruction.add_shot( shot )

        else:
            logger.warning("Image doesn't have corresponding GPS with compass: {}".format( img ) )

    reconstruction.alignment.aligned = True
    reconstruction.alignment.num_correspondences = len( target_images )

    reconstructions = []
    reconstructions.append( reconstruction )
                    
    data.save_reconstruction( reconstructions )
    
    return report


def incremental_reconstruction(data):
    """Run the entire incremental reconstruction pipeline."""
    logger.info("Starting incremental reconstruction")
    report = {}
    chrono = Chronometer()
    if not data.reference_lla_exists() and not data.config['use_provided_reference_lla']:
        data.invent_reference_lla()

    graph = data.load_tracks_graph()
    tracks, images = matching.tracks_and_images(graph)

    target_images = data.config.get('target_images', [] )
    
    if target_images:
        images = target_images

    chrono.lap('load_tracks_graph')
    
    gcp = None
    if data.ground_control_points_exist():
        gcp = data.load_ground_control_points()
    common_tracks = matching.all_common_tracks(graph, tracks)
    reconstructions = []
    pairs = compute_image_pairs(common_tracks, data)
    chrono.lap('compute_image_pairs')
    report['num_candidate_image_pairs'] = len(pairs)
    report['reconstructions'] = []

    full_images = list(set(images))

    full_images.sort()

    image_groups = []

    # Breakup image sets along specified GPS locations

    gps_points_dict = {}
    if data.gps_points_exist():
        gps_points_dict = data.load_gps_points()
        split_images = []
        for img in full_images:
            split_images.append( img )
            gps_info = gps_points_dict.get( img )
            if gps_info is not None and gps_info[4]:
                image_groups.append( split_images )
                split_images = [ img ]

        image_groups.append( split_images )

    else:
        image_groups.append( full_images )

    for remaining_images in image_groups:

        for im1, im2 in pairs:
            if im1 in remaining_images and im2 in remaining_images:
                rec_report = {}
                report['reconstructions'].append(rec_report)
                tracks, p1, p2 = common_tracks[im1, im2]
                reconstruction, rec_report['bootstrap'] = bootstrap_reconstruction(
                    data, graph, im1, im2, p1, p2)

                if reconstruction:
                    remaining_images.remove(im1)
                    remaining_images.remove(im2)
                    reconstruction, rec_report['grow'] = grow_reconstruction(
                        data, graph, reconstruction, remaining_images, gcp)
                    reconstructions.append(reconstruction)
                    reconstructions = sorted(reconstructions,
                                             key=lambda x: -len(x.shots))
                    data.save_reconstruction(reconstructions)

    for k, r in enumerate(reconstructions):
        logger.info("Reconstruction {}: {} images, {} points".format(
            k, len(r.shots), len(r.points)))
    logger.info("{} partial reconstructions in total.".format(
        len(reconstructions)))
    chrono.lap('compute_reconstructions')
    report['wall_times'] = dict(chrono.lap_times())
    report['not_reconstructed_images'] = list(remaining_images)
    return report


def remove_sky_features(data, graph):
    """
    remove features that has pitch angle above horizon. also remove
    features that appear in more than 100 images (which are likely
    features in the sky or from far away buildings)
    :param data:
    :param graph:
    :return:
    """
    try:
        cameras = data.load_camera_models()
        camera = cameras[data.load_exif('0000000000.jpg')['camera']]
    except:
        return

    tracks = []
    for n in graph.nodes(data=True):
        if n[1]['bipartite'] == 1:
            tracks.append(n[0])

    for track in tracks:
        if len(graph[track]) > 100:
            graph.remove_node(track)
            continue

        bad_count = 0
        for shot_id in graph[track]:
            x = graph[track][shot_id]['feature']
            b = camera.pixel_bearing(x)

            if b[1] < -0.00:
                bad_count += 1

            if bad_count > len(graph[track])/2:
                graph.remove_node(track)
                break


def incremental_reconstruction_sequential(data):
    """Run the entire incremental reconstruction pipeline."""
    logger.info("Starting incremental reconstruction sequentially")
    report = {}
    chrono = Chronometer()
    if not data.reference_lla_exists() and not data.config['use_provided_reference_lla']:
        data.invent_reference_lla()

    graph = data.load_tracks_graph()

    if 'remove_sky_features' in data.config and data.config['remove_sky_features']:
        remove_sky_features(data, graph)

    tracks, images = matching.tracks_and_images(graph)

    target_images = data.config.get('target_images', [] )
    
    if target_images:
        images = target_images

    chrono.lap('load_tracks_graph')
    
    gcp = None
    if data.ground_control_points_exist():
        gcp = data.load_ground_control_points()

    common_tracks = matching.all_common_tracks(graph, tracks)

    # debug - print # of common tracks between adjacent images
    for (im1, im2), (tracks, p1, p2) in iteritems(common_tracks):
        if _shot_id_to_int(im2) - _shot_id_to_int(im1) == 1:
            logger.info("({}, {}, #={}".format(im1, im2, len(tracks)))

    reconstructions = []
    report['reconstructions'] = []
    image_groups = []

    full_images = list(set(images))
    full_images.sort()

    # Breakup image sets along specified GPS locations (no longer used, but kept code)

    if data.gps_points_exist():
        gps_points_dict = data.load_gps_points()
        split_images = []
        for img in full_images:
            split_images.append( img )
            gps_info = gps_points_dict.get( img )
            if gps_info is not None and gps_info[4]:
                image_groups.append( split_images )
                split_images = [ img ]

        image_groups.append( split_images )

    else:
        image_groups.append( full_images )

    # load pdr data and globally align with gps points
    init_pdr_predictions(data)

    for remaining_images in image_groups:
        curr_idx = 0
        while curr_idx < len(remaining_images) - 1 > 0:
            im1 = remaining_images[curr_idx]
            im2 = remaining_images[curr_idx+1]

            if (im1, im2) not in common_tracks:
                curr_idx += 1
                continue

            rec_report = {}
            report['reconstructions'].append(rec_report)
            tracks, p1, p2 = common_tracks[im1, im2]

            reconstruction, rec_report['bootstrap'] = bootstrap_reconstruction(
                data, graph, im1, im2, p1, p2)

            if reconstruction:
                remaining_images.remove(im1)
                remaining_images.remove(im2)
                reconstruction, rec_report['grow'] = grow_reconstruction_sequential(
                    data, graph, reconstruction, common_tracks, remaining_images, gcp)
                reconstructions.append(reconstruction)

                #debug_rotation_prior(reconstruction, data)

                curr_idx = 0
            else:
                # bootstrap didn't work, try the next pair
                curr_idx += 1

    if reconstructions:
        if len(target_images) == 0:
            # only tries pdr alignment when we are not subsetting
            align_reconstructions_to_pdr(reconstructions, data)

        reconstructions = sorted(reconstructions, key=lambda x: -len(x.shots))
        data.save_reconstruction(reconstructions)

        uneven_images = []
        uneven_images_thresh = 3 / data.config['reconstruction_scale_factor']

        num_aligned = 0
        for k, r in enumerate(reconstructions):
            logger.info("Reconstruction {}: {} images, {} points, aligned = {}, num_corrs = {},".format(
                k, len(r.shots), len(r.points), r.alignment.aligned, r.alignment.num_correspondences))

            if r.alignment.aligned:
                num_aligned += len(r.shots)

                for shot_id in r.shots:
                    if abs(r.shots[shot_id].pose.get_origin()[2]) > uneven_images_thresh:
                        uneven_images.append(shot_id)

        coverage = int(100 * num_aligned / len(full_images))
        logger.info("{} partial reconstructions in total. {}% images aligned".format(len(reconstructions), coverage))

        if len(uneven_images) > 0:
            logger.info("{} images may not be level: {}".format(len(uneven_images), uneven_images))

    chrono.lap('compute_reconstructions')
    report['wall_times'] = dict(chrono.lap_times())
    report['not_reconstructed_images'] = list(remaining_images)
    return report


def breakup_reconstruction(graph, reconstruction):

    recon_points = set(reconstruction.points)

    shot_points = {}
    for shot_id in reconstruction.shots:
        shot_points[shot_id] = recon_points & set(graph[shot_id])

    cliques = []

    curr_idx = 0
    remaining_images = list(reconstruction.shots.keys())
    while curr_idx < len(remaining_images) - 1:
        im1 = remaining_images[curr_idx]
        im2 = remaining_images[curr_idx+1]

        ok = len(shot_points[im1] & shot_points[im2]) > 5
        if ok:
            clique = [im1, im2]
            clique_points = shot_points[im1] | shot_points[im2]

            remaining_images.remove(im1)
            remaining_images.remove(im2)

            while True:
                for im in remaining_images:
                    ok = len(set(clique_points) & shot_points[im]) > 5

                    if not ok:
                        continue

                    clique.append(im)
                    clique_points = list(set(clique_points) | shot_points[im])

                    remaining_images.remove(im)
                    break
                else:
                    break

            cliques.append(clique)

            curr_idx = 0
        else:
            curr_idx += 1

    logger.info("cliques={}".format(len(cliques)))


class Chronometer:
    def __init__(self):
        self.start()

    def start(self):
        t = timer()
        lap = ('start', 0, t)
        self.laps = [lap]
        self.laps_dict = {'start': lap}

    def lap(self, key):
        t = timer()
        dt = t - self.laps[-1][2]
        lap = (key, dt, t)
        self.laps.append(lap)
        self.laps_dict[key] = lap

    def lap_time(self, key):
        return self.laps_dict[key][1]

    def lap_times(self):
        return [(k, dt) for k, dt, t in self.laps[1:]]

    def total_time(self):
        return self.laps[-1][2] - self.laps[0][2]
