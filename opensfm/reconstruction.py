# -*- coding: utf-8 -*-
"""Incremental reconstruction pipeline"""

import datetime
import logging
import math
from collections import defaultdict
from itertools import combinations

import cv2
import numpy as np
import networkx as nx
import pyopengv
import six
from timeit import default_timer as timer
from six import iteritems

from opensfm import csfm
from opensfm import geo
from opensfm import log
from opensfm import tracking
from opensfm import multiview
from opensfm import types
from opensfm.align import align_reconstruction, apply_similarity
from opensfm.align_pdr import init_pdr_predictions, direct_align_pdr, hybrid_align_pdr, align_reconstruction_to_pdr
from opensfm.align_pdr import update_pdr_prediction_position
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
        ba.add_shot(shot.id, shot.camera.id, r, t, False)

    for point in reconstruction.points.values():
        ba.add_point(point.id, point.coordinates, False)

    for shot_id in reconstruction.shots:
        if shot_id in graph:
            for track in graph[shot_id]:
                if track in reconstruction.points:
                    point = graph[shot_id][track]['feature']
                    scale = graph[shot_id][track]['feature_scale']
                    ba.add_point_projection_observation(
                            shot_id, track, point[0], point[1], scale)

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

    ba.set_point_projection_loss_function(config['loss_function'], 
            config['loss_function_threshold'])

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
        shot.pose.rotation = [s.r[0], s.r[1], s.r[2]]
        shot.pose.translation = [s.t[0], s.t[1], s.t[2]]

    for point in reconstruction.points.values():
        p = ba.get_point(str(point.id))
        point.coordinates = [p.p[0], p.p[1], p.p[2]]
        point.reprojection_errors = p.reprojection_errors

    chrono.lap('teardown')

    logger.debug(ba.brief_report())
    report = {
        'wall_times': dict(chrono.lap_times()),
        'brief_report': ba.brief_report(),
    }
    return report


def bundle_single_view(graph, reconstruction, shot_id, data):
    """Bundle adjust a single camera."""
    config = data.config
    ba = csfm.BundleAdjuster()
    shot = reconstruction.shots[shot_id]
    camera = shot.camera

    _add_camera_to_bundle(ba, camera, constant=True)

    r = shot.pose.rotation
    t = shot.pose.translation
    ba.add_shot(shot.id, camera.id, r, t, False)

    for track_id in graph[shot_id]:
        if track_id in reconstruction.points:
            track = reconstruction.points[track_id]
            ba.add_point(track_id, track.coordinates, True)
            point = graph[shot_id][track_id]['feature']
            scale = graph[shot_id][track_id]['feature_scale']
            ba.add_point_projection_observation(
                shot_id, track_id, point[0], point[1], scale)

    if config['bundle_use_gps']:
        g = shot.metadata.gps_position
        ba.add_position_prior(str(shot.id), g[0], g[1], g[2],
                              shot.metadata.gps_dop)

    if config['bundle_use_pdr'] and shot.metadata.gps_dop == 999999.0:
        p, stddev1 = update_pdr_prediction_position(shot.id, reconstruction, data)
        ba.add_position_prior(str(shot.id), p[0], p[1], p[2], stddev1)

    ba.set_point_projection_loss_function(config['loss_function'],
            config['loss_function_threshold'])

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
    shot.pose.rotation = [s.r[0], s.r[1], s.r[2]]
    shot.pose.translation = [s.t[0], s.t[1], s.t[2]]


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
        ba.add_shot(shot.id, shot.camera.id, r, t, shot.id in boundary)

    for point_id in point_ids:
        point = reconstruction.points[point_id]
        ba.add_point(point.id, point.coordinates, False)

    for shot_id in interior | boundary:
        if shot_id in graph:
            for track in graph[shot_id]:
                if track in point_ids:
                    point = graph[shot_id][track]['feature']
                    scale = graph[shot_id][track]['feature_scale']
                    ba.add_point_projection_observation(
                            shot_id, track, point[0], point[1], scale)

    if config['bundle_use_gps']:
        for shot_id in interior:
            shot = reconstruction.shots[shot_id]
            g = shot.metadata.gps_position
            ba.add_position_prior(str(shot.id), g[0], g[1], g[2],
                                  shot.metadata.gps_dop)

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

    ba.set_point_projection_loss_function(config['loss_function'], 
            config['loss_function_threshold'])
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
        shot.pose.rotation = [s.r[0], s.r[1], s.r[2]]
        shot.pose.translation = [s.t[0], s.t[1], s.t[2]]

    for point in point_ids:
        point = reconstruction.points[point]
        p = ba.get_point(str(point.id))
        point.coordinates = [p.p[0], p.p[1], p.p[2]]
        point.reprojection_errors = p.reprojection_errors

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
    return point_ids, report


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
        if shot_id not in graph:
            continue
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
    T = multiview.relative_pose_ransac(
        b1, b2, "STEWENIUS", 1 - np.cos(threshold), 1000, 0.999)
    R = T[:, :3]
    t = T[:, 3]
    inliers = _two_view_reconstruction_inliers(b1, b2, R, t, threshold)

    if inliers.sum() > 5:
        T = multiview.relative_pose_optimize_nonlinear(b1[inliers], b2[inliers], t, R)
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

    R = multiview.relative_pose_ransac_rotation_only(
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
        return None, None, report

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

    graph_inliers = nx.Graph()
    triangulate_shot_features(graph, graph_inliers, reconstruction, im1, data.config)

    logger.info("Triangulated: {}".format(len(reconstruction.points)))
    report['triangulated_points'] = len(reconstruction.points)

    if len(reconstruction.points) < min_inliers:
        report['decision'] = "Initial motion did not generate enough points"
        logger.info(report['decision'])
        return None, None, report

    bundle_single_view(graph_inliers, reconstruction, im2, data)
    retriangulate(graph, graph_inliers, reconstruction, data.config)
    if len(reconstruction.points) < min_inliers:
         report['decision'] = "Re-triangulation after initial motion did not generate enough points"
         logger.info(report['decision'])
         return None, None, report
    bundle_single_view(graph_inliers, reconstruction, im2, data)

    report['decision'] = 'Success'
    report['memory_usage'] = current_memory_usage()
    return reconstruction, graph_inliers, report


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


def reconstructed_points_for_images_sequential(graph, reconstruction, images, last_image):
    """Number of reconstructed points visible on each image.

    Returns:
        A list of (image, num_point) pairs sorted by decreasing number of points.
        if a preferred image, i.e., the next or the previous image, exists, it
        will be placed at top of the list
    """
    next_image = _next_shot_id(last_image)
    prev_image = _prev_shot_id(last_image)
    res = []
    for image in images:
        if image not in reconstruction.shots:
            common_tracks = 0
            for track in graph[image]:
                if track in reconstruction.points:
                    common_tracks += 1

            # we prefer doing reconstruction sequentially, when the link is not very weak
            if image == next_image:
                common_tracks = 10000
            elif image == prev_image:
                common_tracks = 9999

            res.append((image, common_tracks))
    return sorted(res, key=lambda x: -x[1])


def resect(data, graph, graph_inliers, reconstruction, shot_id,
    camera, metadata, threshold, min_inliers):
    """Try resecting and adding a shot to the reconstruction.

    Return:
        True on success.
    """
    bs, Xs, ids = [], [], []
    for track in graph[shot_id]:
        if track in reconstruction.points:
            x = graph[track][shot_id]['feature']
            b = camera.pixel_bearing(x)
            bs.append(b)
            Xs.append(reconstruction.points[track].coordinates)
            ids.append(track)
    bs = np.array(bs)
    Xs = np.array(Xs)

    if len(bs) < 6:
        return False, {'num_common_points': len(bs)}

    T = multiview.absolute_pose_ransac(
        bs, Xs, "EPNP", 1 - np.cos(threshold), 1000, 0.999)

    R = T[:, :3]
    t = T[:, 3]

    reprojected_bs = R.T.dot((Xs - t).T).T
    reprojected_bs /= np.linalg.norm(reprojected_bs, axis=1)[:, np.newaxis]

    inliers = np.linalg.norm(reprojected_bs - bs, axis=1) < threshold
    ninliers = int(sum(inliers))

    #min_inliers = get_resection_min_inliers(data, graph, reconstruction, shot_id, ids, inliers, ninliers)

    #logger.info("{} resection inliers: {} / {}, threshod {}".format(
        #shot_id, ninliers, len(bs), min_inliers))
    report = {
        'num_common_points': len(bs),
        'num_inliers': ninliers,
    }

    if ninliers >= min_inliers:
        try:
            R = T[:, :3].T
            t = -R.dot(T[:, 3])
            shot = types.Shot()
            shot.id = shot_id
            shot.camera = camera
            shot.pose = types.Pose()
            shot.pose.set_rotation_matrix(R)
            shot.pose.translation = t
            shot.metadata = metadata
        except:
            return False, report

        reconstruction.add_shot(shot)
        for i, succeed in enumerate(inliers):
             if succeed:
                 copy_graph_data(graph, graph_inliers, shot_id, ids[i])
        bundle_single_view(graph_inliers, reconstruction, shot_id, data)
        return True, report
    else:
        #return resect_structureless(data, graph, reconstruction, shot_id)
        return False, report


def rotation_close_to_preint(im1, im2, two_view_rel_rot, pdr_shots_dict):
    """
    compare relative rotation of robust matching to that of imu gyro preintegration,
    if they are not close, it is considered to be an erroneous epipoar geometry
    """
    # calculate relative rotation from preintegrated gyro input
    preint_im1_rot = cv2.Rodrigues(np.asarray([pdr_shots_dict[im1][7], pdr_shots_dict[im1][8], pdr_shots_dict[im1][9]]))[0]
    preint_im2_rot = cv2.Rodrigues(np.asarray([pdr_shots_dict[im2][7], pdr_shots_dict[im2][8], pdr_shots_dict[im2][9]]))[0]
    preint_rel_rot = np.dot(preint_im2_rot, preint_im1_rot.T)

    # convert this rotation from sensor frame to camera frame
    b_to_c = np.asarray([1, 0, 0, 0, 0, -1, 0, 1, 0]).reshape(3, 3)
    preint_rel_rot = cv2.Rodrigues(b_to_c.dot(cv2.Rodrigues(preint_rel_rot)[0].ravel()))[0]

    diff_rot = np.dot(preint_rel_rot, two_view_rel_rot.T)
    geo_diff = np.linalg.norm(cv2.Rodrigues(diff_rot)[0].ravel())

    if geo_diff < math.pi/6.0:
        logger.debug("{} {} preint/robust geodesic {} within threshold".format(im1, im2, geo_diff))
        return True
    else:
        logger.debug("{} {} preint/robust geodesic {} exceeds threshold".format(im1, im2, geo_diff))
        return False


def resect_structureless(data, graph, reconstruction, shot_id):
    """
    Try resecting using structureless resection and adding a shot to the reconstruction.

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

    # reload common_tracks - we do structureless resection rarely so this shouldn't be
    # too bad on performance
    tracks, images = tracking.tracks_and_images(graph)
    common_tracks = tracking.all_common_tracks(graph, tracks)

    for image_a in reconstruction.shots.keys():
        if image_a != shot_id:

            tracks, pcur, pa = get_common_tracks(shot_id, image_a, common_tracks)

            num_tracks = len(tracks)
            if num_tracks > max_common_tracks:
                max_common_tracks = num_tracks
                image_one = image_a
                pr = pcur
                p1r = pa

    logger.info('image: ' + shot_id)
    logger.info('image_one: ' + str(image_one))

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

                tracks02, pcur02, pb02 = get_common_tracks(shot_id, image_b, common_tracks)
                tracks01, pcur01, pb01 = get_common_tracks(image_one, image_b, common_tracks)

                num_common_to_both = len(set(tracks01).intersection(set(tracks02)))

                if num_common_to_both > max_common_to_both:
                    max_common_to_both = num_common_to_both
                    image_two = image_b
                    pa = pcur02
                    p2a = pb02

    logger.info('image_two: ' + str(image_two))

    if image_one is None:
        return False, {'num_common_points_one': 0, 'num_common_points_two': 0, 'relative_result': "Failure"}
    elif image_two is None:
        return False, {'num_common_points_one': len(pr), 'num_common_points_two': 0, 'relative_result': "Failure"}

    logger.info('num of common points 0/1={}, 0/2={}, 0/1/2={}'.format(len(pr), len(pa), max_common_to_both))

    logger.info("Computing relative motion with {} and {}".format(shot_id, image_one))

    report = {
        'image_pair_relative': (shot_id, image_one),
        'num_common_points_one': len(pr),
        'num_common_points_two': len(p2a),
    }

    cameras = data.load_camera_models()
    camera = cameras[data.load_exif(shot_id)['camera']]
    camera1 = cameras[data.load_exif(image_one)['camera']]

    threshold = data.config['five_point_algo_threshold']
    min_inliers = data.config['five_point_algo_min_inliers']

    R_rel, t, inliers, report['image_one_relative'] = \
        two_view_reconstruction_general(pr, p1r, camera, camera1, threshold)

    R_rel = cv2.Rodrigues(R_rel)[0]

    logger.info("Two-view relative inliers: {} / {}".format(len(inliers), len(pr)))

    if len(inliers) <= 5:
        report['relative_result'] = "Could not find relative motion"
        logger.info(report['relative_result'])
        return False, report

    report['relative_result'] = "Success"

    logger.info("Computing absolute motion with {}, {} and {}".format(shot_id, image_one, image_two))

    shot1 = reconstruction.shots[image_one]

    R1 = shot1.pose.get_rotation_matrix()
    o1 = shot1.pose.get_origin()

    b0_cam = camera.pixel_bearing_many(pa)

    camera2 = cameras[data.load_exif(image_two)['camera']]
    shot2 = reconstruction.shots[image_two]

    R2 = shot2.pose.get_rotation_matrix()
    o2 = shot2.pose.get_origin()
    b2_cam = camera2.pixel_bearing_many(p2a)

    b2_wrld = np.dot(b2_cam, R2.T)

    # The combination of shot one rotation and the relative rotation to the shot being considered.
    R0 = np.dot(R1, R_rel)

    # Translation is an unscaled unit vector in world coordinates.
    t = R1.T.dot(t)

    t /= np.linalg.norm(t)

    # print( "Resect Structureless R0: " + str(R0) )
    # print( "Resect Structureless t (in): " + str(t) )
    # print( "Resect Structureless rod: " +str( cv2.Rodrigues(R0)[0].ravel() ) )

    threshold = data.config['resection_threshold']

    bs, Xs, ids = [], [], []
    for track in graph[shot_id]:
        if track in reconstruction.points:
            x = graph[track][shot_id]['feature']
            b = camera.pixel_bearing(x)
            bs.append(b)
            Xs.append(reconstruction.points[track].coordinates)
            ids.append(track)
    bs = np.array(bs)
    Xs = np.array(Xs)

    # Start by using relative motion for rotation and then estimate translation direction
    # and magnitude using triangulated track points.

    twopt_success = True

    if len(bs) < 2:
        report['absolute_result_twopt'] = "Could not find absolute translation"
        twopt_success = False
    
    else:
    
        T = pyopengv.absolute_pose_twopt_ransac( bs, Xs, R0.T, 1 - np.cos(threshold), 1000, 0.999 )
        
        R2pt = T[:, :3]
        t2pt = T[:, 3]

        reprojected_bs = R2pt.T.dot((Xs - t2pt).T).T
        reprojected_bs /= np.linalg.norm(reprojected_bs, axis=1)[:, np.newaxis]

        inliers = np.linalg.norm(reprojected_bs - bs, axis=1) < threshold
        ninliers = int(sum(inliers))

        report['num_inliers_twopt'] = ninliers
        
        if ninliers < 4:
            report['absolute_result_twopt'] = "Could not find absolute translation"
            logger.info("Resection twopt failed with {}, {} and {}. Number of Inliers: {}".format(shot_id, image_one, image_two, str(ninliers)) )
            twopt_success = False
        else:
            report['absolute_result_twopt'] = "Success"
            logger.info("Resection twopt succeeded with {}, {} and {}. Number of Inliers: {}".format(shot_id, image_one, image_two, str(ninliers)) )
    
    # If the two point algorithm doesn't succeed try the one point algorithm using triangulated track points
    
    onept_success = True
    
    if not twopt_success:
    
        T = pyopengv.absolute_pose_onept_ransac(bs, Xs, o1, R0.T, t, 1 - np.cos(threshold), 1000, 0.999)
        
        # Assess success here. For now we mandate a couple of inliers and try to be strict since we
        # have structureless as a backup.
        
        if np.allclose(T[:, 3], np.zeros(3)):
            report['absolute_result_onept'] = "Could not find absolute translation"
            onept_success = False
            logger.info("Resection onept failed with {}, {} and {}".format(shot_id, image_one, image_two))
        else:
            report['absolute_result_onept'] = "Success"
            logger.info("Resection onept succeeded with {}, {} and {}".format(shot_id, image_one, image_two))
        
    if not onept_success:
    
        # Due to row major (OpenCV) to column major (Eigen) take the transpose of R0. t is a direction in world coordinates
        # so it should be uneffected.

        T = pyopengv.absolute_pose_onept_structureless_ransac(b0_cam, b2_wrld, o1, o2, R0.T, t, 1 - np.cos(threshold), 1000, 0.999)

        # Failure is indicated by a translation value of the zero vector
        if np.allclose(T[:, 3], np.zeros(3)):
            report['absolute_result_onept_structureless'] = "Could not find absolute translation"
            logger.info("Structureless resection failed with {}, {} and {}".format(shot_id, image_one, image_two))
            return False, report
        else:
            logger.info("Structureless resection succeeded with {}, {} and {}".format(shot_id, image_one, image_two))
            report['absolute_result_onept_structureless'] = "Success"

    # debugging
    if data.pdr_shots_exist():
        pdr_shots_dict = data.load_pdr_shots()
        R = T[:, :3].T
        logger.info("test 0")
        rel1 = np.dot(R.T, R1)
        rotation_close_to_preint(shot_id, image_one, rel1, pdr_shots_dict)
        logger.info("test 1")
        rel2 = np.dot(R.T, R2)
        rotation_close_to_preint(shot_id, image_two, rel2, pdr_shots_dict)
    
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
        logger.info("Structureless resection failed (exception) with {}, {} and {}".format(shot_id, image_one, image_two))
        return False, report

    reconstruction.add_shot(shot)
    bundle_single_view(graph, reconstruction, shot_id, data)

    logger.info("Resection successful with {}, {} and {}".format(shot_id, image_one, image_two))
    return True, report


def get_common_tracks(image_a, image_b, common_tracks):
    return common_tracks.get(tuple(sorted([image_a, image_b])), ([], [], []))


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


def corresponding_tracks(tracks1, tracks2):
    features1 = {tracks1[t1]["feature_id"]: t1 for t1 in tracks1}
    corresponding_tracks = []
    for t2 in tracks2:
        feature_id = tracks2[t2]["feature_id"]
        if feature_id in features1:
            corresponding_tracks.append((features1[feature_id], t2))
    return corresponding_tracks


def compute_common_tracks(reconstruction1, reconstruction2,
                          graph1, graph2):
    common_tracks = set()
    common_images = set(reconstruction1.shots.keys()).intersection(
        reconstruction2.shots.keys())
    for image in common_images:
        for t1, t2 in corresponding_tracks(graph1[image], graph2[image]):
            if t1 in reconstruction1.points and t2 in reconstruction2.points:
                common_tracks.add((t1, t2))
    return list(common_tracks)


def resect_reconstruction(reconstruction1, reconstruction2, graph1,
                          graph2, threshold, min_inliers):
    common_tracks = compute_common_tracks(
        reconstruction1, reconstruction2, graph1, graph2)


    worked, similarity, inliers = align_two_reconstruction(
        reconstruction1, reconstruction2, common_tracks, threshold)
    if not worked:
        return False, [], []

    inliers = [common_tracks[inliers[i]] for i in range(len(inliers))]
    return True, similarity, inliers


def copy_graph_data(graph_from, graph_to, shot_id, track_id):
    if shot_id not in graph_to:
        graph_to.add_node(shot_id, bipartite=0)
    if track_id not in graph_to:
        graph_to.add_node(track_id, bipartite=1)
    edge_data = graph_from.get_edge_data(shot_id, track_id)
    graph_to.add_edge(shot_id, track_id,
                      feature=edge_data['feature'],
                      feature_scale=edge_data['feature_scale'],
                      feature_id=edge_data['feature_id'],
                      feature_color=edge_data['feature_color'])


class TrackTriangulator:
    """Triangulate tracks in a reconstruction.

    Caches shot origin and rotation matrix
    """

    def __init__(self, graph, graph_inliers, reconstruction):
        """Build a triangulator for a specific reconstruction."""
        self.graph = graph
        self.graph_inliers = graph_inliers
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

    def triangulate_robust(self, track, reproj_threshold, min_ray_angle_degrees):
        """Triangulate track in a RANSAC way and add point to reconstruction."""
        os, bs, ids = [], [], []
        for shot_id in self.graph[track]:
            if shot_id in self.reconstruction.shots:
                shot = self.reconstruction.shots[shot_id]
                os.append(self._shot_origin(shot))
                x = self.graph[track][shot_id]['feature']
                b = shot.camera.pixel_bearing(np.array(x))
                r = self._shot_rotation_inverse(shot)
                bs.append(r.dot(b))
                ids.append(shot_id)

        if len(ids) < 2:
            return

        best_inliers = []
        best_point = types.Point()
        best_point.id = track

        combinatiom_tried = set()
        ransac_tries = 11 # 0.99 proba, 60% inliers
        all_combinations = list(combinations(range(len(ids)), 2))

        thresholds = len(os) * [reproj_threshold]
        for i in range(ransac_tries):
            random_id = int(np.random.rand()*(len(all_combinations)-1))
            if random_id in combinatiom_tried:
                continue

            i, j = all_combinations[random_id]
            combinatiom_tried.add(random_id)

            os_t = [os[i], os[j]]
            bs_t = [bs[i], bs[j]]

            e, X = csfm.triangulate_bearings_midpoint(
                os_t, bs_t, thresholds, np.radians(min_ray_angle_degrees))

            if X is not None:
                reprojected_bs = X-os
                reprojected_bs /= np.linalg.norm(reprojected_bs, axis=1)[:, np.newaxis]
                inliers = np.linalg.norm(reprojected_bs - bs, axis=1) < reproj_threshold

                if sum(inliers) > sum(best_inliers):
                    best_inliers = inliers
                    best_point.coordinates = X.tolist()

                    pout = 0.99
                    inliers_ratio = float(sum(best_inliers))/len(ids)
                    if inliers_ratio == 1.0:
                        break
                    optimal_iter = math.log(1.0-pout)/math.log(1.0-inliers_ratio*inliers_ratio)
                    if optimal_iter <= ransac_tries:
                        break

        # the following line reads "if len(best_inliers) > 1:", which appears to be a bug. change to 'sum'
        if sum(best_inliers) > 1:
            self.reconstruction.add_point(best_point)
            for i, succeed in enumerate(best_inliers):
                if succeed:
                    self._add_track_to_graph_inlier(track, ids[i])

    def triangulate(self, track, reproj_threshold, min_ray_angle_degrees):
        """Triangulate track and add point to reconstruction."""
        os, bs, ids = [], [], []
        for shot_id in self.graph[track]:
            if shot_id in self.reconstruction.shots:
                shot = self.reconstruction.shots[shot_id]
                os.append(self._shot_origin(shot))
                x = self.graph[track][shot_id]['feature']
                b = shot.camera.pixel_bearing(np.array(x))
                r = self._shot_rotation_inverse(shot)
                bs.append(r.dot(b))
                ids.append(shot_id)

        if len(os) >= 2:
            thresholds = len(os) * [reproj_threshold]
            e, X = csfm.triangulate_bearings_midpoint(
                os, bs, thresholds, np.radians(min_ray_angle_degrees))

            if X is not None:
                point = types.Point()
                point.id = track
                point.coordinates = X.tolist()

                self.reconstruction.add_point(point)
                for shot_id in ids:
                     self._add_track_to_graph_inlier(track, shot_id)

    def triangulate_dlt(self, track, reproj_threshold, min_ray_angle_degrees):
        """Triangulate track using DLT and add point to reconstruction."""
        Rts, bs, ids = [], [], []
        for shot_id in self.graph[track]:
            if shot_id in self.reconstruction.shots:
                shot = self.reconstruction.shots[shot_id]
                Rts.append(self._shot_Rt(shot))
                x = self.graph[track][shot_id]['feature']
                b = shot.camera.pixel_bearing(np.array(x))
                bs.append(b)
                ids.append(shot_id)

        if len(Rts) >= 2:
            if len(Rts) >= 3:
                min_ray_angle_degrees /= 2.0

            e, X = csfm.triangulate_bearings_dlt(
                Rts, bs, reproj_threshold, np.radians(min_ray_angle_degrees))
            if X is not None:
                '''
                reprojected_bs = []
                for i in range(len(Rts)):
                    reprojected_bs.append(Rts[i][:, :3].dot(X) + Rts[i][:, 3])

                reprojected_bs /= np.linalg.norm(reprojected_bs, axis=1)[:, np.newaxis]
                logger.debug("reproj err {}".format(np.linalg.norm(reprojected_bs - bs, axis=1)))
                '''

                point = types.Point()
                point.id = track
                point.coordinates = X.tolist()
                self.reconstruction.add_point(point)
                for shot_id in ids:
                    self._add_track_to_graph_inlier(track, shot_id)

    def _add_track_to_graph_inlier(self, track_id, shot_id):
        copy_graph_data(self.graph, self.graph_inliers, shot_id, track_id)

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


def triangulate_shot_features(graph, graph_inliers, reconstruction, shot_id, config):
    """Reconstruct as many tracks seen in shot_id as possible."""
    reproj_threshold = config['triangulation_threshold']
    min_ray_angle = config['triangulation_min_ray_angle']

    triangulator = TrackTriangulator(graph, graph_inliers, reconstruction)

    for track in graph[shot_id]:
        if track not in reconstruction.points:
            triangulator.triangulate_dlt(track, reproj_threshold, min_ray_angle)


def retriangulate(graph, graph_inliers, reconstruction, config):
    """Retrianguate all points"""
    chrono = Chronometer()
    report = {}
    report['num_points_before'] = len(reconstruction.points)
    threshold = config['triangulation_threshold']
    min_ray_angle = config['triangulation_min_ray_angle']

    graph_inliers.clear()
    reconstruction.points = {}

    triangulator = TrackTriangulator(graph, graph_inliers, reconstruction)

    tracks, images = tracking.tracks_and_images(graph)
    for track in tracks:
        if config['triangulation_type'] == 'ROBUST':
            triangulator.triangulate_robust(track, threshold, min_ray_angle)
        elif config['triangulation_type'] == 'FULL':
            triangulator.triangulate(track, threshold, min_ray_angle)
    report['num_points_after'] = len(reconstruction.points)
    chrono.lap('retriangulate')
    report['wall_time'] = chrono.total_time()
    return report


def get_error_distribution(points):
    all_errors = []
    for track in points.values():
        all_errors += track.reprojection_errors.values()
    robust_mean = np.median(all_errors, axis=0)
    robust_std = 1.486*np.median(np.linalg.norm(all_errors-robust_mean, axis=1))
    return robust_mean, robust_std


def get_actual_threshold(config, points):
    filter_type = config['bundle_outlier_filtering_type']
    if filter_type == 'FIXED':
        return config['bundle_outlier_fixed_threshold']
    elif filter_type == 'AUTO':
        mean, std = get_error_distribution(points)
        return config['bundle_outlier_auto_ratio']*np.linalg.norm(mean+std)
    else:
        return 1.0


def remove_outliers(graph, reconstruction, config, points=None):
    """Remove points with large reprojection error.
    A list of point ids to be processed can be given in ``points``.
    """
    if points is None:
         points = reconstruction.points
    threshold = get_actual_threshold(config, reconstruction.points)
    outliers = []
    for point_id in points:
        for shot_id, error in reconstruction.points[point_id].reprojection_errors.items():
            if np.linalg.norm(error) > threshold:
                outliers.append((point_id, shot_id))

    for track, shot_id in outliers:
        del reconstruction.points[track].reprojection_errors[shot_id]
        graph.remove_edge(track, shot_id)
    for track, _ in outliers:
        if track not in reconstruction.points:
            continue
        if len(graph[track]) < 2:
            del reconstruction.points[track]
            graph.remove_node(track)
    logger.info("Removed outliers: {}".format(len(outliers)))
    return len(outliers)


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


def align_two_reconstruction(r1, r2, common_tracks, threshold):
    """Estimate similarity transform between two reconstructions."""
    t1, t2 = r1.points, r2.points

    if len(common_tracks) > 6:
        p1 = np.array([t1[t[0]].coordinates for t in common_tracks])
        p2 = np.array([t2[t[1]].coordinates for t in common_tracks])

        # 3 samples / 100 trials / 50% outliers = 0.99 probability
        # with probability = 1-(1-(1-outlier)^model)^trial
        T, inliers = multiview.fit_similarity_transform(
            p1, p2, max_iterations=100, threshold=threshold)
        if len(inliers) > 0:
            return True, T, inliers
    return False, None, None


def merge_two_reconstructions(r1, r2, config, threshold=1):
    """Merge two reconstructions with common tracks IDs."""
    common_tracks = list(set(r1.points) & set(r2.points))
    worked, T, inliers = align_two_reconstruction(
        r1, r2, common_tracks, threshold)

    if worked and len(inliers) >= 10:
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


def grow_reconstruction_sequential(data, graph, graph_inliers, reconstruction, images, gcp):
    """Incrementally add shots to an initial reconstruction."""
    config = data.config
    report = {'steps': []}

    bundle(graph, reconstruction, None, config)
    remove_outliers(graph_inliers, reconstruction, config)
    align_reconstruction(reconstruction, gcp, config)

    should_bundle = ShouldBundle(data, reconstruction)
    should_retriangulate = ShouldRetriangulate(data, reconstruction)

    last_image = sorted(reconstruction.shots)[-1]
    while True:
        if config['save_partial_reconstructions']:
            paint_reconstruction(data, graph, reconstruction)
            data.save_reconstruction(
                [reconstruction], 'reconstruction.{}.json'.format(
                    datetime.datetime.now().isoformat().replace(':', '_')))

        candidates = reconstructed_points_for_images_sequential(
            graph, reconstruction, images, last_image)
        if not candidates:
            break

        logger.info("-------------------------------------------------------")
        threshold = data.config['resection_threshold']
        min_inliers = data.config['resection_min_inliers']
        for image, num_tracks in candidates:
            camera = reconstruction.cameras[data.load_exif(image)['camera']]
            metadata = get_image_metadata(data, image)
            ok, resrep = resect(data, graph, graph_inliers, reconstruction, image,
                                camera, metadata, threshold, min_inliers)
            if not ok:
                continue

            logger.info("Adding {} to the reconstruction, num_tracks {}".format(image, num_tracks))
            step = {
                'image': image,
                'resection': resrep,
                'memory_usage': current_memory_usage()
            }
            report['steps'].append(step)
            images.remove(image)

            np_before = len(reconstruction.points)
            triangulate_shot_features(graph, graph_inliers, reconstruction, image, config)
            np_after = len(reconstruction.points)
            step['triangulated_points'] = np_after - np_before
            logger.info("grow_reconstruction_sequential: {} points in the reconstruction".format(np_after))

            if should_retriangulate.should():
                logger.info("Re-triangulating")
                for iteration in range(3):
                    logger.info("grow_reconstruction_sequential: iteration refinement #{}".format(iteration))
                    b1rep = bundle(graph_inliers, reconstruction, None, config)
                    rrep = retriangulate(graph, graph_inliers, reconstruction, config)
                    b2rep = bundle(graph_inliers, reconstruction, None, config)
                    remove_outliers(graph_inliers, reconstruction, config)
                    if 'Termination: CONVERGENCE' in b2rep['brief_report']:
                        logger.info("grow_reconstruction_sequential: iteration refinement done")
                        break
                align_reconstruction(reconstruction, gcp, config)
                step['bundle'] = b1rep
                step['retriangulation'] = rrep
                step['bundle_after_retriangulation'] = b2rep
                should_retriangulate.done()
                should_bundle.done()
            elif should_bundle.should():
                logger.info("Global bundle adjustment")
                brep = bundle(graph_inliers, reconstruction, None, config)
                remove_outliers(graph_inliers, reconstruction, config)
                align_reconstruction(reconstruction, gcp, config)
                step['bundle'] = brep
                should_bundle.done()
            elif config['local_bundle_radius'] > 0:
                bundled_points, brep = bundle_local(
                    graph_inliers, reconstruction, None, image, data)
                remove_outliers(graph_inliers, reconstruction, config, bundled_points)
                step['local_bundle'] = brep

            last_image = image
            break
        else:
            break

        max_recon_size = config.get( 'reconstruction_max_images', -1 )
        if max_recon_size != -1:
            if len( reconstruction.shots ) >= max_recon_size:
                break

    logger.info("-------------------------------------------------------")

    bundle(graph_inliers, reconstruction, gcp, config)
    remove_outliers(graph_inliers, reconstruction, config)
    align_reconstruction(reconstruction, gcp, config)
    paint_reconstruction(data, graph, reconstruction)

    if len(images) > 0:
        # remaining images may be ordered quite randomly, so sort them
        images.sort()
        logger.info("{} images can not be added {}".format(len(images), images))

    return reconstruction, report


def grow_reconstruction(data, graph, graph_inliers, reconstruction, images, gcp):
    """Incrementally add shots to an initial reconstruction."""
    config = data.config
    report = {'steps': []}

    bundle(graph, reconstruction, None, config)
    remove_outliers(graph_inliers, reconstruction, config)
    align_reconstruction(reconstruction, gcp, config)

    should_bundle = ShouldBundle(data, reconstruction)
    should_retriangulate = ShouldRetriangulate(data, reconstruction)
    while True:
        if config['save_partial_reconstructions']:
            paint_reconstruction(data, graph, reconstruction)
            data.save_reconstruction(
                [reconstruction], 'reconstruction.{}.json'.format(
                    datetime.datetime.now().isoformat().replace(':', '_')))

        candidates = reconstructed_points_for_images(
            graph, reconstruction, images)
        if not candidates:
            break

        logger.info("-------------------------------------------------------")
        threshold = data.config['resection_threshold']
        min_inliers = data.config['resection_min_inliers']
        for image, num_tracks in candidates:
            camera = reconstruction.cameras[data.load_exif(image)['camera']]
            metadata = get_image_metadata(data, image)
            ok, resrep = resect(data, graph, graph_inliers, reconstruction, image,
                                camera, metadata, threshold, min_inliers)
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
            triangulate_shot_features(graph, graph_inliers, reconstruction, image, config)
            np_after = len(reconstruction.points)
            step['triangulated_points'] = np_after - np_before

            if should_retriangulate.should():
                logger.info("Re-triangulating")
                b1rep = bundle(graph_inliers, reconstruction, None, config)
                rrep = retriangulate(graph, graph_inliers, reconstruction, config)
                b2rep = bundle(graph_inliers, reconstruction, None, config)
                remove_outliers(graph_inliers, reconstruction, config)
                align_reconstruction(reconstruction, gcp, config)
                step['bundle'] = b1rep
                step['retriangulation'] = rrep
                step['bundle_after_retriangulation'] = b2rep
                should_retriangulate.done()
                should_bundle.done()
            elif should_bundle.should():
                brep = bundle(graph_inliers, reconstruction, None, config)
                remove_outliers(graph_inliers, reconstruction, config)
                align_reconstruction(reconstruction, gcp, config)
                step['bundle'] = brep
                should_bundle.done()
            elif config['local_bundle_radius'] > 0:
                bundled_points, brep = bundle_local(
                    graph_inliers, reconstruction, None, image, data)
                remove_outliers(graph_inliers, reconstruction, config, bundled_points)
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

    bundle(graph_inliers, reconstruction, gcp, config)
    remove_outliers(graph_inliers, reconstruction, config)
    align_reconstruction(reconstruction, gcp, config)
    paint_reconstruction(data, graph, reconstruction)
    return reconstruction, report


def hybrid_align_reconstruction_pdr( data ):

    target_images = data.config.get('target_images', [])

    report = {}
    report['reconstructions'] = []
    rec_report = {}
    report['reconstructions'].append(rec_report)
    rec_report['subset'] = target_images

    data.save_reconstruction(hybrid_align_pdr(data))

    return report


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


def _length_histogram(points, graph):
     hist = defaultdict(int)
     for p in points:
         hist[len(graph[p])] += 1
     return np.array(list(hist.keys())), np.array(list(hist.values()))


def compute_statistics(reconstruction, graph):
     stats = {}
     stats['points_count'] = len(reconstruction.points)
     stats['cameras_count'] = len(reconstruction.shots)

     hist, values = _length_histogram(reconstruction.points, graph)
     stats['observations_count'] = int(sum(hist * values))
     if len(reconstruction.points) > 0:
         stats['average_track_length'] = float(stats['observations_count']) / len(reconstruction.points)
     else:
         stats['average_track_length'] = -1
     tracks_notwo = sum([1 if len(graph[p]) > 2 else 0 for p in reconstruction.points])
     if tracks_notwo > 0:
         stats['average_track_length_notwo'] = float(sum(hist[1:]*values[1:]))/tracks_notwo
     else:
         stats['average_track_length_notwo'] = -1
     return stats


def incremental_reconstruction(data, graph):
    """Run the entire incremental reconstruction pipeline."""
    logger.info("Starting incremental reconstruction")
    report = {}
    chrono = Chronometer()

    tracks, images = tracking.tracks_and_images(graph)
    chrono.lap('load_tracks_graph')

    if not data.reference_lla_exists() and not data.config['use_provided_reference_lla']:
        data.invent_reference_lla()

    target_images = data.config.get('target_images', [] )
    
    if target_images:
        images = target_images

    gcp = None
    if data.ground_control_points_exist():
        gcp = data.load_ground_control_points()
    common_tracks = tracking.all_common_tracks(graph, tracks)
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
                reconstruction, graph_inliers, rec_report['bootstrap'] = bootstrap_reconstruction(
                    data, graph, im1, im2, p1, p2)

                if reconstruction:
                    remaining_images.remove(im1)
                    remaining_images.remove(im2)
                    reconstruction, rec_report['grow'] = grow_reconstruction(
                        data, graph, graph_inliers, reconstruction, remaining_images, gcp)
                    reconstructions.append(reconstruction)
                    reconstructions = sorted(reconstructions,
                                             key=lambda x: -len(x.shots))
                    rec_report['stats'] = compute_statistics(reconstruction, graph_inliers)
                    logger.info(rec_report['stats'])
                    data.save_reconstruction(reconstructions)

    for k, r in enumerate(reconstructions):
        logger.info("Reconstruction {}: {} images, {} points".format(
            k, len(r.shots), len(r.points)))
    logger.info("{} partial reconstructions in total.".format(
        len(reconstructions)))
    chrono.lap('compute_reconstructions')
    report['wall_times'] = dict(chrono.lap_times())
    report['not_reconstructed_images'] = list(remaining_images)
    return report, reconstructions


def incremental_reconstruction_sequential(data, graph):
    """Run the entire incremental reconstruction pipeline."""
    logger.info("Starting incremental reconstruction sequentially")
    report = {}
    chrono = Chronometer()

    tracks, images = tracking.tracks_and_images(graph)
    chrono.lap('load_tracks_graph')

    if not data.reference_lla_exists() and not data.config['use_provided_reference_lla']:
        data.invent_reference_lla()

    target_images = data.config.get('target_images', [] )
    
    if target_images:
        images = target_images

    gcp = None
    if data.ground_control_points_exist():
        gcp = data.load_ground_control_points()
    common_tracks = tracking.all_common_tracks(graph, tracks)
    reconstructions = []
    pairs = compute_image_pairs(common_tracks, data)
    chrono.lap('compute_image_pairs')
    report['num_candidate_image_pairs'] = len(pairs)
    report['reconstructions'] = []

    full_images = list(set(images))
    full_images.sort()
    cnt_images = len(full_images)

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

    # load pdr data and globally align with gps points, if any
    if data.pdr_shots_exist():
        init_pdr_predictions(data)

    # select as seed the pair with largest number of common tracks
    for remaining_images in image_groups:

        for im1, im2 in pairs:
            if im1 in remaining_images and im2 in remaining_images:
                rec_report = {}
                report['reconstructions'].append(rec_report)
                tracks, p1, p2 = common_tracks[im1, im2]
                reconstruction, graph_inliers, rec_report['bootstrap'] = bootstrap_reconstruction(
                    data, graph, im1, im2, p1, p2)

                if reconstruction:
                    remaining_images.remove(im1)
                    remaining_images.remove(im2)
                    reconstruction, rec_report['grow'] = grow_reconstruction_sequential(
                        data, graph, graph_inliers, reconstruction, remaining_images, gcp)
                    reconstructions.append(reconstruction)
                    reconstructions = sorted(reconstructions,
                                             key=lambda x: -len(x.shots))
                    rec_report['stats'] = compute_statistics(reconstruction, graph_inliers)
                    logger.info(rec_report['stats'])

    if reconstructions:
        if data.pdr_shots_exist():
            # level and scale recons to pdr
            reconstructions[:] = [align_reconstruction_to_pdr(recon, data) for recon in reconstructions]

            # remove frames from recons that are obviously wrong according to pdr
            remove_bad_frames(reconstructions)

        reconstructions = sorted(reconstructions, key=lambda x: -len(x.shots))
        data.save_reconstruction(reconstructions)

        # for gps picker tool, because it doesn't need point cloud, we save the reconstructions without points.
        # this cuts down the time it needs to download and parse.
        data.save_reconstruction_no_point(reconstructions)

        # for gps picker tool, calculate and save a recon quality factor. pdr/hybrid will be based on it.
        if cnt_images <= 100:
            if len(reconstructions) == 1:
                quality_factor = 100
            else:
                quality_factor = 0
        else:
            cnt_large_recon = 0
            for recon in reconstructions:
                if len(recon.shots) > 100:
                    cnt_large_recon += len(recon.shots)

            quality_factor = int(100 * cnt_large_recon / cnt_images)
        data.save_recon_quality(str(quality_factor))

    chrono.lap('compute_reconstructions')
    report['wall_times'] = dict(chrono.lap_times())
    report['not_reconstructed_images'] = list(remaining_images)
    return report, reconstructions


def remove_bad_frames(reconstructions):
    height_thresh = 1.0
    distance_thresh = 2.0

    empty_recons = []
    for k, r in enumerate(reconstructions):
        logger.info("Reconstruction {}: {} images".format(k, len(r.shots)))

        uneven_images = []
        suspicious_images = []

        for shot_id in r.shots:
            if abs(r.shots[shot_id].pose.get_origin()[2]) > height_thresh:
                uneven_images.append(shot_id)
                continue

            next_shot_id = _next_shot_id(shot_id)
            prev_shot_id = _prev_shot_id(shot_id)
            if next_shot_id in r.shots and prev_shot_id in r.shots:
                d1 = np.linalg.norm(r.shots[next_shot_id].pose.get_origin() - r.shots[shot_id].pose.get_origin())
                d2 = np.linalg.norm(r.shots[prev_shot_id].pose.get_origin() - r.shots[shot_id].pose.get_origin())
                if d1 > distance_thresh and d2 > distance_thresh:
                    suspicious_images.append(shot_id)

        for shot_id in uneven_images:
            logger.info("removing uneven image: {}".format(shot_id))
            r.shots.pop(shot_id, None)

        for shot_id in suspicious_images:
            logger.info("removing suspicious image: {}".format(shot_id))
            r.shots.pop(shot_id, None)

        if len(r.shots) == 0:
            empty_recons.append(r)

    for r in empty_recons:
        reconstructions.remove(r)


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
