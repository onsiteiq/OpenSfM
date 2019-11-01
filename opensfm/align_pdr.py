"""affine transform pdr predictions to align with GPS points or SfM output."""

import os
import operator
import logging
import math
import numpy as np
from cmath import rect, phase

from itertools import combinations

from opensfm import geo
from opensfm import multiview
from opensfm import types
from opensfm import transformations as tf

from opensfm.debug_plot import debug_plot_pdr

logger = logging.getLogger(__name__)


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

    # debug
    debug_plot_pdr(topocentric_gps_points_dict, pdr_predictions_dict)

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

    :param curr_gps_points:
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

    if len(curr_gps_points_dict) < 2:
        if len(curr_gps_points_dict) < 1:
            offset = (2000, 2000, 0)
            num = 1
        else:
            for shot_id in curr_gps_points_dict:
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


def test_geo_hash(reconstruction, data):
    if not data.pdr_shots_exist():
        return

    pdr_shots_dict = data.load_pdr_shots()

    X, Xp = [], []
    onplane, verticals = [], []
    for shot_id in reconstruction.shots.keys():
        X.append(reconstruction.shots[shot_id].pose.get_origin())
        Xp.append(pdr_shots_dict[shot_id])
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


def export_coordinates(reconstructions):
    with open(os.path.join('.', 'test_geo_hash.txt'), 'w') as f:
        for reconstruction in reconstructions:
            f.write("new reconstruction")
            for shot_id in reconstruction.shots.keys():
                o = reconstruction.shots[shot_id].pose.get_origin()
                f.write("{} {} {}\n".format(o[0], o[1], shot_id))


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


def cull_resection_pdr(shot_id, reconstruction, data, bs, Xs, track_ids):
    """
    use pdr rotation and position predictions to mask out questionable features before resection
    :param shot_id:
    :param reconstruction:
    :param data:
    :param bs:
    :param Xs:
    :param track_ids:
    :return: masked bs and Xs
    """
    if data.pdr_shots_exist():
        p, stddev1 = update_pdr_prediction_position(shot_id, reconstruction, data)
        r, stddev2 = update_pdr_prediction_rotation(shot_id, reconstruction, data)

        if stddev1 <= 100.0 and stddev2 <= 1.0:
            min_inliers = data.config['resection_min_inliers']
            threshold = 2*data.config['resection_threshold']

            R = _euler_angles_to_rotation_matrix(r)
            reprojected_bs = R.dot((Xs - p).T).T
            reprojected_bs /= np.linalg.norm(reprojected_bs, axis=1)[:, np.newaxis]

            distances = np.linalg.norm(reprojected_bs - bs, axis=1)

            # relax threshold until 90% of features are through
            # TODO: put 90% in config
            for i in range(1, 100):
                mask = distances < i*threshold
                if int(sum(mask)) > max(len(mask)*0.9, min_inliers):
                    break

            if int(sum(mask)) < min_inliers:
                return bs, Xs, track_ids

            logger.debug("culling {}/{} features before resection".format(len(mask)-int(sum(mask)), len(mask)))

            masked_bs = [x for i, x in enumerate(bs) if mask[i]]
            masked_Xs = [x for i, x in enumerate(Xs) if mask[i]]
            masked_track_ids = [x for i, x in enumerate(track_ids) if mask[i]]

            return np.array(masked_bs), np.array(masked_Xs), masked_track_ids

    return bs, Xs, track_ids


def validate_position(reconstruction, data, shot):
    p_sfm = shot.pose.get_origin()
    p_pdr, stddev = update_pdr_prediction_position(shot.id, reconstruction, data)

    p_dist = np.linalg.norm(p_pdr[:2] - p_sfm[:2])
    if p_dist > stddev*2:
        logger.debug("validate_position: p_dist {}".format(p_dist))

    return p_dist < stddev*2


def validate_rotation(reconstruction, data, shot):
    r_sfm = _rotation_matrix_to_euler_angles(shot.pose.get_rotation_matrix())
    r_pdr, stddev = update_pdr_prediction_rotation(shot.id, reconstruction, data)

    q_0 = tf.quaternion_from_euler(r_pdr[0], r_pdr[1], r_pdr[2])
    q_1 = tf.quaternion_from_euler(r_sfm[0], r_sfm[1], r_sfm[2])

    r_dist = tf.quaternion_distance(q_0, q_1)

    if r_dist > stddev:
        logger.debug("validate_resection_pdr: r_dist{}".format(r_dist))

    return r_dist < stddev


def validate_resection_pdr(reconstruction, data, shot):
    is_pos_ok = is_rot_ok = True
    if data.pdr_shots_exist():
        is_pos_ok = validate_position(reconstruction, data, shot)
        is_rot_ok = validate_rotation(reconstruction, data, shot)

    return is_pos_ok, is_rot_ok


def debug_rotation_prior(reconstruction, data):
    if len(reconstruction.shots) < 3:
        return

    dists = []
    for shot_id in reconstruction.shots:
        rotation_prior, dop = update_pdr_prediction_rotation(shot_id, reconstruction, data)
        q_p = tf.quaternion_from_euler(rotation_prior[0], rotation_prior[1], rotation_prior[2])
        q_s = tf.quaternion_from_matrix(reconstruction.shots[shot_id].pose.get_rotation_matrix())

        #logger.debug("{}, rotation prior/sfm distance is {} degrees".format(shot_id, np.degrees(tf.quaternion_distance(q_p, q_s))))
        dists.append(np.degrees(tf.quaternion_distance(q_p, q_s)))

    logger.debug("avg prior/sfm diff {} degrees".format(np.mean(np.array(dists))))


def scale_reconstruction_to_pdr(reconstruction, data):
    """
    scale the reconstruction to pdr predictions
    """
    if not data.gps_points_exist():
        return

    if not data.pdr_shots_exist():
        return

    pdr_predictions_dict = data.load_pdr_predictions()
    
    if not pdr_predictions_dict:
        return

    ref_shots_dict = {}
    for shot in reconstruction.shots.values():
        if pdr_predictions_dict and shot.id in pdr_predictions_dict:
            ref_shots_dict[shot.id] = pdr_predictions_dict[shot.id][:3]

    two_ref_shots_dict = ref_shots_dict
    if len(ref_shots_dict) > 2:
        two_ref_shots_dict, distance = get_farthest_shots(ref_shots_dict)

    transform_reconstruction(reconstruction, two_ref_shots_dict)

    # setting scaled=true will prevent this routine from being called again. we will perform
    # this scaling operation a couple times at beginning of a reconstruction
    if len(reconstruction.shots) > 3:
        reconstruction.alignment.scaled = True


def align_reconstructions_to_pdr(reconstructions, data):
    """
    for any reconstruction that's not aligned with gps, use pdr predictions to align them
    """
    reconstructions[:] = [align_reconstruction_to_pdr(recon, data) for recon in reconstructions]

    # debugging
    export_coordinates(reconstructions)


def align_reconstruction_to_pdr(reconstruction, data):
    if not reconstruction.alignment.aligned:
        pdr_predictions_dict = init_pdr_predictions(data, True)

        if not pdr_predictions_dict:
            reconstruction = test_geo_hash(reconstruction, data)
        else:
            reconstruction = direct_align_pdr(data, reconstruction.shots.keys())

    return reconstruction


def align_reconstructions_to_pdr_old(reconstructions, data):
    """
    attempt to align un-anchored reconstructions
    """
    if not data.gps_points_exist():
        return

    if not data.pdr_shots_exist():
        return

    # get updated predictions
    aligned_sfm_points_dict = {}

    for reconstruction in reconstructions:
        if reconstruction.alignment.aligned:
            for shot in reconstruction.shots.values():
                aligned_sfm_points_dict[shot.id] = shot.pose.get_origin()

    gps_points_dict = data.load_topocentric_gps_points()
    pdr_shots_dict = data.load_pdr_shots()
    scale_factor = data.config['reconstruction_scale_factor']

    walkthrough_dict = pdr_walkthrough(aligned_sfm_points_dict, pdr_shots_dict)

    for reconstruction in reconstructions:
        if not reconstruction.alignment.aligned:
            reconstruction.alignment.aligned = \
                align_reconstruction_to_pdr_old(reconstruction, walkthrough_dict, gps_points_dict, scale_factor)


def align_reconstruction_to_pdr_old(reconstruction, walkthrough_dict, gps_points_dict, scale_factor):
    """
    align one partial reconstruction to 'best' pdr predictions
    """
    recon_predictions_dict = {}
    for shot_id in reconstruction.shots:
        recon_predictions_dict[shot_id] = walkthrough_dict[shot_id]

    # sort shots in the reconstruction by distance value
    sorted_by_distance_shot_ids = sorted(recon_predictions_dict, key=lambda k: recon_predictions_dict[k][1])

    ref_shots_dict = {}
    for i in range(len(sorted_by_distance_shot_ids)):
        shot_id = sorted_by_distance_shot_ids[i]

        if recon_predictions_dict[shot_id][1] < 2:
            ref_shots_dict[shot_id] = recon_predictions_dict[shot_id][0]
        elif shot_id in gps_points_dict:
            ref_shots_dict[shot_id] = gps_points_dict[shot_id]
            break

    if len(ref_shots_dict) >= 2:
        two_ref_shots_dict, distance = get_farthest_shots(ref_shots_dict)

        # only do this alignment if the reference shots are far enough away from each other,
        # or if we have only a few shots in this reconstruction. attempting to do this on
        # two nearby shots could lead to a lot of distortion
        if distance > 5/scale_factor or len(reconstruction.shots) < 10:
            transform_reconstruction(reconstruction, two_ref_shots_dict)

            # debug
            logger.debug("align_reconstructon_to_pdr: align to shots")
            for shot_id in two_ref_shots_dict:
                logger.debug("{}, position={}".format(shot_id, two_ref_shots_dict[shot_id]))

            return True

    # if do alignment based on pdr not doable, we will settle with updating the shots with
    # latest pdr predictions
    reposition_reconstruction(reconstruction, recon_predictions_dict)
    return False


def pdr_walk_forward(aligned_sfm_points_dict, pdr_shots_dict):
    """
    based on all aligned points, walk forward to predict all unaligned points
    :param aligned_sfm_points_dict:
    :param pdr_shots_dict:
    :return:
    """
    walkthrough_dict = {}
    for shot_id in pdr_shots_dict:
        walkthrough_dict[shot_id] = [0, 0, 0], 999999

    aligned_shot_ids = sorted(aligned_sfm_points_dict.keys())

    # find first consecutive frames
    idx = -1
    for i in range(len(aligned_shot_ids)-1):
        if _shot_id_to_int(aligned_shot_ids[i]) == _shot_id_to_int(aligned_shot_ids[i+1]) - 1:
            idx = i
            break

    if idx != -1:
        start_idx = _shot_id_to_int(aligned_shot_ids[idx])

        walkthrough_dict[_int_to_shot_id(start_idx)] = aligned_sfm_points_dict[_int_to_shot_id(start_idx)]
        walkthrough_dict[_int_to_shot_id(start_idx+1)] = aligned_sfm_points_dict[_int_to_shot_id(start_idx+1)]

        for i in range(start_idx, len(pdr_shots_dict)-2):
            dist_1_id = _int_to_shot_id(i+1)
            dist_2_id = _int_to_shot_id(i)
            pred_id = _int_to_shot_id(i+2)
            if pred_id not in aligned_sfm_points_dict:
                trust1 = trust2 = 0
                if dist_1_id in aligned_sfm_points_dict:
                    dist_1_coords = aligned_sfm_points_dict[dist_1_id]
                else:
                    dist_1_coords, trust1 = walkthrough_dict[dist_1_id]

                if dist_2_id in aligned_sfm_points_dict:
                    dist_2_coords = aligned_sfm_points_dict[dist_2_id]
                else:
                    dist_2_coords, trust2 = walkthrough_dict[dist_2_id]

                pdr_info_dist_1 = pdr_shots_dict[dist_1_id]
                pdr_info = pdr_shots_dict[pred_id]

                delta_heading = tf.delta_heading(np.radians(pdr_info_dist_1[3:6]), np.radians(pdr_info[3:6]))
                if pdr_info_dist_1[6] > 1e-1:
                    delta_distance = pdr_info[6] / pdr_info_dist_1[6] * np.linalg.norm(np.array(dist_1_coords)-np.array(dist_2_coords))
                else:
                    delta_distance = pdr_info_dist_1[6]

                trust = max(trust1, trust2) + 1
                walkthrough_dict[pred_id] = position_extrapolate(dist_1_coords, dist_2_coords, delta_heading, delta_distance), trust
            else:
                walkthrough_dict[pred_id] = aligned_sfm_points_dict[pred_id], 0

    return walkthrough_dict


def pdr_walk_backward(aligned_sfm_points_dict, pdr_shots_dict):
    """
    based on all aligned points, walk forward to predict all unaligned points
    :param aligned_sfm_points_dict:
    :param pdr_shots_dict:
    :return:
    """
    walkthrough_dict = {}
    for shot_id in pdr_shots_dict:
        walkthrough_dict[shot_id] = [0, 0, 0], 999999

    aligned_shot_ids = sorted(aligned_sfm_points_dict.keys(), reverse=True)

    # find first consecutive frames
    idx = -1
    for i in range(len(aligned_shot_ids)-1):
        if _shot_id_to_int(aligned_shot_ids[i]) == _shot_id_to_int(aligned_shot_ids[i+1]) + 1:
            idx = i
            break

    if idx != -1:
        start_idx = _shot_id_to_int(aligned_shot_ids[idx])

        for i in range(start_idx, 1, -1):
            dist_1_id = _int_to_shot_id(i-1)
            dist_2_id = _int_to_shot_id(i)
            pred_id = _int_to_shot_id(i-2)
            if pred_id not in aligned_sfm_points_dict:
                trust1 = trust2 = 0
                if dist_1_id in aligned_sfm_points_dict:
                    dist_1_coords = aligned_sfm_points_dict[dist_1_id]
                else:
                    dist_1_coords, trust1 = walkthrough_dict[dist_1_id]

                if dist_2_id in aligned_sfm_points_dict:
                    dist_2_coords = aligned_sfm_points_dict[dist_2_id]
                else:
                    dist_2_coords, trust2 = walkthrough_dict[dist_2_id]

                pdr_info_dist_1 = pdr_shots_dict[dist_1_id]
                pdr_info = pdr_shots_dict[pred_id]

                delta_heading = tf.delta_heading(np.radians(pdr_info_dist_1[3:6]), np.radians(pdr_info[3:6]))
                if pdr_info_dist_1[6] > 1e-1:
                    delta_distance = pdr_info[6] / pdr_info_dist_1[6] * np.linalg.norm(np.array(dist_1_coords)-np.array(dist_2_coords))
                else:
                    delta_distance = pdr_info_dist_1[6]

                trust = max(trust1, trust2) + 1
                walkthrough_dict[pred_id] = position_extrapolate(dist_1_coords, dist_2_coords, delta_heading, delta_distance), trust
            else:
                walkthrough_dict[pred_id] = aligned_sfm_points_dict[pred_id], 0

    return walkthrough_dict


def pdr_walkthrough(aligned_sfm_points_dict, pdr_shots_dict):
    f_dict = pdr_walk_forward(aligned_sfm_points_dict, pdr_shots_dict)
    b_dict = pdr_walk_backward(aligned_sfm_points_dict, pdr_shots_dict)

    final_dict = {}
    for shot_id in pdr_shots_dict:
        f_dist = f_dict[shot_id][1]
        b_dist = b_dict[shot_id][1]

        if f_dist < b_dist:
            final_dict[shot_id] = f_dict[shot_id]
        else:
            final_dict[shot_id] = b_dict[shot_id]

    return final_dict


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


def transform_reconstruction(reconstruction, ref_shots_dict):
    """
    transform recon based on two reference positions
    :param reconstruction:
    :param ref_shots_dict:
    :return:
    """
    X, Xp = [], []
    onplane, verticals = [], []
    for shot_id in ref_shots_dict:
        X.append(reconstruction.shots[shot_id].pose.get_origin())
        Xp.append(ref_shots_dict[shot_id])
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


def reposition_reconstruction(reconstruction, recon_predictions_dict):
    """
    reposition reconstruction to pdr positions
    :param reconstruction:
    :param recon_predictions_dict:
    :return:
    """
    for shot_id in reconstruction.shots:
        reconstruction.shots[shot_id].pose.set_origin(recon_predictions_dict[shot_id][0])


def update_pdr_global_2d(gps_points_dict, pdr_shots_dict, scale_factor, skip_bad=True):
    """
    *globally* align pdr predictions to GPS points

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

    return pdr_predictions_dict


def update_pdr_global(gps_points_dict, pdr_shots_dict, scale_factor, skip_bad=True, stride_len=3):
    """
    *globally* align pdr predictions to GPS points

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


def update_pdr_local(shot_id, sfm_points_dict, pdr_shots_dict, scale_factor):
    """
    *locally* align pdr predictions to SfM output. the SfM points have been aligned with
    GPS points

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
        if pdr_info_dist_1[6] > 1e-1:
            delta_distance = pdr_info[6] / pdr_info_dist_1[6] * np.linalg.norm(dist_1_coords-dist_2_coords)
        else:
            delta_distance = pdr_info_dist_1[6]

        # TODO: put 200 in config
        return position_extrapolate(dist_1_coords, dist_2_coords, delta_heading, delta_distance), 200
    else:
        # we cannot find 2 consecutive shots to extrapolate, so use 3 shots to estimate affine
        #sorted_shot_ids = get_closest_shots(shot_id, sfm_points_dict.keys())
        #return update_pdr_local_affine(shot_id, sfm_points_dict, pdr_shots_dict, scale_factor, sorted_shot_ids)
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


