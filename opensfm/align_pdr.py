"""affine transform pdr predictions to align with GPS points or SfM output."""

import operator
import logging
import math
import numpy as np

from itertools import combinations

from opensfm import geo
from opensfm import multiview
from opensfm import transformations as tf

from opensfm.debug_plot import debug_plot_pdr

logger = logging.getLogger(__name__)


def init_pdr_predictions(data):
    """
    globally align pdr path to gps points

    :param data:
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

    pdr_predictions_dict = update_pdr_global(topocentric_gps_points_dict, pdr_shots_dict, scale_factor)

    data.save_topocentric_gps_points(topocentric_gps_points_dict)
    data.save_pdr_predictions(pdr_predictions_dict)

    # debug
    debug_plot_pdr(topocentric_gps_points_dict, pdr_predictions_dict)

    return pdr_predictions_dict


def update_pdr_prediction_position(shot_id, reconstruction, data):
    if data.pdr_shots_exist():
        if reconstruction.alignment.aligned:
            # get updated predictions
            sfm_points_dict = {}

            for shot in reconstruction.shots.values():
                sfm_points_dict[shot.id] = shot.pose.get_origin()

            pdr_shots_dict = data.load_pdr_shots()
            scale_factor = data.config['reconstruction_scale_factor']

            return update_pdr_local(shot_id, sfm_points_dict, pdr_shots_dict, scale_factor)
        else:
            pdr_predictions_dict = data.load_pdr_predictions()
            return pdr_predictions_dict[shot_id][:3], pdr_predictions_dict[shot_id][3]

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

        distances_dict = {}
        for a_id in reconstruction.shots:
            if a_id != shot_id:
                distances_dict[a_id] = _shot_id_to_int(shot_id) - _shot_id_to_int(a_id)

        # sort shot ids by absolute difference in sequence number
        sorted_shot_ids = sorted(distances_dict, key=lambda k: abs(distances_dict[k]))

        base_shot_id_0 = sorted_shot_ids[0]
        base_shot_id_1 = sorted_shot_ids[1]

        prediction_0 = get_rotation_prediction(shot_id, base_shot_id_0, reconstruction, data)
        prediction_1 = get_rotation_prediction(shot_id, base_shot_id_1, reconstruction, data)

        # 0.1 radians is roughly 6 degrees
        # TODO: put 0.1 in config
        tolerance = 0.1

        #if not np.allclose(prediction_0, prediction_1, rtol=1, atol=tolerance):
            #logger.debug("predict rotation for {} based on {}={}, {}={}, differ too much".format(shot_id, base_shot_id_0, prediction_0, base_shot_id_1, prediction_1))
            #return [0, 0, 0], 999999.0

        q_0 = tf.quaternion_from_euler(prediction_0[0], prediction_0[1], prediction_0[2])
        q_1 = tf.quaternion_from_euler(prediction_1[0], prediction_1[1], prediction_1[2])

        if tf.quaternion_distance(q_0, q_1) > tolerance:
            logger.debug("{}, rotation prior 0/1 distance is {} degrees".format(shot_id, np.degrees(tf.quaternion_distance(q_0, q_1))))
            return [0, 0, 0], 999999.0

        distance_to_base = distances_dict[base_shot_id_0]
        return prediction_0, tolerance*distance_to_base

    return [0, 0, 0], 999999.0


def get_rotation_prediction(shot_id, base_shot_id, reconstruction, data):
    pdr_shots_dict = data.load_pdr_shots()

    base_pdr_rotation = pdr_shots_dict[base_shot_id][3:6]
    pdr_rotation = pdr_shots_dict[shot_id][3:6]
    base_sfm_rotation = reconstruction.shots[base_shot_id].pose.get_rotation_matrix()

    pred_sfm_rotation = rotation_extrapolate(base_pdr_rotation, pdr_rotation, base_sfm_rotation)
    return pred_sfm_rotation


def debug_rotation_prior(reconstruction, data):
    for shot_id in reconstruction.shots:
        rotation_prior, dop = update_pdr_prediction_rotation(shot_id, reconstruction, data)

        rotation_sfm = tf.euler_from_quaternion(tf.quaternion_from_matrix(reconstruction.shots[shot_id].pose.get_rotation_matrix()))

        #deviation_x = (np.degrees(rotation_sfm[0] - rotation_prior[0]) + 360) % 360
        #deviation_y = (np.degrees(rotation_sfm[1] - rotation_prior[1]) + 360) % 360
        #deviation_z = (np.degrees(rotation_sfm[2] - rotation_prior[2]) + 360) % 360

        #if deviation_x > 180:
            #deviation_x -= 360
        #if deviation_y > 180:
            #deviation_y -= 360
        #if deviation_z > 180:
            #deviation_z -= 360

        #logger.debug("{}, rotation prior deviation {}, {}, {}".format(shot_id, deviation_x, deviation_y, deviation_z))

        q_p = tf.quaternion_from_euler(rotation_prior[0], rotation_prior[1], rotation_prior[2])
        q_s = tf.quaternion_from_euler(rotation_sfm[0], rotation_sfm[1], rotation_sfm[2])
        logger.debug("{}, rotation prior/sfm distance is {} degrees".format(shot_id, np.degrees(tf.quaternion_distance(q_p, q_s))))


def scale_reconstruction_to_pdr(reconstruction, data):
    """
    scale the reconstruction to pdr predictions
    """
    if not data.pdr_shots_exist():
        return

    reconstruction.alignment.scaled = True
    pdr_predictions_dict = data.load_pdr_predictions()

    ref_shots_dict = {}
    for shot in reconstruction.shots.values():
        if pdr_predictions_dict and shot.id in pdr_predictions_dict:
            ref_shots_dict[shot.id] = pdr_predictions_dict[shot.id][:3]

    two_ref_shots_dict = ref_shots_dict
    if len(ref_shots_dict) > 2:
        two_ref_shots_dict = get_farthest_shots(ref_shots_dict)

    transform_reconstruction(reconstruction, two_ref_shots_dict)


def align_reconstructions_to_pdr(reconstructions, data):
    """
    align all un-anchored reconstructions
    """
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

    pdr_predictions_dict = pdr_walkthrough(aligned_sfm_points_dict, gps_points_dict, pdr_shots_dict, scale_factor)

    for reconstruction in reconstructions:
        if not reconstruction.alignment.aligned:
            reconstruction.alignment.aligned = \
                align_reconstruction_to_pdr(reconstruction, pdr_predictions_dict)

            #reconstruction.alignment.aligned = \
                #align_reconstruction_to_pdr_old(reconstruction,
                                            #aligned_sfm_points_dict, gps_points_dict, pdr_shots_dict, scale_factor)


def align_reconstruction_to_pdr(reconstruction, pdr_predictions_dict):
    """
    align one partial reconstruction to 'best' pdr predictions
    """
    recon_predictions_dict = {}
    for shot_id in reconstruction.shots:
        recon_predictions_dict[shot_id] = pdr_predictions_dict[shot_id]

    # sort shots in the reconstruction by distance value
    sorted_by_distance_shot_ids = sorted(recon_predictions_dict, key=lambda k: recon_predictions_dict[k][1])

    ref_shots_dict = {}
    for i in range(len(sorted_by_distance_shot_ids)):
        shot_id = sorted_by_distance_shot_ids[i]

        if recon_predictions_dict[shot_id][1] < 2:
            ref_shots_dict[shot_id] = recon_predictions_dict[shot_id][0]
        else:
            break

    if len(ref_shots_dict) >= 2:
        transform_reconstruction(reconstruction, ref_shots_dict)

        # debug
        #logger.debug("align_reconstructon_to_pdr: align to shots")
        #for shot_id in ref_shots_dict:
            #logger.debug("{}, position={}".format(shot_id, ref_shots_dict[shot_id]))

        return True
    else:
        reposition_reconstruction(reconstruction, recon_predictions_dict)
        return False


def align_reconstruction_to_pdr_unused(reconstruction,
                                    aligned_sfm_points_dict, gps_points_dict, pdr_shots_dict, scale_factor):
    """
    align one partial reconstruction to 'best' pdr predictions (unused, kept code)
    """
    aligned_shot_ids = aligned_sfm_points_dict.keys()

    distances_dict = {}
    for shot in reconstruction.shots.values():
        distances_dict[shot.id] = min_distance_to_aligned_shots(shot.id, aligned_shot_ids, gps_points_dict)

    # sort by absolute value
    sorted_distances_keys = sorted(distances_dict, key=lambda k: abs(distances_dict[k]))

    ref_shots_dict = {}
    for i in range(len(sorted_distances_keys)):
        shot_id = sorted_distances_keys[i]
        if shot_id in gps_points_dict:
            ref_shots_dict[shot_id] = gps_points_dict[shot_id]
        else:
            p, stddev = update_pdr_local(shot_id, aligned_sfm_points_dict, pdr_shots_dict, scale_factor)
            if stddev != 999999:
                ref_shots_dict[shot_id] = p

        if len(ref_shots_dict) == 2:
            transform_reconstruction(reconstruction, ref_shots_dict)
            return True

    return False


def pdr_walk_forward(aligned_sfm_points_dict, gps_points_dict, pdr_shots_dict, scale_factor):
    """
    based on all aligned points, walk forward to predict all unaligned points
    :param aligned_sfm_points_dict:
    :param gps_points_dict:
    :param pdr_shots_dict:
    :param scale_factor:
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
                if pred_id not in gps_points_dict:
                    trust1 = trust2 = 0
                    if dist_1_id in aligned_sfm_points_dict:
                        dist_1_coords = aligned_sfm_points_dict[dist_1_id]
                    else:
                        dist_1_coords, trust1 = walkthrough_dict[dist_1_id]

                    if dist_2_id in aligned_sfm_points_dict:
                        dist_2_coords = aligned_sfm_points_dict[dist_2_id]
                    else:
                        dist_2_coords, trust2 = walkthrough_dict[dist_2_id]

                    pdr_info_dist1 = pdr_shots_dict[dist_1_id]
                    pdr_info = pdr_shots_dict[pred_id]

                    delta_heading = tf.delta_heading(np.radians(pdr_info_dist1[3:6]), np.radians(pdr_info[3:6]))
                    delta_distance = pdr_info[6] / (scale_factor * 0.3048)

                    trust = max(trust1, trust2) + 1
                    walkthrough_dict[pred_id] = position_extrapolate(dist_1_coords, dist_2_coords, delta_heading, delta_distance), trust
                else:
                    walkthrough_dict[pred_id] = gps_points_dict[pred_id], 0
            else:
                walkthrough_dict[pred_id] = aligned_sfm_points_dict[pred_id], 0

    return walkthrough_dict


def pdr_walk_backward(aligned_sfm_points_dict, gps_points_dict, pdr_shots_dict, scale_factor):
    """
    based on all aligned points, walk forward to predict all unaligned points
    :param aligned_sfm_points_dict:
    :param gps_points_dict:
    :param pdr_shots_dict:
    :param scale_factor:
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
                if pred_id not in gps_points_dict:
                    trust1 = trust2 = 0
                    if dist_1_id in aligned_sfm_points_dict:
                        dist_1_coords = aligned_sfm_points_dict[dist_1_id]
                    else:
                        dist_1_coords, trust1 = walkthrough_dict[dist_1_id]

                    if dist_2_id in aligned_sfm_points_dict:
                        dist_2_coords = aligned_sfm_points_dict[dist_2_id]
                    else:
                        dist_2_coords, trust2 = walkthrough_dict[dist_2_id]

                    pdr_info_dist1 = pdr_shots_dict[dist_1_id]
                    pdr_info = pdr_shots_dict[pred_id]

                    delta_heading = tf.delta_heading(np.radians(pdr_info_dist1[3:6]), np.radians(pdr_info[3:6]))
                    delta_distance = pdr_info[6] / (scale_factor * 0.3048)


                    trust = max(trust1, trust2) + 1
                    walkthrough_dict[pred_id] = position_extrapolate(dist_1_coords, dist_2_coords, delta_heading, delta_distance), trust
                else:
                    walkthrough_dict[pred_id] = gps_points_dict[pred_id], 0
            else:
                walkthrough_dict[pred_id] = aligned_sfm_points_dict[pred_id], 0

    return walkthrough_dict


def pdr_walkthrough(aligned_sfm_points_dict, gps_points_dict, pdr_shots_dict, scale_factor):
    f_dict = pdr_walk_forward(aligned_sfm_points_dict, gps_points_dict, pdr_shots_dict, scale_factor)
    b_dict = pdr_walk_backward(aligned_sfm_points_dict, gps_points_dict, pdr_shots_dict, scale_factor)

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


def rotation_extrapolate(base_pdr_rotation, pdr_rotation, base_sfm_rotation):
    """
    based on pdr rotations of base shot and current shot, calculate a delta rotation,
    then apply this delta rotation to sfm rotation of base to obtain a prediction/prior
    for sfm rotation of current shot
    :param base_pdr_rotation:
    :param base_sfm_rotation:
    :param pdr_rotation:
    :return: prediction for sfm rotation of current shot
    """

    qdiff = tf.quaternion_diff(np.radians(base_pdr_rotation), np.radians(pdr_rotation))
    qr = tf.quaternion_from_matrix(base_sfm_rotation.T)

    qnew = tf.quaternion_inverse(tf.quaternion_multiply(qdiff, qr))

    return tf.euler_from_quaternion(qnew)


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


def update_pdr_global_2d(gps_points_dict, pdr_shots_dict, scale_factor):
    """
    *globally* align pdr predictions to GPS points (not used, kept code)

    use 2 gps points at a time to align pdr predictions

    :param gps_points_dict: gps points in topocentric coordinates
    :param pdr_shots_dict: position of each shot as predicted by pdr
    :param scale_factor: reconstruction_scale_factor
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
            pdr_coords.append([pdr_shots_dict[shot_id][0], -pdr_shots_dict[shot_id][1], 0])

        s, A, b = get_affine_transform_2d(gps_coords, pdr_coords)

        # the closer s is to expected_scale, the better the fit, and the less the deviation
        deviation = math.fabs(1.0 - s/expected_scale)

        # debugging
        [x, y, z] = _rotation_matrix_to_euler_angles(A)
        logger.info("rotation=%f, %f, %f", np.degrees(x), np.degrees(y), np.degrees(z))

        if not ((0.50 * expected_scale) < s < (2.0 * expected_scale)):
            logger.info("s/expected_scale={}, discard".format(s/expected_scale))
            continue

        start_shot_id = all_gps_shot_ids[i]
        end_shot_id = all_gps_shot_ids[i+1]

        # in first iteration, we transform pdr from first shot
        # in last iteration, we transform pdr until last shot
        if i == 0:
            start_shot_id = _int_to_shot_id(0)
        elif i == len(gps_points_dict)-2:
            end_shot_id = _int_to_shot_id(len(pdr_shots_dict)-1)

        new_dict = apply_affine_transform(pdr_shots_dict, start_shot_id, end_shot_id,
                                          s, A, b,
                                          deviation, [all_gps_shot_ids[i], all_gps_shot_ids[i+1]])
        pdr_predictions_dict.update(new_dict)

    return pdr_predictions_dict


def update_pdr_global(gps_points_dict, pdr_shots_dict, scale_factor, stride_len=3):
    """
    *globally* align pdr predictions to GPS points

    Move a sliding window through the gps points and get 3 neighboring points at a time;
    use them to piece-wise affine transform pdr predictions to align with GPS points

    :param gps_points_dict: gps points in topocentric coordinates
    :param pdr_shots_dict: position of each shot as predicted by pdr
    :param scale_factor: reconstruction_scale_factor - scale factor feet per pixel
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
        if deviation > 0.5:
            last_deviation = 1.0
            continue

        # if x/y rotation is not close to 0, then likely it's 'flipped' and no good
        [x, y, z] = _rotation_matrix_to_euler_angles(A)
        if math.fabs(x) > 1.0 or math.fabs(y) > 1.0:
            last_deviation = 1.0
            continue

        # debugging
        #logger.info("deviation=%f, rotation=%f, %f, %f", deviation, x, y, z)

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
    # get sorted shot ids that are closest to shot_id (in sequence number)
    sorted_sfm_shot_ids = get_closest_shots(shot_id, sfm_points_dict.keys())
    #logger.info("shot {} closest two shots in sfm = {} {}".format(shot_id, sorted_sfm_shot_ids[0], sorted_sfm_shot_ids[1]))

    prev1 = _prev_shot_id(shot_id)
    prev2 = _prev_shot_id(prev1)
    next1 = _next_shot_id(shot_id)
    next2 = _next_shot_id(next1)

    if prev1 == sorted_sfm_shot_ids[0] and prev2 == sorted_sfm_shot_ids[1]:
        dist_1_coords = sfm_points_dict[prev1]
        dist_2_coords = sfm_points_dict[prev2]

        pdr_info_dist1 = pdr_shots_dict[prev1]
        pdr_info = pdr_shots_dict[shot_id]

        delta_heading = tf.delta_heading(np.radians(pdr_info_dist1[3:6]), np.radians(pdr_info[3:6]))
        delta_distance = pdr_info[6] / (scale_factor * 0.3048)
        # TODO: put 100 in config
        return position_extrapolate(dist_1_coords, dist_2_coords, delta_heading, delta_distance), 100
    elif next1 == sorted_sfm_shot_ids[0] and next2 == sorted_sfm_shot_ids[1]:
        dist_1_coords = sfm_points_dict[next1]
        dist_2_coords = sfm_points_dict[next2]

        pdr_info_dist1 = pdr_shots_dict[next1]
        pdr_info = pdr_shots_dict[shot_id]

        delta_heading = tf.delta_heading(np.radians(pdr_info_dist1[3:6]), np.radians(pdr_info[3:6]))
        delta_distance = pdr_info[6] / (scale_factor * 0.3048)
        # TODO: put 50 in config
        return position_extrapolate(dist_1_coords, dist_2_coords, delta_heading, delta_distance), 100
    else:
        # we cannot find 2 consecutive shots to extrapolate, so use 3 shots to estimate affine
        return update_pdr_local_affine(shot_id, sfm_points_dict, pdr_shots_dict, scale_factor, sorted_sfm_shot_ids)


def update_pdr_local_affine(shot_id, sfm_points_dict, pdr_shots_dict, scale_factor, sorted_sfm_shot_ids):
    """
    estimating the affine transform between a set of SfM point coordinates and a set of
    original pdr predictions. Then affine transform the pdr predictions

    :param shot_id: the shot to update
    :param sfm_points_dict: sfm point coordinates
    :param pdr_shots_dict: original predictions
    :param scale_factor: reconstruction_scale_factor - scale factor feet per pixel
    :param sorted_sfm_shot_ids: sfm_shot_ids sorted by their closeness to shot_id
    :return: updated pdr predictions for shot_id
    """
    if len(sorted_sfm_shot_ids) < 3:
        return [0, 0, 0], 999999

    # reconstruction_scale_factor is from oiq_config.yaml, and it's feet per pixel.
    # 0.3048 is meter per foot. 1.0 / (reconstruction_scale_factor * 0.3048) is
    # therefore pixels/meter, and since pdr output is in meters, it's the
    # expected scale
    expected_scale = 1.0 / (scale_factor * 0.3048)

    sfm_coords = []
    pdr_coords = []

    for i in range(3):
        a_id = sorted_sfm_shot_ids[i]
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

    update = apply_affine_transform(pdr_shots_dict, shot_id, shot_id,
                                     s, A, b,
                                     deviation)

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

        # TODO: read default dop 100 from config
        dop = 100 + min(distances)*10*(1+deviation)
    else:
        dop = 100 * (1+deviation)

    return dop


def get_farthest_shots(pdr_shots):
    """get two shots that are most far apart in physical distance"""
    distances_dict = {}
    pdr_shot_ids = pdr_shots.keys()
    for (i, j) in combinations(pdr_shot_ids, 2):
        distances_dict[(i, j)] = np.linalg.norm(np.array(pdr_shots[i]) - np.array(pdr_shots[j]))

    (shot1, shot2) = max(distances_dict.items(), key=operator.itemgetter(1))[0]

    return {shot1: pdr_shots[shot1], shot2: pdr_shots[shot2]}


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


