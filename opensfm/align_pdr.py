"""affine transform pdr predictions to align with GPS points or SfM output."""

import logging
import math
import numpy as np

from opensfm import multiview
from opensfm import transformations as tf

logger = logging.getLogger(__name__)


def align_reconstruction_to_pdr(reconstruction, pdr_predictions_dict):
    """
    orient the two view reconstruction to pdr priors
    """
    X, Xp = [], []
    onplane, verticals = [], []
    for shot in reconstruction.shots.values():
        X.append(shot.pose.get_origin())
        Xp.append(pdr_predictions_dict[shot.id][:3])
        R = shot.pose.get_rotation_matrix()
        onplane.append(R[0,:])
        onplane.append(R[2,:])
        verticals.append(R[1,:])

    X = np.array(X)
    Xp = np.array(Xp)

    # Estimate ground plane.
    p = multiview.fit_plane(X - X.mean(axis=0), onplane, verticals)
    Rplane = multiview.plane_horizontalling_rotation(p)
    X = Rplane.dot(X.T).T

    # Estimate 2d similarity to align to GPS
    if (len(X) < 2 or
            X.std(axis=0).max() < 1e-8 or  # All points are the same.
            Xp.std(axis=0).max() < 0.01):  # All GPS points are the same.
        # Set the arbitrary scale proportional to the number of cameras.
        s = len(X) / max(1e-8, X.std(axis=0).max())
        A = Rplane
        b = Xp.mean(axis=0) - X.mean(axis=0)
    else:
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


def align_pdr_global_2d(gps_points_dict, pdr_shots_dict, reconstruction_scale_factor):
    """
    *globally* align pdr predictions to GPS points

    use 2 gps points at a time to align pdr predictions

    :param gps_points_dict: gps points in topocentric coordinates
    :param pdr_shots_dict: position of each shot as predicted by pdr
    :return: aligned pdr shot predictions
    """
    if len(gps_points_dict) < 2 or len(pdr_shots_dict) < 2:
        return {}

    # reconstruct_scale_factor is from oiq_config.yaml, and it's feet per pixel.
    # 0.3048 is meter per foot. 1.0 / (reconstruction_scale_factor * 0.3048) is
    # therefore pixels/meter, and since pdr output is in meters, it's the
    # expected scale
    expected_scale = 1.0 / (reconstruction_scale_factor * 0.3048)

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
        logger.info("rotation=%f, %f, %f", x, y, z)

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


def align_pdr_global(gps_points_dict, pdr_shots_dict, reconstruction_scale_factor, stride_len=3):
    """
    *globally* align pdr predictions to GPS points

    Move a sliding window through the gps points and get 3 neighboring points at a time;
    use them to piece-wise affine transform pdr predictions to align with GPS points

    :param gps_points_dict: gps points in topocentric coordinates
    :param pdr_shots_dict: position of each shot as predicted by pdr
    :param reconstruction_scale_factor: scale factor feet per pixel
    :return: aligned pdr shot predictions - [x, y, z, dop]
    """
    if len(gps_points_dict) < stride_len or len(pdr_shots_dict) < stride_len:
        logger.info("align_pdr_global: need more gps points. supplied only {}", len(gps_points_dict))
        return {}

    pdr_predictions_dict = {}

    # reconstruct_scale_factor is from oiq_config.yaml, and it's feet per pixel.
    # 0.3048 is meter per foot. 1.0 / (reconstruction_scale_factor * 0.3048) is
    # therefore pixels/meter, and since pdr output is in meters, it's the
    # expected scale
    expected_scale = 1.0 / (reconstruction_scale_factor * 0.3048)

    last_deviation = 1.0

    all_gps_shot_ids = sorted(gps_points_dict.keys())
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
        logger.info("deviation=%f, rotation=%f, %f, %f", deviation, x, y, z)

        # based on deviation, we choose different starting pdr shot to transform
        if deviation < last_deviation:
            pdr_start_shot_id = gps_shot_ids[0]
        else:
            pdr_start_shot_id = gps_shot_ids[1]

        pdr_end_shot_id = _int_to_shot_id(len(pdr_shots_dict)-1)

        if i == 0:
            # in first iteration, we transform pdr from first shot
            pdr_start_shot_id = _int_to_shot_id(0)

        new_dict = apply_affine_transform(pdr_shots_dict, pdr_start_shot_id, pdr_end_shot_id,
                                          s, A, b,
                                          deviation, gps_shot_ids)
        pdr_predictions_dict.update(new_dict)

        last_deviation = deviation

    return pdr_predictions_dict


def align_pdr_local(sfm_points_dict, pdr_shots_dict, reconstruction_scale_factor, num_history=3, num_predictions=3):
    """
    *locally* align pdr predictions to SfM output. the SfM points have been aligned with
    GPS points

    :param sfm_points_dict: sfm point coordinates
    :param pdr_shots_dict: original predictions
    :param reconstruction_scale_factor: scale factor feet per pixel
    :param num_history: number of sfm shots to use for the calculation
    :param num_predictions: number of shots beyond current last shot to align
    :return: updated pdr predictions for num_predictions shots
    """
    if len(sfm_points_dict) < num_history:
        return {}

    sfm_shot_ids = sorted(sfm_points_dict.keys())

    sfm_coords = []
    for shot_id in sfm_shot_ids[-num_history:]:
        sfm_coords.append(sfm_points_dict[shot_id])

    sfm_directions = []
    for i in range(1, len(sfm_coords)):
        direction = np.arctan2(sfm_coords[i][1] - sfm_coords[i-1][1], sfm_coords[i][0] - sfm_coords[i-1][0])
        sfm_directions.append(direction)

    direction_stddev = np.std(np.array(sfm_directions))

    if np.degrees(direction_stddev) > 20.0:
        # if directions differ by a lot, we use affine alignment
        updates = align_pdr_local_affine(sfm_points_dict, pdr_shots_dict, reconstruction_scale_factor,
                                         num_history, num_predictions)
    else:
        # if we are walking on roughly a straight line, then just extrapolate
        ref_coord = sfm_coords[-1]
        ref_dir = np.mean(np.array(sfm_directions))

        start_idx = min(len(pdr_shots_dict)-1, _shot_id_to_int(sfm_shot_ids[-1]) + 1)
        end_idx = min(len(pdr_shots_dict)-1, start_idx + num_predictions)

        delta_heading_distance_dict = {}
        for i in range(start_idx, end_idx):
            shot_id = _int_to_shot_id(i)
            pdr_info = pdr_shots_dict[shot_id]
            delta_heading_distance_dict[shot_id] = (pdr_info[3], pdr_info[4])

        updates = align_pdr_local_extrapolate(ref_coord, ref_dir, delta_heading_distance_dict)

    return updates


def align_pdr_local_extrapolate(ref_coord, ref_dir, delta_heading_distance_dict):
    """
    update pdr predictions based on extrapolating last SfM position and direction

    :param ref_coord: last sfm point's coordinate
    :param ref_dir: lst sfm point's direction
    :param delta_heading_distance_dict: delta heading and distance for the next n
           shots based on pdr calculation
    :return: updated pdr predictions for next len(delta_heading_distance_dict) shots
    """
    logger.info("align_pdr_local_extrapolate")
    updates = {}

    last_dir = ref_dir
    last_coord = ref_coord

    for shot_id in sorted(delta_heading_distance_dict.keys()):
        curr_dir = last_dir + delta_heading_distance_dict[shot_id][0]
        x = last_coord[0] + delta_heading_distance_dict[shot_id][1]*np.cos(curr_dir)
        y = last_coord[1] + delta_heading_distance_dict[shot_id][1]*np.sin(curr_dir)
        z = last_coord[2]
        curr_coord = [x, y, z]
        updates[shot_id] = [x, y, z, 100]

        last_dir = curr_dir
        last_coord = curr_coord

    return updates


def align_pdr_local_affine(sfm_points_dict, pdr_shots_dict, reconstruction_scale_factor, num_history, num_predictions):
    """
    estimating the affine transform between a set of SfM point coordinates and a set of
    original pdr predictions. Then affine transform the pdr predictions

    :param sfm_points_dict: sfm point coordinates
    :param pdr_shots_dict: original predictions
    :param reconstruction_scale_factor: scale factor feet per pixel
    :param num_predictions: number of shots to extrapolate prediction
    :return: updated pdr predictions for num_to_predict shots
    """
    logger.info("align_pdr_local_affine")
    # reconstruct_scale_factor is from oiq_config.yaml, and it's feet per pixel.
    # 0.3048 is meter per foot. 1.0 / (reconstruction_scale_factor * 0.3048) is
    # therefore pixels/meter, and since pdr output is in meters, it's the
    # expected scale
    expected_scale = 1.0 / (reconstruction_scale_factor * 0.3048)

    sfm_coords = []
    pdr_coords = []

    sfm_shot_ids = sorted(sfm_points_dict.keys())
    for shot_id in sfm_shot_ids[-num_history:]:
        sfm_coords.append(sfm_points_dict[shot_id])
        pdr_coords.append(pdr_shots_dict[shot_id][0:3])

    s, A, b = get_affine_transform(sfm_coords, pdr_coords)

    # the closer s is to expected_scale, the better the fit, and the less the deviation
    ratio = s/expected_scale
    if ratio > 1.0:
        ratio = 1/ratio

    # if deviation is very large, skip it
    deviation = math.fabs(1.0 - ratio)
    if deviation > 0.5:
        return {}

    # if x/y rotation is not close to 0, then likely it's 'flipped' and no good
    [x, y, z] = _rotation_matrix_to_euler_angles(A)
    if math.fabs(x) > 1.0 or math.fabs(y) > 1.0:
        return {}

    range_start_idx = min(len(pdr_shots_dict)-1, _shot_id_to_int(sfm_shot_ids[-1]) + 1)
    range_end_idx = min(len(pdr_shots_dict)-1, range_start_idx + num_predictions)

    updates = apply_affine_transform(pdr_shots_dict,
                                     _int_to_shot_id(range_start_idx), _int_to_shot_id(range_end_idx),
                                     s, A, b,
                                     deviation)

    #logger.info("align_pdr_local: range_start_idx={}, range_end_idx={}, s={}".format(range_start_idx, range_end_idx, s))
    return updates


def apply_affine_transform(pdr_shots_dict, start_shot_id, end_shot_id, s, A, b, deviation, gps_shot_ids=[]):
    """Apply a similarity (y = s A x + b) to a reconstruction.

    :param pdr_shots_dict: all pdr predictions
    :param start_shot_id: start shot id to perform transform
    :param end_shot_id: end shot id to perform transform
    :param s: The scale (a scalar)
    :param A: The rotation matrix (3x3)
    :param b: The translation vector (3)
    :param gps_shot_ids: gps shot ids the affine transform is based on
    :param deviation: a measure of how closely pdr predictions match gps points
    :return: pdr shots between start and end shot id transformed
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

