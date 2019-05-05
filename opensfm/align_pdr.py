"""affine transform pdr predictions to align with GPS points or SfM output."""

import logging
import math
import numpy as np

from opensfm import multiview
from opensfm import transformations as tf

logger = logging.getLogger(__name__)


def align_pdr_global_2d(gps_points_dict, pdr_shots_dict, reconstruction_scale_factor):
    """
    *globally* align pdr predictions to GPS points

    Move a sliding window through the gps points and get 2 neighboring points at a time;
    use them to piece-wise affine transform pdr predictions to align with GPS points

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

    aligned_pdr_shots_dict = {}

    gps_shot_ids = sorted(gps_points_dict.keys())
    for i in range(len(gps_shot_ids) - 1):
        gps_coords = []
        pdr_coords = []

        for j in range(2):
            shot_id = gps_shot_ids[i+j]
            gps_coords.append(gps_points_dict[shot_id])
            pdr_coords.append([pdr_shots_dict[shot_id][0], -pdr_shots_dict[shot_id][1], 0])

        s, A, b = get_affine_transform_2d(gps_coords, pdr_coords)

        [x, y, z] = _rotation_matrix_to_euler_angles(A)
        logger.info("rotation=%f, %f, %f", x, y, z)

        if not ((0.50 * expected_scale) < s < (2.0 * expected_scale)):
            logger.info("s/expected_scale={}, discard".format(s/expected_scale))
            continue

        start_shot_id = gps_shot_ids[i]
        end_shot_id = gps_shot_ids[i+1]

        # in first iteration, we transform pdr from first shot
        # in last iteration, we transform pdr until last shot
        if i == 0:
            start_shot_id = _int_to_shot_id(0)
        elif i == len(gps_points_dict)-2:
            end_shot_id = _int_to_shot_id(len(pdr_shots_dict)-1)

        new_dict = apply_affine_transform(pdr_shots_dict, start_shot_id, end_shot_id, s, A, b)
        aligned_pdr_shots_dict.update(new_dict)

    return aligned_pdr_shots_dict


def align_pdr_global(gps_points_dict, pdr_shots_dict, reconstruction_scale_factor):
    """
    *globally* align pdr predictions to GPS points

    Move a sliding window through the gps points and get 3 neighboring points at a time;
    use them to piece-wise affine transform pdr predictions to align with GPS points

    :param gps_points_dict: gps points in topocentric coordinates
    :param pdr_shots_dict: position of each shot as predicted by pdr
    :return: aligned pdr shot predictions
    """
    if len(gps_points_dict) < 3 or len(pdr_shots_dict) < 3:
        return {}

    aligned_pdr_shots_dict = {}

    # reconstruct_scale_factor is from oiq_config.yaml, and it's feet per pixel.
    # 0.3048 is meter per foot. 1.0 / (reconstruction_scale_factor * 0.3048) is
    # therefore pixels/meter, and since pdr output is in meters, it's the
    # expected scale
    expected_scale = 1.0 / (reconstruction_scale_factor * 0.3048)

    last_deviation = 1.0

    gps_shot_ids = sorted(gps_points_dict.keys())
    for i in range(len(gps_shot_ids) - 2):
        gps_coords = []
        pdr_coords = []

        for j in range(3):
            shot_id = gps_shot_ids[i+j]
            gps_coords.append(gps_points_dict[shot_id])
            pdr_coords.append(pdr_shots_dict[shot_id][0:3])

        s, A, b = get_affine_transform(gps_coords, pdr_coords)

        # the closer s is to expected_scale, the better the fit, and the less the deviation
        deviation = math.fabs(1.0 - s/expected_scale)

        [x, y, z] = _rotation_matrix_to_euler_angles(A)
        logger.info("deviation=%f, rotation=%f, %f, %f", deviation, x, y, z)

        # in last iteration, we transform pdr until last shot
        if i == 0:
            # in first iteration, we transform pdr from first shot
            start_shot_id = _int_to_shot_id(0)
            end_shot_id = gps_shot_ids[2]
        else:
            if deviation < last_deviation:
                start_shot_id = gps_shot_ids[i]
            else:
                start_shot_id = gps_shot_ids[i+1]

            if i == len(gps_points_dict)-3:
                end_shot_id = _int_to_shot_id(len(pdr_shots_dict)-1)
            else:
                end_shot_id = gps_shot_ids[i+2]

        new_dict = apply_affine_transform(pdr_shots_dict, start_shot_id, end_shot_id, s, A, b)
        aligned_pdr_shots_dict.update(new_dict)

        last_deviation = deviation

    return aligned_pdr_shots_dict


def align_pdr_local(sfm_points_dict, pdr_shots_dict, start_shot_id, end_shot_id):
    """
    *locally* align pdr predictions to SfM output. note the SfM may or may not have been
    aligned with GPS points

    Estimating the affine transform between a set of SfM point coordinates and a set of
    pdr shot predictions. Then affine transform the pdr predictions from start_shot_id
    to end_shot_id

    :param sfm_points_dict: sfm point coordinates
    :param pdr_shots_dict: position of each shot as predicted by pdr
    :param start_shot_id: start shot id
    :param end_shot_id: end shot id
    :return: aligned pdr shot predictions for shots between start/end_shot_id
    """
    sfm_coords = []
    pdr_coords = []

    sfm_shot_ids = sorted(sfm_points_dict.keys())
    for i in range(len(sfm_shot_ids)):
        shot_id = sfm_shot_ids[i]

        if shot_id in pdr_shots_dict:
            sfm_coords.append(sfm_points_dict[shot_id])
            pdr_coords.append(pdr_shots_dict[shot_id][0:3])

    if len(sfm_coords) < 3:
        return {}

    s, A, b = get_affine_transform(sfm_coords, pdr_coords)

    #[x, y, z] = _rotation_matrix_to_euler_angles(A)
    #if (0.80 < s < 1.2) and (math.fabs(z) < 20):
        #aligned_pdr_shots_dict = apply_affine_transform(pdr_shots_dict, start_shot_id, end_shot_id, s, A, b)
        #return aligned_pdr_shots_dict
    #else:
        #logger.info("discard s=%f, z=%f", s, z)
        #return {}

    aligned_pdr_shots_dict = apply_affine_transform(pdr_shots_dict, start_shot_id, end_shot_id, s, A, b)
    return aligned_pdr_shots_dict


def apply_affine_transform(pdr_shots_dict, start_shot_id, end_shot_id, s, A, b):
    """Apply a similarity (y = s A x + b) to a reconstruction.

    :param pdr_shots_dict: all pdr predictions
    :param start_shot_id: start shot id to perform transform
    :param end_shot_id: end shot id to perform transform
    :param s: The scale (a scalar)
    :param A: The rotation matrix (3x3)
    :param b: The translation vector (3)
    :return: pdr shots between start and end shot id transformed
    """
    new_dict = {}

    start_index = _shot_id_to_int(start_shot_id)
    end_index = _shot_id_to_int(end_shot_id)

    # transform pdr shots
    for i in range(start_index, end_index + 1):
        shot_id = _int_to_shot_id(i)

        if shot_id in pdr_shots_dict:
            Xp = s * A.dot(pdr_shots_dict[shot_id][0:3]) + b
            new_dict[shot_id] = Xp.tolist()

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

