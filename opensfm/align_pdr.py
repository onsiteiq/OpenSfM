"""affine transform pdr predictions to align with GPS points or SfM output."""

import logging
import math
import numpy as np

from opensfm import transformations as tf

logger = logging.getLogger(__name__)


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

    # reconstruct_scale_factor is from oiq_config.yaml, and it's feet per pixel
    # 0.3048 is meter per foot. pdr output is in meters. so based on the two
    # we can calculate the approximate scale
    expected_scale = 1.0 / (reconstruction_scale_factor * 0.3048)

    aligned_pdr_shots_dict = {}

    gps_shot_ids = sorted(gps_points_dict.keys())
    for i in range(len(gps_shot_ids) - 2):
        gps_coords = []
        pdr_coords = []

        for j in range(3):
            shot_id = gps_shot_ids[i+j]
            gps_coords.append(gps_points_dict[shot_id])
            pdr_coords.append(pdr_shots_dict[shot_id][0:3])

        s, A, b = get_affine_transform(gps_coords, pdr_coords)

        #[x, y, z] = _rotation_matrix_to_euler_angles(A)
        #logger.info("rotation=%f, %f, %f", x, y, z)

        if not ((0.80 * expected_scale) < s < (1.2 * expected_scale)):
            logger.info("s/expected_scale={}, discard".format(s/expected_scale))
            continue

        start_shot_id = gps_shot_ids[i]
        end_shot_id = gps_shot_ids[i+2]

        # in first iteration, we transform pdr from first shot
        # in last iteration, we transform pdr until last shot
        if i == 0:
            start_shot_id = _int_to_shot_id(0)
        elif i == len(gps_points_dict)-3:
            end_shot_id = _int_to_shot_id(len(pdr_shots_dict)-1)

        new_dict = apply_affine_transform(pdr_shots_dict, start_shot_id, end_shot_id, s, A, b)
        aligned_pdr_shots_dict.update(new_dict)

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
    if len(sfm_points_dict) < 3:
        return {}

    sfm_coords = []
    pdr_coords = []

    sfm_shot_ids = sorted(sfm_points_dict.keys())
    for i in range(len(sfm_shot_ids)):
        shot_id = sfm_shot_ids[i]

        if shot_id in pdr_shots_dict:
            sfm_coords.append(sfm_points_dict[shot_id])
            pdr_coords.append(pdr_shots_dict[shot_id][0:3])

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


def get_affine_transform(gps_coords, pdr_coords):
    """Align pdr output with GPS data.

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

