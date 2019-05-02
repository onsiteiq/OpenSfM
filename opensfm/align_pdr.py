"""affine transform pdr output to align with GPS data."""

import logging
import numpy as np

from opensfm import transformations as tf

logger = logging.getLogger(__name__)


def align_pdr(gps_points_dict, pdr_shots_dict):
    """
    Move a sliding window through the gps points and get 3 neighboring points at a time;
    use them to align pdr output of shots that fall within that window

    :param gps_points_dict: gps points in topocentric coordinates
    :param pdr_shots_dict: position of each shot as predicted by pdr
    :return: pdr shot positions aligned with gps points
    """
    if len(gps_points_dict) < 3:
        return {}

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

        new_dict = apply_affine_transform(pdr_shots_dict, gps_shot_ids[i], gps_shot_ids[i+2], s, A, b)
        aligned_pdr_shots_dict.update(new_dict)

    return aligned_pdr_shots_dict


def apply_affine_transform(pdr_shots_dict, start_shot_id, end_shot_id, s, A, b):
    """Apply a similarity (y = s A x + b) to a reconstruction.

    :param pdr_shots_dict: all pdr predictions
    :param start_shot_id: shot id of 1st gps point
    :param end_shot_id: shot id of 3rd gps point
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

