"""Convert door frame detections to GPS points."""

import logging
import math
import numpy as np
from pomegranate import *

logger = logging.getLogger(__name__)


def df_gps(data):
    """
    globally align door frames to floor plan door annotations.
    the alignment is based on pdr using a bayesian net

    :param data:
    """
    # TODO: TESTING
    #doors = data.load_high_level_features()

    # door coordinates for 2 Wash, 05/09, 'AX-109A - CONSTRUCTION FLOOR - 15'
    doors = [
        (3350, 4127, 0),
        (4483, 3632, 0),
        (4377, 3550, 0),
        (4483, 3800, 0)]

    # TODO: TESTING
    #df_indices = data.load_df_detect()

    # detected door frame image indices
    df_indices = [0, 100, 200, 300]

    if not data.pdr_shots_exist():
        return

    scale_factor = data.config['reconstruction_scale_factor']
    pdr_shots_dict = data.load_pdr_shots()

    # build the bayesian network model. each detected door is a node in this network. each node
    # corresponds to a door on the floor plan. the probability of it being a particular door on
    # the floor plan depends on the previous two doors (i.e. this is a 2nd order markov chain)
    # and pdr path predictions
    bayes_model = construct_bayes_net(df_indices, pdr_shots_dict, doors)

    # make a maximum likelihood prediction over which doors on floor plan corresponds to detected
    # door frames
    predicted_doors = make_door_prediction(bayes_model)

    # convert predictions to gps points
    topocentric_gps_points_dict = predictions_to_gps(df_indices, predicted_doors)
    data.save_topocentric_gps_points(topocentric_gps_points_dict)


def gen_1st_node_distribution(num_doors):
    prob_dict = {}
    for i in range(num_doors):
        prob_dict[str(i)] = 1./num_doors

    return DiscreteDistribution(prob_dict)


def gen_2nd_node_distribution(df_indices, parent_node):
    cond_table = ConditionalProbabilityTable()

    return cond_table


def gen_next_node_distribution(node_idx, df_indices, pdr_shots_dict, doors, parent_node_1, parent_node_2):
    cond_table = ConditionalProbabilityTable()
    return cond_table


def construct_bayes_net(df_indices, pdr_shots_dict, doors):
    model = BayesianNetwork("Door Predictor")

    s1 = Node(gen_1st_node_distribution(len(doors)), name="Node 1")
    model.add_state(s1)

    s2 = Node(gen_2nd_node_distribution(df_indices, s1), name="Node 2")
    model.add_state(s2)
    model.add_edge(s1, s2)

    parent_node_1 = s1
    parent_node_2 = s2
    for i in range(2, len(df_indices)):
        dist = gen_next_node_distribution(i, df_indices, pdr_shots_dict, doors, parent_node_1, parent_node_2)
        s = Node(dist, name="Node "+str(i))
        model.add_state(s)
        model.add_edge(parent_node_1, s)
        model.add_edge(parent_node_2, s)
        parent_node_1 = parent_node_2
        parent_node_2 = s

    model.bake()
    return model


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

        if shot_id in pdr_shots_dict:
            X = pdr_shots_dict[shot_id]
            A_dot_X = [A[0][0]*X[0] + A[0][1]*X[1] + A[0][2]*X[2],
                          A[1][0]*X[0] + A[1][1]*X[1] + A[1][2]*X[2],
                          A[2][0]*X[0] + A[2][1]*X[1] + A[2][2]*X[2]]
            Xp = [i*s + j for i, j in zip(A_dot_X, b)]
            new_dict[shot_id] = [Xp[0], Xp[1], Xp[2]]

    return new_dict


def make_door_prediction(model, df_indices):
    node_list = []
    for i in range(len(df_indices)):
        node_list.append(None)

    return model.predict(node_list)


def predictions_to_gps(df_indices, predicted_doors):
    return {}


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
