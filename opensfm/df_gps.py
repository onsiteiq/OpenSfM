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


def extrapolate_pos(node_idx, df_indices, doors, door_1_idx, door_2_idx, pdr_shots_dict):
    df_1_idx = df_indices[node_idx-2]
    df_2_idx = df_indices[node_idx-1]

    door_gps_coords = []
    door_gps_coords.append(doors[door_1_idx])
    door_gps_coords.append(doors[door_2_idx])

    pdr_coords = []
    pdr_coords.append([pdr_shots_dict[df_1_idx][0], pdr_shots_dict[df_1_idx][1], 0])
    pdr_coords.append([pdr_shots_dict[df_2_idx][0], pdr_shots_dict[df_2_idx][1], 0])

    s, A, b = get_affine_transform_2d_no_numpy(door_gps_coords, pdr_coords)

    pos = apply_affine_transform_no_numpy(pdr_shots_dict, df_indices[node_idx], s, A, b)
    return pos


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


def apply_affine_transform_no_numpy(pdr_shots_dict, shot_id, s, A, b):
    X = pdr_shots_dict[shot_id]
    A_dot_X = [A[0][0]*X[0] + A[0][1]*X[1] + A[0][2]*X[2],
               A[1][0]*X[0] + A[1][1]*X[1] + A[1][2]*X[2],
               A[2][0]*X[0] + A[2][1]*X[1] + A[2][2]*X[2]]
    Xp = [i*s + j for i, j in zip(A_dot_X, b)]

    return Xp


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