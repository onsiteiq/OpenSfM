"""Convert door frame detections to GPS points."""

import logging
import math
import numpy as np
from pomegranate import *
from itertools import combinations

logger = logging.getLogger(__name__)


def df_gps(data):
    """
    globally align door frames detections to floor plan door annotations. the alignment
    is done using a bayesian network, where conditional probabilities are determined by
    pdr measurements

    :param data:
    """
    # TODO: TESTING
    #doors = data.load_high_level_features()

    # door coordinates for 2 Wash, 05/09, 'AX-109A - CONSTRUCTION FLOOR - 15'
    doors = [
        (3340, 3909, 0),
        (4595, 3356, 0),
        (5146, 4408, 0),
        (4680, 3227, 0)]

    # TODO: TESTING
    #df_indices = data.load_df_detect()

    # detected door frame image indices
    df_indices = [13, 90, 159, 173]

    if not data.pdr_shots_exist():
        return

    scale_factor = data.config['reconstruction_scale_factor']
    pdr_shots_dict = data.load_pdr_shots()

    # build the bayesian network model. each detected door is a node in this network. each node
    # corresponds to a door on the floor plan. the probability of it being a particular door on
    # the floor plan depends on the previous two doors (i.e. this is a 2nd order markov chain)
    # and pdr path predictions
    bayes_model = construct_bayes_net(df_indices, pdr_shots_dict, doors, scale_factor)

    # make a maximum likelihood prediction over which doors on floor plan corresponds to detected
    # door frames
    predicted_doors = make_door_predictions(bayes_model, len(df_indices))
    logger.debug(predicted_doors)

    # convert predictions to gps points
    topocentric_gps_points_dict = predictions_to_gps(predicted_doors, doors, df_indices)
    data.save_topocentric_gps_points(topocentric_gps_points_dict)


def gen_node_0_distribution(num_doors):
    logger.debug("gen_node_distribution: node 0")

    prob_dict = {}
    for i in range(num_doors):
        prob_dict[str(i)] = 1./num_doors

    return DiscreteDistribution(prob_dict)


def gen_node_1_distribution(scale_factor, df_indices, pdr_shots_dict, doors, parent_node):
    logger.debug("gen_node_distribution: node 1")

    shot_id_1 = _int_to_shot_id(df_indices[0])
    shot_id_2 = _int_to_shot_id(df_indices[1])

    pdr_diff = ((pdr_shots_dict[shot_id_1][0]-pdr_shots_dict[shot_id_2][0])/(scale_factor * 0.3048),
                (pdr_shots_dict[shot_id_1][1]-pdr_shots_dict[shot_id_2][1])/(scale_factor * 0.3048), 0)
    pdr_distance = np.linalg.norm(pdr_diff)
    #logger.debug("pdr_distance = {}".format(pdr_distance))

    cpt_entries = []
    for i in range(len(doors)):
        distance_doors = np.linalg.norm(np.asarray(doors) - np.asarray(doors[i]), axis=1)

        prob = 1./(np.fabs(distance_doors - pdr_distance))
        prob = prob/np.sum(prob)
        for j in range(len(doors)):
            cpt_entries.append([str(i), str(j), prob[j]])

    cond_table = ConditionalProbabilityTable(cpt_entries, [parent_node.distribution])
    #logger.debug("gen_2nd_node_distribution: {}".format(cond_table))

    return cond_table


def gen_node_n_distribution(scale_factor, node_idx, df_indices, pdr_shots_dict, doors, parent_node_1, parent_node_2):
    logger.debug("gen_node_distribution: node {}".format(node_idx+1))

    cpt_entries = []
    for i in range(len(doors)):
        for j in range(len(doors)):
            if i != j:
                pos = extrapolate_pos(node_idx, df_indices, doors, i, j, pdr_shots_dict)

                # probability for any door is inversely proportional to its squared distance to pos.
                prob = 1./np.linalg.norm(np.asarray(doors) - np.asarray(pos), axis=1)
                prob = prob/np.sum(prob)
            else:
                shot_id_1 = _int_to_shot_id(df_indices[node_idx-1])
                shot_id_2 = _int_to_shot_id(df_indices[node_idx])

                pdr_diff = ((pdr_shots_dict[shot_id_1][0]-pdr_shots_dict[shot_id_2][0])/(scale_factor * 0.3048),
                            (pdr_shots_dict[shot_id_1][1]-pdr_shots_dict[shot_id_2][1])/(scale_factor * 0.3048), 0)
                pdr_distance = np.linalg.norm(pdr_diff)

                distance_doors = np.linalg.norm(np.asarray(doors) - np.asarray(doors[j]), axis=1)

                prob = 1./(np.fabs(distance_doors - pdr_distance))

            prob = prob/np.sum(prob)
            for k in range(len(doors)):
                cpt_entries.append([str(i), str(j), str(k), prob[k]])

    cond_table = ConditionalProbabilityTable(cpt_entries,
                                             [parent_node_1.distribution, parent_node_2.distribution])
    #logger.debug("gen_next_node_distribution: {}".format(cond_table))
    return cond_table


def extrapolate_pos(node_idx, df_indices, doors, door_1_idx, door_2_idx, pdr_shots_dict):
    shot_id_1 = _int_to_shot_id(df_indices[node_idx-2])
    shot_id_2 = _int_to_shot_id(df_indices[node_idx-1])
    shot_id = _int_to_shot_id(df_indices[node_idx])

    door_gps_coords = []
    door_gps_coords.append(doors[door_1_idx])
    door_gps_coords.append(doors[door_2_idx])

    pdr_coords = []
    pdr_coords.append([pdr_shots_dict[shot_id_1][0], pdr_shots_dict[shot_id_1][1], 0])
    pdr_coords.append([pdr_shots_dict[shot_id_2][0], pdr_shots_dict[shot_id_2][1], 0])

    s, A, b = get_affine_transform_2d_no_numpy(door_gps_coords, pdr_coords)

    pos = apply_affine_transform_no_numpy(pdr_shots_dict, shot_id, s, A, b)
    return pos


def get_affine_transform_2d_no_numpy(gps_coords, pdr_coords):
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


def construct_bayes_net(df_indices, pdr_shots_dict, doors, scale_factor):
    model = BayesianNetwork("Door Predictor")

    all_nodes = []
    all_nodes.append(Node(gen_node_0_distribution(len(doors)), name="Node 0"))
    model.add_state(all_nodes[0])

    all_nodes.append(Node(gen_node_1_distribution(scale_factor, df_indices, pdr_shots_dict, doors,
                                                    all_nodes[0]), name="Node 1"))
    model.add_state(all_nodes[1])
    model.add_edge(all_nodes[0], all_nodes[1])

    for n in range(2, len(df_indices)):
        dist = gen_node_n_distribution(scale_factor, n, df_indices, pdr_shots_dict, doors,
                                       all_nodes[n-2], all_nodes[n-1])
        all_nodes.append(Node(dist, name="Node "+str(n)))
        model.add_state(all_nodes[n])
        model.add_edge(all_nodes[n-2], all_nodes[n])
        model.add_edge(all_nodes[n-1], all_nodes[n])

    model.bake()

    logger.debug("Model {}".format(model))
    return model


def make_door_predictions(model, num_nodes):
    pred_node_list = []
    for i in range(num_nodes):
        pred_node_list.append(None)

    #return model.predict([pred_node_list])
    #return model.predict([['0', '1', '2', '3']])
    #return model.predict([['0', '1', '2', None]])
    #return model.predict([['0', '1', None, '3']])
    #return model.predict([['0', None, '2', '3']])
    #return model.predict([[None, '1', '2', '3']])
    #return model.predict([[None, '1', '2', None]])
    return model.predict([['0', '1', None, None]])


def predictions_to_gps(predicted_doors, doors, df_indices):
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
