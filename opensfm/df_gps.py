"""Convert door frame detections to GPS points."""

import operator
import logging
import math
import numpy as np
from pomegranate import *

from opensfm import geo
from opensfm import transformations as tf

from opensfm.debug_plot import debug_plot_pdr

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


def make_door_prediction(model, df_indices):
    node_list = []
    for i in range(len(df_indices)):
        node_list.append(None)

    return model.predict(node_list)


def predictions_to_gps(df_indices, predicted_doors):
    return {}
