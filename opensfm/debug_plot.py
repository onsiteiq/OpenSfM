import os
import sys
import json
import glob
import logging
import six
import cv2

from six import iteritems

from opensfm import io
from opensfm import geo

debug = False

if debug:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

logger = logging.getLogger(__name__)


def debug_plot_pdr(topocentric_gps_points_dict, pdr_predictions_dict):
    """
    draw floor plan and aligned pdr shot positions on top of it
    """
    if not debug:
        return

    logger.info("debug_plot_pdr {}".format(len(pdr_predictions_dict)))

    for key, value in topocentric_gps_points_dict.items():
        logger.info("gps point {} = {} {} {}".format(key, value[0], value[1], value[2]))

    for key, value in pdr_predictions_dict.items():
        logger.info("aligned pdr point {} = {} {} {}, dop = {}".
                    format(key, value[0], value[1], value[2], value[3]))

    # floor plan
    # floor plan
    plan_paths = []
    for plan_type in ('./*FLOOR*.png', './*ROOF*.png'):
        plan_paths.extend(glob.glob(plan_type))

    if not plan_paths or not os.path.exists(plan_paths[0]):
        return

    #img = mpimg.imread(plan_paths[0])
    img = cv2.imread(plan_paths[0], cv2.IMREAD_COLOR)

    fig, ax = plt.subplots()
    ax.imshow(img)

    shot_ids = sorted(pdr_predictions_dict.keys())
    X = []
    Y = []
    for shot_id in shot_ids:
        value = pdr_predictions_dict[shot_id]
        X.append(value[0])
        Y.append(value[1])
        #logger.info("aligned pdr positions {} = {}, {}, {}".format(shot_id, value[0], value[1], value[2]))

    plt.plot(X, Y, linestyle='-', color='red', linewidth=3)

    for key, value in topocentric_gps_points_dict.items():
        circle = plt.Circle((value[0], value[1]), color='green', radius=100)
        ax.add_artist(circle)
        ax.text(value[0], value[1], str(_shot_id_to_int(key)), fontsize=8)
        #logger.info("topocentric gps positions {} = {}, {}, {}".format(shot_id, value[0], value[1], value[2]))

    plt.show()
    #fig.savefig('./aligned_pdr_path.png', dpi=200)


def debug_plot_reconstruction(reconstruction):
    '''
    draw an individual reconstruction

    :param reconstruction:
    :return:
    '''
    if not debug:
        return

    fig, ax = plt.subplots(nrows=1, ncols=1)

    X = []
    Y = []
    for shot in reconstruction.shots.values():
        p = shot.pose.get_origin()

        X.append(p[0])
        Y.append(p[1])

        ax.text(p[0], p[1], str(_shot_id_to_int(shot.id)), fontsize=6)

    plt.plot(X, Y, linestyle='-', color='green', linewidth=3)
    plt.show()


def debug_plot_reconstructions(reconstructions):
    """
    draw floor plan and aligned pdr shot positions on top of it
    """
    if not debug:
        return

    # floor plan
    plan_paths = []
    for plan_type in ('./*FLOOR*.png', './*ROOF*.png'):
        plan_paths.extend(glob.glob(plan_type))

    if not plan_paths or not os.path.exists(plan_paths[0]):
        print("No floor plan image found. Quitting")
        return

    #img = mpimg.imread(plan_paths[0])
    img = cv2.imread(plan_paths[0], cv2.IMREAD_COLOR)

    fig, ax = plt.subplots()
    ax.imshow(img)

    topocentric_gps_points_dict = _load_topocentric_gps_points()
    for key, value in topocentric_gps_points_dict.items():
        circle = plt.Circle((value[0], value[1]), color='black', radius=50)
        ax.add_artist(circle)
        ax.text(value[0], value[1], str(_shot_id_to_int(key)), color='white', fontsize=6)

    # show each recon in different colors
    colors = ["green", "blue", "cyan", "magenta", "yellow"]
    color_ind = 0

    for reconstruction in reconstructions:
        if not reconstruction.alignment.aligned:
            color = 'red'
        else:
            color = colors[color_ind]
            color_ind = (color_ind + 1) % len(colors)

        for shot in reconstruction.shots.values():
            if shot.metadata.gps_dop != 999999.0:
                radius = 50
            else:
                radius = 25

            p = shot.pose.get_origin()
            circle = plt.Circle((p[0], p[1]), color=color, radius=radius)
            ax.add_artist(circle)
            ax.text(p[0], p[1], str(_shot_id_to_int(shot.id)), fontsize=6)

    plt.show()
    #fig.savefig('./recon.png', dpi=200)


def debug_print_origin(reconstruction, start_shot_idx, end_shot_idx):
    """
    print origin of shots between start/end_shot_idx
    """
    logger.debug("debug_print_origin: origin of shots between {} and {}".format(start_shot_idx, end_shot_idx))
    for i in range(start_shot_idx, end_shot_idx):
        id = _int_to_shot_id(i)
        if id in reconstruction.shots:
            o = reconstruction.shots[id].pose.get_origin()
            logger.debug("debug_print_origin: id={}, pos={} {} {}".format(i, o[0], o[1], o[2]))


def debug_save_reconstruction(data, graph, reconstruction, curr_shot_idx, start_shot_idx, end_shot_idx):
    """
    save partial recon if shot idx falls between start/end_shot_idx
    """
    if curr_shot_idx in range(start_shot_idx, end_shot_idx):
        """Set the color of the points from the color of the tracks."""
        for k, point in iteritems(reconstruction.points):
            point.color = six.next(six.itervalues(graph[k]))['feature_color']
        data.save_reconstruction(
            [reconstruction], 'reconstruction.{}.json'.format(curr_shot_idx))


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


def _load_topocentric_gps_points():
    topocentric_gps_points_dict = {}

    with open("gps_list.txt") as fin:
        gps_points_dict = io.read_gps_points_list(fin)

    with io.open_rt("reference_lla.json") as fin:
        reflla = io.json_load(fin)

    for key, value in gps_points_dict.items():
        x, y, z = geo.topocentric_from_lla(
            value[0], value[1], value[2],
            reflla['latitude'], reflla['longitude'], reflla['altitude'])
        topocentric_gps_points_dict[key] = (x, y, z)

    return topocentric_gps_points_dict


# Entry point
if __name__ == "__main__":
    debug = True
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    show_num = -1

    if len(sys.argv) > 1:
        show_num = int(sys.argv[1])

    with open('reconstruction.json') as fin:
        reconstructions = io.reconstructions_from_json(json.load(fin))

    reconstructions = sorted(reconstructions,
                             key=lambda x: -len(x.shots))
    if show_num == -1:
        debug_plot_reconstructions(reconstructions)
    else:
        debug_plot_reconstruction(reconstructions[show_num])
