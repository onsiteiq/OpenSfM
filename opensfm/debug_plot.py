import os
import sys
import json
import glob
import logging

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
    floor_plan_paths = glob.glob('./*FLOOR*.png')

    if not floor_plan_paths or not os.path.exists(floor_plan_paths[0]):
        return

    img = mpimg.imread(floor_plan_paths[0])

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


def debug_plot_reconstructions(reconstructions):
    """
    draw floor plan and aligned pdr shot positions on top of it
    """
    if not debug:
        return

    # floor plan
    floor_plan_paths = glob.glob('./*FLOOR*.png')

    if not floor_plan_paths or not os.path.exists(floor_plan_paths[0]):
        print("No floor plan image found. Quitting")
        return

    img = mpimg.imread(floor_plan_paths[0])

    fig, ax = plt.subplots()
    ax.imshow(img)

    topocentric_gps_points_dict = _load_topocentric_gps_points()
    for key, value in topocentric_gps_points_dict.items():
        circle = plt.Circle((value[0], value[1]), color='blue', radius=50)
        ax.add_artist(circle)
        ax.text(value[0], value[1], str(_shot_id_to_int(key)), color='white', fontsize=6)

    # show each recon in different colors
    colors = ["green", "blue", "cyan", "magenta", "yellow", "white"]
    color_ind = 0

    for reconstruction in reconstructions:
        if color_ind < len(colors):
            color = colors[color_ind]
            color_ind += 1
        else:
            color = colors[-1]

        if not reconstruction.alignment.aligned:
            color = 'red'

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


def _shot_id_to_int(shot_id):
    """
    Returns: shot id to integer
    """
    tokens = shot_id.split(".")
    return int(tokens[0])


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

    if len(sys.argv) > 1:
        filename = str(sys.argv[1])
    else:
        filename = "reconstruction.json"

    with open(filename) as fin:
        reconstructions = io.reconstructions_from_json(json.load(fin))

    debug_plot_reconstructions(reconstructions)
