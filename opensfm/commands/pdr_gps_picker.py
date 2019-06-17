import os
import sys
import glob
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logger = logging.getLogger(__name__)


class LabeledCircle(object):
    def __init__(self, ax, shot_id, center, radius, color, alpha=1.0, fontsize=6):
        self.center = center
        self.fontsize = fontsize

        self.circle = patches.Circle(center, radius, fc=color, alpha=alpha)
        self.text = str(_shot_id_to_int(shot_id))

        ax.add_patch(self.circle)
        ax.text(self.center, self.text, fontsize=self.fontsize)

    def move(self, center):

class DragMover(object):

    def __init__(self, artists):
        self.artists = artists
        self.colors = [a.get_facecolor() for a in self.artists]
        # assume all artists are in the same figure, otherwise selection is meaningless
        self.fig = self.artists[0].figure
        self.ax = self.artists[0].axes

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.currently_selecting = False
        self.currently_dragging = False
        self.selected_artist = None
        self.offset = np.zeros((1, 2))

    def on_press(self, event):
        # is the press over some artist
        isonartist = False
        for artist in self.artists:
            if artist.contains(event)[0]:
                isonartist = artist
        if isonartist:
            # add clicked artist to selection
            self.select_artist(isonartist)
            # start dragging
            self.currently_dragging = True
            ac = np.array([self.selected_artist.center])
            ec = np.array([event.xdata, event.ydata])
            self.offset = ac - ec
        else:
            #start selecting
            self.currently_selecting = True
            self.deselect_artist()

    def on_release(self, event):
        if self.currently_selecting:
            self.fig.canvas.draw_idle()
            self.currently_selecting = False
        elif self.currently_dragging:
            self.currently_dragging = False

    def on_motion(self, event):
        if self.currently_dragging:
            newcenters = np.array([event.xdata, event.ydata])+self.offset
            self.selected_artist.center = newcenters[0]
            self.fig.canvas.draw_idle()

    def select_artist(self, artist):
        self.deselect_artist()

        self.selected_artist = artist
        self.selected_artist.set_color('k')

    def deselect_artist(self):
        for artist, color in zip(self.artists, self.colors):
            artist.set_color(color)
        self.selected_artist = None


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


def draw_floor_plan(plan_path):
    """
    draw floor plan
    """
    fig, ax = plt.subplots(1, 1)

    img = cv2.imread(plan_path, cv2.IMREAD_COLOR)
    ax.imshow(img)

    return ax


def load_pdr_shots(ax, pdr_shots_path):
    """
    draw pdr shots
    """
    pdr_shots_dict = {}

    with open(pdr_shots_path) as fin:
        for line in fin:
            (shot_id, x, y, z, roll, pitch, heading, delta_distance) = line.split()
            pdr_shots_dict[shot_id] = (float(x), float(y), float(z),
                                       float(roll), float(pitch), float(heading),
                                       float(delta_distance))

    circles = [patches.Circle((400.0, 400.0), 100, fc='r', alpha=1.0),
               patches.Circle((1000.0, 1000.0), 100, fc='b', alpha=1.0),
               patches.Circle((2000.0, 2000.0), 100, fc='g', alpha=1.0)]

    for circle in circles:
        ax.add_patch(circle)


def pdr_gps_picker(plan_path, pdr_shots_path):
    """
    main routine to launch gps picker
    """
    # draw floor plan
    ax = draw_floor_plan(plan_path)

    # load pdr shots
    shots = load_pdr_shots(pdr_shots_path)

    # start drag mover
    drag_mover = DragMover(ax, shots)

    plt.show()


if __name__ == '__main__':
    # floor plan
    plan_paths = []
    for plan_type in ('./*FLOOR*.png', './*ROOF*.png'):
        plan_paths.extend(glob.glob(plan_type))

    if not plan_paths or not os.path.exists(plan_paths[0]):
        logger.error("floor plan not found!")
        exit(0)

    pdr_shots_paths = []
    for pdr_shots_type in ('./pdr_shots.txt', './osfm/pdr_shots.txt'):
        pdr_shots_paths.extend(glob.glob(pdr_shots_type))

    if not pdr_shots_paths or not os.path.exists(pdr_shots_paths[0]):
        logger.error("pdr shots not found!")
        exit(0)

    pdr_gps_picker(plan_paths[0], pdr_shots_paths[0])
