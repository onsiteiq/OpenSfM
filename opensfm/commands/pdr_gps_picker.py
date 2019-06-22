import os
import sys
import glob
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.text as text

from opensfm import config
from opensfm import align_pdr

logger = logging.getLogger(__name__)


class LabeledCircle(object):
    circle_radius = 30
    circle_radius_gps = 50

    circle_color = 'g'
    circle_color_gps = 'r'
    circle_color_selected = 'y'

    text_size = 4
    text_size_gps = 6

    def __init__(self, shot_id, center):
        self.is_gps = False
        self.shot_id = shot_id

        self.circle = patches.Circle(center, self.circle_radius, fc=self.circle_color)
        self.text = text.Text(center[0], center[1], str(self._shot_id_to_int(shot_id)))

    def show(self, ax):
        ax.add_patch(self.circle)
        ax.add_artist(self.text)

    def contains(self, event):
        return self.circle.contains(event)

    def get_shot_id(self):
        return self.shot_id

    def set_is_gps(self, is_gps):
        self.is_gps = is_gps

        if is_gps:
            self.circle.set_color(self.circle_color_gps)
            self.circle.set_radius(self.circle_radius_gps)

    def set_selected(self, is_selected):
        if is_selected:
            self.circle.set_color(self.circle_color_selected)
        elif self.is_gps:
            self.circle.set_color(self.circle_color_gps)
        else:
            self.circle.set_color(self.circle_color)

    def get_is_gps(self):
        return self.is_gps

    def set_center(self, center):
        self.circle.center = center
        self.text.set_position(center)

    def get_center(self):
        return self.circle.center

    def get_facecolor(self):
        return self.circle.get_facecolor()

    def _shot_id_to_int(self, shot_id):
        """
        Returns: shot id to integer
        """
        tokens = shot_id.split(".")
        return int(tokens[0])

    def _int_to_shot_id(self, shot_int):
        """
        Returns: integer to shot id
        """
        return str(shot_int).zfill(10) + ".jpg"


class DragMover(object):
    def __init__(self, fig, ax, pdr_shots_dict, scale_factor, num_extrapolation):
        self.fig = fig
        self.ax = ax
        self.pdr_shots_dict = pdr_shots_dict
        self.scale_factor = scale_factor
        self.num_extrapolation = num_extrapolation
        self.shot_objs = []

        # scale shots to roughly the right scale and make them visible
        self.update()

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.currently_selecting = False
        self.currently_dragging = False
        self.selected_shot_obj = None
        self.offset = np.zeros((1, 2))

    def on_press(self, event):
        # is the press over some shot object
        is_on_shot_obj = False
        for shot_obj in self.shot_objs:
            if shot_obj.contains(event)[0]:
                is_on_shot_obj = shot_obj
        if is_on_shot_obj:
            # add clicked artist to selection
            self.select_shot_obj(is_on_shot_obj)
            # start dragging
            self.currently_dragging = True
            ac = np.array(self.selected_shot_obj.get_center())
            ec = np.array([event.xdata, event.ydata])
            self.offset = tuple(np.subtract(ac, ec))
        else:
            #start selecting
            self.currently_selecting = True
            self.deselect_shot_obj()

        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.currently_selecting:
            self.fig.canvas.draw_idle()
            self.currently_selecting = False
        elif self.currently_dragging:
            self.currently_dragging = False

    def on_motion(self, event):
        if self.currently_dragging:
            self.update(event)

    def update(self, event=None):
        if event:
            new_centers = np.array([event.xdata, event.ydata])+self.offset
            self.selected_shot_obj.set_center(new_centers[0])
            self.selected_shot_obj.set_is_gps(True)

        curr_gps_points_dict = {}
        for shot_obj in self.shot_objs:
            if shot_obj.get_is_gps():
                curr_gps_points_dict[shot_obj.get_shot_id()] = (shot_obj.get_center()[0], shot_obj.get_center()[1], 0)

        pdr_predictions_dict = align_pdr.update_gps_picker(curr_gps_points_dict, self.pdr_shots_dict,
                                                           self.scale_factor, self.num_extrapolation)

        for shot_obj in self.shot_objs:
            shot_id = shot_obj.get_shot_id()

            if shot_id in pdr_predictions_dict:
                shot_obj.set_center(pdr_predictions_dict[shot_id])
                del pdr_predictions_dict[shot_id]

        for shot_id in pdr_predictions_dict:
            shot_obj = LabeledCircle(shot_id, pdr_predictions_dict[shot_id])
            shot_obj.show(self.ax)
            self.shot_objs.append(shot_obj)

        self.fig.canvas.draw_idle()

    def select_shot_obj(self, shot_obj):
        self.deselect_shot_obj()

        self.selected_shot_obj = shot_obj
        self.selected_shot_obj.set_selected(True)

    def deselect_shot_obj(self):
        for shot_obj in self.shot_objs:
            shot_obj.set_selected(False)
        self.selected_shot_obj = None


def draw_floor_plan(plan_path):
    """
    draw floor plan
    """
    fig, ax = plt.subplots(1, 1)

    img = cv2.imread(plan_path, cv2.IMREAD_COLOR)
    ax.imshow(img)

    return fig, ax


def load_pdr_shots(pdr_shots_path):
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

    return pdr_shots_dict


def pdr_gps_picker(plan_path, pdr_shots_path, scale_factor, num_extrapolation):
    """
    main routine to launch gps picker
    """
    # draw floor plan
    fig, ax = draw_floor_plan(plan_path)

    # load pdr shots
    pdr_shots_dict = load_pdr_shots(pdr_shots_path)

    # start drag mover
    drag_mover = DragMover(fig, ax, pdr_shots_dict, scale_factor, num_extrapolation)

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        pdr_extrapolation_frames = int(sys.argv[1])
    else:
        pdr_extrapolation_frames = 50

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

    config_file_paths = []
    for config_file_type in ('./config.yaml', './osfm/config.yaml'):
        config_file_paths.extend(glob.glob(config_file_type))

    if not config_file_paths or not os.path.exists(config_file_paths[0]):
        logger.error("config file not found!")
        exit(0)
    else:
        data_config = config.load_config(config_file_paths[0])

    pdr_gps_picker(plan_paths[0], pdr_shots_paths[0],
                   data_config['reconstruction_scale_factor'], pdr_extrapolation_frames)
