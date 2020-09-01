import os
import sys
import json
import glob
import logging
import six
import cv2
import math
import numpy as np

from six import iteritems

from opensfm import io
from opensfm import geo
from opensfm import multiview
from opensfm import transformations as tf
from scipy.interpolate import splprep, splev, spalde


import matplotlib.pyplot as plt

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def debug_plot_pdr(topocentric_gps_points_dict, pdr_predictions_dict):
    """
    draw floor plan and aligned pdr shot positions on top of it
    """
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


def _vector_angle(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)


def _prev_shot_id(curr_shot_id):
    """
    Returns: previous shot id
    """
    return _int_to_shot_id(_shot_id_to_int(curr_shot_id) - 1)


def _next_shot_id(curr_shot_id):
    """
    Returns: next shot id
    """
    return _int_to_shot_id(_shot_id_to_int(curr_shot_id) + 1)


def diff_recon_preint(im1, im2, reconstruction, pdr_shots_dict):
    """
    compare rel recon rotation of two images to that of imu gyro preintegration,
    if they are not close, the recon of the images is considered to be an erroneous
    """
    # calculate relative rotation from preintegrated gyro input
    preint_im1_rot = cv2.Rodrigues(np.asarray([pdr_shots_dict[im1][6], pdr_shots_dict[im1][7], pdr_shots_dict[im1][8]]))[0]
    preint_im2_rot = cv2.Rodrigues(np.asarray([pdr_shots_dict[im2][6], pdr_shots_dict[im2][7], pdr_shots_dict[im2][8]]))[0]
    preint_rel_rot = np.dot(preint_im2_rot, preint_im1_rot.T)

    # convert this rotation from sensor frame to camera frame
    b_to_c = np.asarray([1, 0, 0, 0, 0, -1, 0, 1, 0]).reshape(3, 3)
    preint_rel_rot = cv2.Rodrigues(b_to_c.dot(cv2.Rodrigues(preint_rel_rot)[0].ravel()))[0]

    R1 = reconstruction.shots[im1].pose.get_rotation_matrix()
    R2 = reconstruction.shots[im2].pose.get_rotation_matrix()
    recon_rel_rot = np.dot(R1, R2.T)

    #logger.debug("preint {}".format(np.degrees(_rotation_matrix_to_euler_angles(preint_rel_rot))))
    #logger.debug("recon {}".format(np.degrees(_rotation_matrix_to_euler_angles(recon_rel_rot))))

    diff_rot = np.dot(preint_rel_rot, recon_rel_rot.T)

    return np.linalg.norm(cv2.Rodrigues(diff_rot)[0].ravel())

    # instead of returning norm of the axis-angle difference, we return only the Y component which is
    # more relevant to the mapping use case we have
    #return abs(_rotation_matrix_to_euler_angles(diff_rot)[1])


def debug_plot_reconstruction(reconstruction, pdr_shots_dict, culling_dict):
    '''
    draw an individual reconstruction

    :param reconstruction:
    :return:
    '''
    fig, ax = plt.subplots(nrows=1, ncols=1)

    X = []
    Y = []
    for shot in reconstruction.shots.values():
        p = shot.pose.get_origin()

        X.append(p[0])
        Y.append(p[1])

        ax.text(p[0], p[1], str(_shot_id_to_int(shot.id)), fontsize=8)

    plt.plot(X, Y, 'g-', linewidth=1)

    X1 = []
    Y1 = []
    U1 = []

    for shot_id in sorted(reconstruction.shots):
        curr_origin = reconstruction.shots[shot_id].pose.get_origin()

        X1.append(curr_origin[0])
        Y1.append(curr_origin[1])
        U1.append(_shot_id_to_int(shot_id))

    #tcku, fp, ier, msg = splprep([X1, Y1], u=U1, s=len(U1)*0.1, full_output=1)
    tcku, fp, ier, msg = splprep([X1, Y1], u=U1, s=0, full_output=1)
    tck, u = tcku

    u_tenth = np.arange(u[0], u[-1], 0.1)
    new_points = splev(u_tenth, tck)
    plt.plot(new_points[0], new_points[1], 'r-')

    alde = spalde(u_tenth, tck)
    x_prime = np.asarray(alde[0])[:, 1]
    x_double_prime = np.asarray(alde[0])[:, 2]
    y_prime = np.asarray(alde[1])[:, 1]
    y_double_prime = np.asarray(alde[1])[:, 2]


    # curvature of parametric curve at (x, y) = |x'y'' - y'x''| / [(x')^2 + (y')^2]^1.5
    curvature = np.divide(np.multiply(x_prime, y_double_prime) - np.multiply(y_prime, x_double_prime),
                          np.power(np.multiply(x_prime, x_prime) + np.multiply(y_prime, y_prime), 1.5))

    '''
    x = np.asarray(new_points[0])
    y = np.asarray(new_points[1])
    for i, val in enumerate(u_tenth):
        logger.debug("{:04.1f} = {} {}, prime={} {}, double_prime={} {}".format(val, x[i], y[i], x_prime[i], y_prime[i], x_double_prime[i], y_double_prime[i]))
        
    for i, c in enumerate(curvature):
        logger.debug("{}={}".format(u_tenth[i], c))

    for i in range(int(u[0]), int(u[-1])):
        start_index = (i - int(u[0]))*10
        max_c = np.amax(curvature[start_index:start_index+10])
        logger.debug("curvature {:04.1f} = {}, {}".format(i, max_c, curvature[start_index]))
    '''

    shot_ids = sorted(reconstruction.shots)
    for shot_id in shot_ids:
        next_shot_id = _next_shot_id(shot_id)
        prev_shot_id = _prev_shot_id(shot_id)

        origin_curr = reconstruction.shots[shot_id].pose.get_origin()

        if next_shot_id in reconstruction.shots and _is_no_culling_in_between(culling_dict, shot_id, next_shot_id) and \
           prev_shot_id in reconstruction.shots and _is_no_culling_in_between(culling_dict, shot_id, prev_shot_id):
            vector_next = reconstruction.shots[next_shot_id].pose.get_origin() - origin_curr
            vector_prev = origin_curr - reconstruction.shots[prev_shot_id].pose.get_origin()
            angle = _vector_angle(vector_next, vector_prev)
            if angle > np.pi * 2.0 / 3.0:
                d_next = np.linalg.norm(vector_next)
                d_prev = np.linalg.norm(vector_prev)

                # given that frames are 0.5s apart (not enough time for turning around), it's highly unlikely to have
                # any significant movement that's forward then backward
                if d_next > 0.25 and d_prev > 0.25 and (d_next > 0.5 or d_prev > 0.5):
                    logger.debug("{} {} {} angle = {} d_prev = {} d_next {}".format(prev_shot_id, shot_id, next_shot_id,
                                                                                    np.degrees(angle), d_prev, d_next))

        if next_shot_id in reconstruction.shots and _is_no_culling_in_between(culling_dict, shot_id, next_shot_id):
            geo_diff = diff_recon_preint(shot_id, next_shot_id, reconstruction, pdr_shots_dict)

            if geo_diff > 10.0/180.0*np.pi: # 10 degrees
                logger.debug("{} {} rotation diff {} degrees".format(shot_id, next_shot_id, np.degrees(geo_diff)))

            vector_next = reconstruction.shots[next_shot_id].pose.get_origin() - origin_curr
            distance = np.linalg.norm(vector_next)
            if distance > 1.25:
                logger.debug("{} {} coords distance {} meters".format(shot_id, next_shot_id, distance))

    plt.show()


def debug_plot_reconstructions(reconstructions):
    """
    draw floor plan and aligned pdr shot positions on top of it
    """
    # floor plan
    plan_paths = []
    for plan_type in ('./*FLOOR*.png', './*ROOF*.png'):
        plan_paths.extend(glob.glob(plan_type))

    if not plan_paths or not os.path.exists(plan_paths[0]):
        print("No floor plan image found. Quitting")
        return

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
            print(i, o[0], o[1], o[2])


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


def transform_reconstruction(reconstruction, ref_shots_dict):
    """
    transform recon based on two reference positions
    :param reconstruction:
    :param ref_shots_dict:
    :return:
    """
    X, Xp = [], []
    onplane, verticals = [], []
    for shot_id in ref_shots_dict:
        X.append(reconstruction.shots[shot_id].pose.get_origin())
        Xp.append(ref_shots_dict[shot_id])
        R = reconstruction.shots[shot_id].pose.get_rotation_matrix()
        onplane.append(R[0,:])
        onplane.append(R[2,:])
        verticals.append(R[1,:])

    X = np.array(X)
    Xp = np.array(Xp)

    # Estimate ground plane.
    p = multiview.fit_plane(X - X.mean(axis=0), onplane, verticals)
    Rplane = multiview.plane_horizontalling_rotation(p)
    X = Rplane.dot(X.T).T

    # Estimate 2d similarity to align to pdr predictions
    T = tf.affine_matrix_from_points(X.T[:2], Xp.T[:2], shear=False)
    s = np.linalg.det(T[:2, :2]) ** 0.5
    A = np.eye(3)
    A[:2, :2] = T[:2, :2] / s
    A = A.dot(Rplane)
    b = np.array([
        T[0, 2],
        T[1, 2],
        Xp[:, 2].mean() - s * X[:, 2].mean()  # vertical alignment
    ])

    # Align points.
    for point in reconstruction.points.values():
        p = s * A.dot(point.coordinates) + b
        point.coordinates = p.tolist()

    # Align cameras.
    for shot in reconstruction.shots.values():
        R = shot.pose.get_rotation_matrix()
        t = np.array(shot.pose.translation)
        Rp = R.dot(A.T)
        tp = -Rp.dot(b) + s * t
        try:
            shot.pose.set_rotation_matrix(Rp)
            shot.pose.translation = list(tp)
        except:
            logger.debug("unable to transform reconstruction!")


def flatten_reconstruction(reconstruction):
    shot_ids = sorted(reconstruction.shots)

    ref_shots_dict = {}
    ref_shots_dict[shot_ids[0]] = (0, 0, 0)
    ref_shots_dict[shot_ids[-1]] = (1, 1, 0)

    transform_reconstruction(reconstruction, ref_shots_dict)


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


def _is_no_culling_in_between(culling_dict, shot_id_1, shot_id_2):
    if not culling_dict:
        return True

    i1 = _shot_id_to_int(culling_dict[shot_id_1])
    i2 = _shot_id_to_int(culling_dict[shot_id_2])
    if abs(i1-i2) == 1:
        return True
    else:
        return False


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


def _rotation_matrix_to_euler_angles(R):
    """
    The result is the same as MATLAB except the order of the euler angles ( x and z are swapped ).
    https://www.learnopencv.com/rotation-matrix-to-euler-angles/
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

    return np.array([x, y, z])


# Entry point
if __name__ == "__main__":

    show_num = -1

    if len(sys.argv) > 1:
        show_num = int(sys.argv[1])

    with open('reconstruction.json') as fin:
        reconstructions = io.reconstructions_from_json(json.load(fin))

    reconstructions = sorted(reconstructions,
                             key=lambda x: -len(x.shots))

    pdr_shots_dict = {}
    with open('pdr_shots.txt') as fin:
        for line in fin:
            (shot_id, x, y, z, roll, pitch, heading, omega_0, omega_1, omega_2, delta_distance) = line.split()
            pdr_shots_dict[shot_id] = (float(x), float(y), float(z),
                                       float(roll), float(pitch), float(heading),
                                       float(omega_0), float(omega_1), float(omega_2),
                                       float(delta_distance))

    culling_dict = {}
    if os.path.exists('../frames_redundant/undo.txt'):
        with open('../frames_redundant/undo.txt') as fin:
            for line in fin:
                (shot_id, orig_shot_id) = line.split()
                culling_dict[shot_id] = orig_shot_id

    if show_num == -1:
        debug_plot_reconstructions(reconstructions)
    else:
        debug_plot_reconstruction(reconstructions[show_num], pdr_shots_dict, culling_dict)
