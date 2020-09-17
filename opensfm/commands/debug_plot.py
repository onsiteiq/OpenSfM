import os
import sys
import json
import glob
import logging
import six
import cv2
import math
import numpy as np
import webbrowser

from six import iteritems

from opensfm import io
from opensfm import geo
from opensfm import multiview
from opensfm import types
from opensfm import tracking
from opensfm import transformations as tf
from scipy.interpolate import splprep, splev, spalde


import matplotlib.pyplot as plt

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# pruning parameters
ABS_HEIGHT_THRESH = 2.5  # in meters
REL_HEIGHT_THRESH = 1.0  # in meters
DISTANCE_THRESH = 1.25  # in meters
MOVEMENT_THRESH = 0.25  # in meters
SIG_MOVEMENT_THRESH = 0.50  # in meters
HEADING_THRESH = 10.0  # in degrees
ANGLE_THRESH = 120  # in degrees

# eliminate NEIGHBOR_RADIUS frames around each side of suspicious frame
NEIGHBOR_RADIUS = 2

# max size of holes. must be smaller than (2*NEIGHBOR_RADIUS+1). set to 1 to disallow holes
MAX_HOLE_SIZE = 3

# number of predictions using PDR
PDR_TRUST_SIZE = 20

# form new recons with more than MIN_RECON_SIZE consecutive frames (with possible hole in the middle)
# this should be twice the PDR_TRUST_SIZE
MIN_RECON_SIZE = 2*PDR_TRUST_SIZE


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
    plan_paths = []
    plan_paths.extend(glob.glob('./*.png'))

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
        ax.text(value[0], value[1], str(_shot_id_to_int(key)), fontsize=10)
        #logger.info("topocentric gps positions {} = {}, {}, {}".format(shot_id, value[0], value[1], value[2]))

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    #fig.savefig('./aligned_pdr_path.png', dpi=200)


def check_scale_change_by_pdr(reconstruction, pdr_shots_dict, culling_dict):
    """
    detect if and where a sudden change of scale happens in the reconstruction. returns a list of
    shot ids where change occur. the caller will need to break the recon at these places.

    :param reconstruction:
    :param pdr_shots_dict:
    :param culling_dict:
    :return:
    """
    shot_ids = sorted(reconstruction.shots)

    distance_dict = {}
    for i in range(_shot_id_to_int(shot_ids[1]), _shot_id_to_int(shot_ids[-2])):
        shot_0 = _int_to_shot_id(i-1)
        shot_1 = _int_to_shot_id(i)
        shot_2 = _int_to_shot_id(i+1)

        if shot_0 in shot_ids and shot_1 in shot_ids and shot_2 in shot_ids and \
                _is_no_culling_in_between(culling_dict, shot_0, shot_1) and \
                _is_no_culling_in_between(culling_dict, shot_1, shot_2):
            vector_prev_recon = reconstruction.shots[shot_1].pose.get_origin() - reconstruction.shots[
                shot_0].pose.get_origin()
            vector_next_recon = reconstruction.shots[shot_2].pose.get_origin() - reconstruction.shots[
                shot_1].pose.get_origin()
            angle_recon = _vector_angle(vector_prev_recon, vector_next_recon)

            vector_prev_pdr = np.asarray(pdr_shots_dict[shot_1][0:3]) - np.asarray(pdr_shots_dict[shot_0][0:3])
            vector_next_pdr = np.asarray(pdr_shots_dict[shot_2][0:3]) - np.asarray(pdr_shots_dict[shot_1][0:3])
            angle_pdr = _vector_angle(vector_prev_pdr, vector_next_pdr)

            if np.linalg.norm(vector_prev_pdr) > 0.2 and np.linalg.norm(vector_next_pdr) > 0.2 and \
                    angle_recon < np.radians(30) and angle_pdr < np.radians(30):
                # we have 3 consecutive shots that, by all indication, are moving on roughly a straight line.
                # we will remember the recon distance that's traveled.
                distance_dict[shot_1] = np.linalg.norm(
                    reconstruction.shots[shot_1].pose.get_origin() - reconstruction.shots[shot_0].pose.get_origin())

    #logger.debug("distance_dict {}".format(distance_dict))

    # detect change event in larger recons
    shot_ids = sorted(distance_dict.keys())
    distances = [distance_dict[i] for i in shot_ids]

    # we look for a place where, recent previous distance measurements are significantly different
    # then the ones that follow. specifically a change in magnitude by more than 40%. the pre and
    # post measurements should each be consistent, specifically the coefficient of variation
    # cannot exceed 33%
    for i in range(len(distance_dict)):
        pre_distances = [distances[j] for j in range(i-1, 0, -1)
                         if _shot_id_to_int(shot_ids[i]) - _shot_id_to_int(shot_ids[j]) < 40]
        if len(pre_distances) < 10:
            continue

        post_distances = [distances[j] for j in range(i, len(distances))
                          if _shot_id_to_int(shot_ids[j]) - _shot_id_to_int(shot_ids[i]) < 40]
        if len(post_distances) < 10:
            continue

        pre_avg = np.mean(pre_distances)
        post_avg = np.mean(post_distances)

        avg_ratio = post_avg / pre_avg
        if avg_ratio < 0.6 or avg_ratio > 1.0 / 0.6:
            pre_stddev = np.std(pre_distances)
            pre_cv = pre_stddev / pre_avg

            post_stddev = np.std(post_distances)
            post_cv = post_stddev/post_avg

            if pre_cv < 0.33 and post_cv < 0.33:
                logger.debug("scale change event is detected at {}, "
                             "pre_avg {}, pre_cv {}, post_avg {}, post_cv {}"
                             .format(shot_ids[i], pre_avg, pre_cv, post_avg, post_cv))

    return True


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


def point_copy(point):
    c = types.Point()
    c.id = point.id
    c.color = point.color
    c.coordinates = point.coordinates.copy()
    c.reprojection_errors = point.reprojection_errors.copy()
    return c


def extract_segment(reconstruction, start_index, end_index, graph, cameras):
    recon_points = set(reconstruction.points)
    segment_points = set()

    segment = types.Reconstruction()
    segment.cameras = cameras

    for shot in reconstruction.shots.values():
        if start_index <= _shot_id_to_int(shot.id) <= end_index:
            segment.add_shot(reconstruction.shots[shot.id])
            segment_points = segment_points | (recon_points & set(graph[shot.id]))

    # need to copy point because a point may belong to more than one segment
    for point_id in segment_points:
        segment.add_point(point_copy(reconstruction.points[point_id]))

    return segment


def prune_reconstructions_by_pdr(reconstructions, pdr_shots_dict, culling_dict, graph, cameras):
    # debugging - see how culling affects breaks
    breaks_total = 0
    breaks_due_to_culling = 0
    for reconstruction in reconstructions:
        break_shot_id = sorted(reconstruction.shots)[-1]
        next_shot_id = _next_shot_id(break_shot_id)
        if _shot_id_to_int(next_shot_id) < len(pdr_shots_dict):
            breaks_total += 1
            if not _is_no_culling_in_between(culling_dict, break_shot_id, next_shot_id):
                breaks_due_to_culling += 1

    if breaks_total > 0:
        logger.debug("culling breaks {} total breaks {}, {:2.1f}%".
                     format(breaks_due_to_culling, breaks_total, breaks_due_to_culling*100/breaks_total))

    segments = []

    recon_quality = 0
    avg_segment_quality = 0
    avg_segment_size = 0
    ratio_shots_in_min_recon = 0
    speedup = 1.0

    total_shots_cnt = 0
    total_bad_shots_cnt = 0

    for reconstruction in reconstructions:
        total_shots_cnt += len(reconstruction.shots)
        new_segments, bad_shots_cnt = prune_reconstruction_by_pdr(reconstruction, pdr_shots_dict, culling_dict, graph, cameras)
        total_bad_shots_cnt += bad_shots_cnt

        segments.extend(new_segments)

    if len(segments) > 0:
        avg_segment_quality = 100-int(total_bad_shots_cnt*100/total_shots_cnt)
        logger.info("Averge segment quality - {}".format(avg_segment_quality))

        segments_shots_cnt = 0
        for segment in segments:
            segments_shots_cnt += len(segment.shots)
        avg_segment_size = segments_shots_cnt/len(segments)
        logger.info("Average good segment size - {}".format(int(avg_segment_size)))

        ratio_shots_in_min_recon = (segments_shots_cnt/total_shots_cnt)
        logger.info("Percentage of shots in good segments - {}%".format(int(100*ratio_shots_in_min_recon)))

        # PDR predicts 20 frames at a time. so if average segment is 40 then we have a 40/20 = 2.0x speed up.
        # the actual speed up depends on other factors and likely lower. avg_segment_size is capped to 100
        # below, because we predicts at most 100 frames at a time
        speedup = 1.0 / (ratio_shots_in_min_recon / math.floor(min(avg_segment_size, 100.0)/20.0)
                         + (1.0 - ratio_shots_in_min_recon))
        logger.info("Estimated speedup Hybrid vs PDR - {:2.1f}x".format(speedup))

        if total_shots_cnt > 4*MIN_RECON_SIZE:
            if speedup < 2.0:
                recon_quality = 0
            else:
                recon_quality = avg_segment_quality
        else:
            if ratio_shots_in_min_recon < 0.5:
                recon_quality = 0
            else:
                recon_quality = avg_segment_quality

        logger.info("Recon quality - {}".format(recon_quality))

    # for gps picker tool, calculate and save a recon quality factor. pdr/hybrid will be based on it.
    #data.save_recon_quality(recon_quality, avg_segment_size, ratio_shots_in_min_recon, speedup)

    return segments


def prune_reconstruction_by_pdr(reconstruction, pdr_shots_dict, culling_dict, graph, cameras):
    segments = []
    suspicious_images = []

    shot_ids = sorted(reconstruction.shots)
    for shot_id in shot_ids:
        next_shot_id = _next_shot_id(shot_id)
        prev_shot_id = _prev_shot_id(shot_id)

        origin_curr = reconstruction.shots[shot_id].pose.get_origin()

        if next_shot_id in reconstruction.shots and _is_no_culling_in_between(culling_dict, shot_id, next_shot_id):
            vector_next = reconstruction.shots[next_shot_id].pose.get_origin() - origin_curr

            # check 1: compare rotations predicted by pre-integration and that of the reconstruction
            geo_diff = diff_recon_preint(shot_id, next_shot_id, reconstruction, pdr_shots_dict)

            if geo_diff > np.radians(HEADING_THRESH): # 10 degrees
                logger.debug("{} {} rotation diff {} degrees".format(shot_id, next_shot_id, np.degrees(geo_diff)))
                suspicious_images.append(shot_id)
                continue

            # check 2: check absolute height of current frame and difference between neighboring frames
            height_diff = abs(vector_next[2])
            if abs(vector_next[2]) > REL_HEIGHT_THRESH or abs(origin_curr[2]) > ABS_HEIGHT_THRESH:
                logger.debug("{} {} height {} diff {} meters".format(shot_id, next_shot_id, origin_curr[2], height_diff))
                suspicious_images.append(shot_id)
                continue

            # check 3: check distance traveled between neighboring frames - we know frames are taken 0.5s apart,
            # and physically the camera should move around 0.5 meters at normal walking speed. but we set the
            # threshold somewhat higher to account for inaccuracies of pdr.
            distance = np.linalg.norm(vector_next)
            if distance > DISTANCE_THRESH:
                logger.debug("{} {} coords distance {} meters".format(shot_id, next_shot_id, distance))
                suspicious_images.append(shot_id)
                continue

            # check 4: we know that frames are 0.5s apart, which is not enough time for a human to turning around
            # at normal walking speed, it's unlikely to have any significant movement that's forward then backward
            if prev_shot_id in reconstruction.shots and _is_no_culling_in_between(culling_dict, shot_id, prev_shot_id):
                vector_prev = origin_curr - reconstruction.shots[prev_shot_id].pose.get_origin()

                angle = _vector_angle(vector_next, vector_prev)
                if angle > np.radians(ANGLE_THRESH):
                    d_next = np.linalg.norm(vector_next)
                    d_prev = np.linalg.norm(vector_prev)

                    if d_next > MOVEMENT_THRESH and d_prev > MOVEMENT_THRESH and \
                            (d_next > SIG_MOVEMENT_THRESH or d_prev > SIG_MOVEMENT_THRESH):
                        logger.debug("{} {} {} angle = {} d_prev = {} d_next {}".
                                     format(prev_shot_id, shot_id, next_shot_id,
                                            np.degrees(angle), d_prev, d_next))
                        suspicious_images.append(shot_id)
                        continue

    # now remove suspicious images and images around them
    remove_images = set()
    for shot_id in suspicious_images:
        index = _shot_id_to_int(shot_id)
        for i in range(index - NEIGHBOR_RADIUS, index + NEIGHBOR_RADIUS + 1):
            remove_images.add(_int_to_shot_id(i))

    # TODO: is it necessary to search for and retriangulate/remove points in removed images?
    for shot_id in sorted(remove_images):
        if shot_id in reconstruction.shots:
            #logger.info("removing image: {}".format(shot_id))
            reconstruction.shots.pop(shot_id, None)

    # find sequential frames longer than MIN_RECON_SIZE to form new recons. the sequence
    # is allowed to have holes no larger than MAX_HOLE_SIZE
    shot_ids = sorted(reconstruction.shots)

    if len(shot_ids) < MIN_RECON_SIZE/MAX_HOLE_SIZE:
        return segments, len(suspicious_images)

    first_index = _shot_id_to_int(shot_ids[0])
    last_index = _shot_id_to_int(shot_ids[-1])

    start_index = end_index = first_index
    while True:
        while True:
            found = False
            for i in range(end_index + MAX_HOLE_SIZE, end_index, -1):
                if _int_to_shot_id(i) in shot_ids:
                    end_index = i
                    found = True
                    break

            if not found:
                break

        if (end_index - start_index + 1) >= MIN_RECON_SIZE:
            logger.debug("extract new recon from {} to {}".format(start_index, end_index))
            segment = extract_segment(reconstruction, start_index, end_index, graph, cameras)
            segments.append(segment)

        found = False
        for i in range(end_index + MAX_HOLE_SIZE, last_index - MIN_RECON_SIZE + 1):
            if _int_to_shot_id(i) in shot_ids:
                start_index = end_index = i
                found = True
                break

        if not found:
            break

    return segments, len(suspicious_images)


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

        ax.text(p[0], p[1], str(_shot_id_to_int(shot.id)), fontsize=10)

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

    check_scale_change_by_pdr(reconstruction, pdr_shots_dict, culling_dict)
    #prune_reconstruction_by_pdr(reconstruction, pdr_shots_dict, culling_dict, graph, cameras)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def debug_plot_reconstructions(reconstructions):
    """
    draw floor plan and aligned pdr shot positions on top of it
    """
    # floor plan
    plan_paths = []
    plan_paths.extend(glob.glob('./*.png'))

    if not plan_paths or not os.path.exists(plan_paths[0]):
        print("No floor plan image found. Quitting")
        return

    img = cv2.imread(plan_paths[0], cv2.IMREAD_COLOR)

    fig, ax = plt.subplots()
    ax.imshow(img)

    topocentric_gps_points_dict = _load_topocentric_gps_points()
    for key, value in topocentric_gps_points_dict.items():
        circle = plt.Circle((value[0], value[1]), color='black', radius=25)
        ax.add_artist(circle)
        ax.text(value[0], value[1], str(_shot_id_to_int(key)), color='white', fontsize=10)

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
                radius = 25
            else:
                radius = 10

            p = shot.pose.get_origin()
            circle = plt.Circle((p[0], p[1]), color=color, radius=radius)
            ax.add_artist(circle)
            ax.text(p[0], p[1], str(_shot_id_to_int(shot.id)), fontsize=10)

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


def debug_rescale_reconstructions(recons):
    """
    rescale recons (which had been aligned)
    :param reconstructions:
    :return:
    """
    all_origins = []
    for recon in recons:
        for shot_id in recon.shots:
            all_origins.append(recon.shots[shot_id].pose.get_origin())

    all_origins = np.asarray(all_origins)
    minx = min(all_origins[:, 0])
    maxx = max(all_origins[:, 0])
    miny = min(all_origins[:, 1])
    maxy = max(all_origins[:, 1])
    meanz = np.mean(all_origins[:, 2])

    s = 100.0/max([maxx-minx, maxy-miny])
    A = np.eye(3)
    b = np.array([
        -(minx+maxx)/2.0*s,
        -(miny+maxy)/2.0*s,
        -meanz*s
    ])

    for recon in recons:
        # Align points.
        for point in recon.points.values():
            p = s * A.dot(point.coordinates) + b
            point.coordinates = p.tolist()

        # Align cameras.
        for shot in recon.shots.values():
            R = shot.pose.get_rotation_matrix()
            t = np.array(shot.pose.translation)
            Rp = R.dot(A.T)
            tp = -Rp.dot(b) + s * t
            try:
                shot.pose.set_rotation_matrix(Rp)
                shot.pose.translation = list(tp)
            except:
                logger.debug("unable to transform reconstruction!")

    os.chdir("/home/cren/source/OpenSfM")
    with io.open_wt('data/rx.json') as fout:
        io.json_dump(io.reconstructions_to_json(recons), fout, False)
    os.system("python3 -m http.server")
    webbrowser.open('http://localhost:8000/viewer/reconstruction.html#file=/data/rx.json', new=2)


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


def _vector_angle(vector_1, vector_2):
    if np.allclose(vector_1, [0, 0, 0]) or np.allclose(vector_2, [0, 0, 0]):
        return 0

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)


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

    try:
        with open("gps_list.txt") as fin:
            gps_points_dict = io.read_gps_points_list(fin)

        with io.open_rt("reference_lla.json") as fin:
            reflla = io.json_load(fin)

        for key, value in gps_points_dict.items():
            x, y, z = geo.topocentric_from_lla(
                value[0], value[1], value[2],
                reflla['latitude'], reflla['longitude'], reflla['altitude'])
            topocentric_gps_points_dict[key] = (x, y, z)
    except os.error:
        return {}

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

    # we produce a few variations of reconstruction output, as follows:
    #   1. after sfm but before alignment
    #       * reconstruction.json (flattened and has roughly same scale as pdr output),
    #       * reconstruction_no_point.json (same as above but with points stripped off, for gps picker)
    #   2. after alignment
    #       * reconstruction.json (aligned with gps points)
    #       * reconstruction.json.bak (saved original sfm)
    #   3. after pipeline executes 'output_aligned_reconstructions'
    #       * aligned_reconstructions.json (same as reconstruction.json)
    show_num = None
    recon_file = None

    # syntax: python3 debug_plot.py [show_num] [recon_file]
    # show_num (None): plot aligned recons on floor plan
    # show_num == -1: run the pruning code
    # show_num == -2: run the rescaling code, then launch browser for viewing
    # show_num == n: n is index of recon to be plotted
    if len(sys.argv) > 1:
        show_num = int(sys.argv[1])

        if len(sys.argv) > 2:
            recon_file = sys.argv[2]

    if recon_file is None:
        if show_num is None:
            recon_file = 'aligned_reconstructions.json'
        elif show_num == -1:
            recon_file = 'reconstruction_no_point.json'
        elif show_num == -2:
            recon_file = 'reconstruction.json.bak'
        else:
            recon_file = 'reconstruction.json'

    with open(recon_file) as fin:
        recons = io.reconstructions_from_json(json.load(fin))
    recons = sorted(recons, key=lambda x: -len(x.shots))

    pdr_shots_dict = {}
    with open('pdr_shots.txt') as fin:
        for line in fin:
            (s_id, x, y, z, roll, pitch, heading, omega_0, omega_1, omega_2, delta_distance) = line.split()
            pdr_shots_dict[s_id] = (float(x), float(y), float(z),
                                    float(roll), float(pitch), float(heading),
                                    float(omega_0), float(omega_1), float(omega_2),
                                    float(delta_distance))

    culling_dict = {}
    if os.path.exists('../frames_redundant/undo.txt'):
        with open('../frames_redundant/undo.txt') as fin:
            for line in fin:
                (new_shot_id, orig_shot_id) = line.split()
                culling_dict[new_shot_id] = orig_shot_id

    if show_num == -1:
        # if show_num is -1, run the pruning code
        with io.open_rt('tracks.csv') as fin:
            graph = tracking.load_tracks_graph(fin)

        with io.open_rt('camera_models.json') as fin:
            obj = json.load(fin)
            cameras = io.cameras_from_json(obj)

        segments = prune_reconstructions_by_pdr(recons, pdr_shots_dict, culling_dict, graph, cameras)
        segments = sorted(segments, key=lambda x: -len(x.shots))

        with io.open_wt('reconstruction.json.new') as fout:
            io.json_dump(io.reconstructions_to_json(segments), fout, False)

    elif show_num == -2:
        # if show_num is -2, run the rescaling code, then launch browser for viewing
        debug_rescale_reconstructions(recons)

    elif show_num is not None:
        # if show_num is not -1 or -2 it is index of recon to be plotted
        debug_plot_reconstruction(recons[show_num], pdr_shots_dict, culling_dict)

    else: #if show_num is None:
        # if there is no argument, plot all recons on floor plan
        debug_plot_reconstructions(recons)
