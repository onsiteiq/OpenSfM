import logging
from itertools import combinations
from collections import defaultdict
import numpy as np

import scipy.spatial as spatial

from opensfm import bow
from opensfm import context


logger = logging.getLogger(__name__)


def has_gps_info(exif):
    return (exif and
            'gps' in exif and
            'latitude' in exif['gps'] and
            'longitude' in exif['gps'])


def calc_pdr_distance(data, image_ref, image_cand):
    if data.pdr_shots_exist():
        pdr_shots_dict = data.load_pdr_shots()
    else:
        return 0

    if image_ref in pdr_shots_dict and image_cand in pdr_shots_dict:
        pos_ref = pdr_shots_dict[image_ref][:3]
        pos_cand = pdr_shots_dict[image_cand][:3]
        distance = np.linalg.norm(np.array(pos_ref) - np.array(pos_cand))

        return distance


def match_candidates_by_distance(images_ref, images_cand, exifs, reference,
                                 max_neighbors, max_distance):
    """Find candidate matching pairs by GPS distance.

    The GPS altitude is ignored because we want images of the same position
    at different altitudes to be matched together.  Otherwise, for drone
    datasets, flights at different altitudes do not get matched.
    """
    if max_neighbors <= 0 and max_distance <= 0:
        return set()
    max_neighbors = max_neighbors or 99999999
    max_distance = max_distance or 99999999.
    k = min(len(images_cand), max_neighbors)

    points = np.zeros((len(images_cand), 3))
    for i, image in enumerate(images_cand):
        gps = exifs[image]['gps']
        points[i] = reference.to_topocentric(
            gps['latitude'], gps['longitude'], 0)

    tree = spatial.cKDTree(points)

    pairs = set()
    for image_ref in images_ref:
        nn = k+1 if image_ref in images_cand else k

        gps = exifs[image_ref]['gps']
        point = reference.to_topocentric(
            gps['latitude'], gps['longitude'], 0)
        distances, neighbors = tree.query(
            point, k=nn, distance_upper_bound=max_distance)

        for j in neighbors:
            if j >= len(images_cand):
                continue
            image_cand = images_cand[j]
            if image_cand != image_ref:
                pairs.add(tuple(sorted((image_ref, image_cand))))
    return pairs


def match_candidates_with_bow(data, images_ref, images_cand,
                              exifs, max_neighbors, order_neighbors,
                              max_pdr_distance, max_index_range,
                              enforce_other_cameras):
    """Find candidate matching pairs using BoW-based distance.

    If enforce_other_cameras is True, we keep max_neighbors images
    with same cameras AND  max_neighbors images from any other different
    camera.
    """
    if max_neighbors <= 0:
        return set()

    # restrict bow searching to 150 index neighbors
    preempted_cand = defaultdict(list)
    n = (max_index_range + 1) // 2
    m = (order_neighbors + 1) // 2

    for i, image_ref in enumerate(images_ref):
        a = max(0, i - n)
        b = min(len(images_cand), i + n)
        c = max(0, i - m)
        d = min(len(images_cand), i + m)
        for j in list(range(a, c)) + list(range(d, b)):
            preempted_cand[image_ref].append(images_cand[j])

    # reduce sets of images from which to load words (RAM saver)
    need_load = set(preempted_cand.keys())
    for v in preempted_cand.values():
        need_load.update(v)

    # construct BoW histograms
    logger.info("Computing %d BoW histograms" % len(need_load))
    histograms = load_histograms(data, need_load)
    args = list(match_bow_arguments(preempted_cand, histograms))

    # parallel BoW neighbors computation
    per_process = 512
    processes = context.processes_that_fit_in_memory(data.config['processes'], per_process)
    batch_size = int(max(1, len(args)/(2*processes)))
    logger.info("Computing BoW candidates with %d processes" % processes)
    results = context.parallel_map(match_bow_unwrap_args, args, processes, batch_size)

    # construct final sets of pairs to match
    pairs = set()
    for im, order, other in results:
        if enforce_other_cameras:
            pairs = pairs.union(pairs_from_neighbors(im, exifs, order, other, max_neighbors))
        else:
            for i in order[:max_neighbors]:
                logger.debug("im={}, i={}, other={}".format(im, i, other[i]))
                dist = calc_pdr_distance(data, im, other[i])
                if dist < max_pdr_distance * 0.3048:
                    pairs.add(tuple(sorted((im, other[i]))))
                    logger.debug("adding pair {} - {}, pdr distance {} feet".format(im, other[i], dist/0.3048))
                else:
                    logger.debug("not adding pair {} - {}, pdr distance {} feet".format(im, other[i], dist/0.3048))
    return pairs


def match_bow_arguments(candidates, histograms):
    """ Generate arguments for parralel processing of BoW """
    for im, cands in candidates.items():
        yield (im, cands, histograms)


def match_bow_unwrap_args(args):
    """ Wrapper for parralel processing of BoW """
    image, other_images, histograms = args
    return bow_distances(image, other_images, histograms)


def match_candidates_by_time(images_ref, images_cand, exifs, max_neighbors):
    """Find candidate matching pairs by time difference."""
    if max_neighbors <= 0:
        return set()
    k = min(len(images_cand), max_neighbors)

    times = np.zeros((len(images_cand), 1))
    for i, image in enumerate(images_cand):
        times[i] = exifs[image]['capture_time']

    tree = spatial.cKDTree(times)

    pairs = set()
    for image_ref in images_ref:
        nn = k+1 if image_ref in images_cand else k

        time = exifs[image_ref]['capture_time']
        distances, neighbors = tree.query([time], k=nn)
        for j in neighbors:
            if j >= len(images_cand):
                continue
            image_cand = images_cand[j]
            if image_ref != image_cand:
                pairs.add(tuple(sorted((image_ref, image_cand))))
    return pairs


def match_candidates_by_order(images_ref, images_cand, max_neighbors):
    """Find candidate matching pairs by sequence order."""
    if max_neighbors <= 0:
        return set()
    n = (max_neighbors + 1) // 2

    pairs = set()
    for i, image_ref in enumerate(images_ref):
        a = max(0, i - n)
        b = min(len(images_cand), i + n)
        for j in range(a, b):
            image_cand = images_cand[j]
            if image_ref != image_cand:
                pairs.add(tuple(sorted([image_ref, image_cand])))
    return pairs


def match_candidates_by_pdr(images_ref, images_cand, pdr_max_distance, data):
    """Find candidate matching pairs by glocally aligned pdr input."""
    if pdr_max_distance <= 0:
        return set()

    if data.pdr_shots_exist():
        max_distance_meters = pdr_max_distance * 0.3048

        # load pdr data and globally align with gps points
        pdr_shots_dict = data.load_pdr_shots()

        pairs = set()
        for image_ref in images_ref:
            for image_cand in images_cand:
                if image_ref != image_cand:
                    if image_ref in pdr_shots_dict and image_cand in pdr_shots_dict:
                        pos_ref = pdr_shots_dict[image_ref][:3]
                        pos_cand = pdr_shots_dict[image_cand][:3]
                        distance = np.linalg.norm(np.array(pos_ref) - np.array(pos_cand))

                        if distance < max_distance_meters:
                            pairs.add(tuple(sorted((image_ref, image_cand))))
        return pairs
    else:
        return set()


def match_candidates_from_metadata(images_ref, images_cand, exifs, data):
    """Compute candidate matching pairs between between images_ref and images_cand

    Returns a list of pairs (im1, im2) such that (im1 in images_ref) is true.
    Returned pairs are unique given that (i, j) == (j, i).
    """
    max_distance = data.config['matching_gps_distance']
    gps_neighbors = data.config['matching_gps_neighbors']
    time_neighbors = data.config['matching_time_neighbors']
    order_neighbors = data.config['matching_order_neighbors']
    pdr_max_distance = data.config['matching_pdr_distance']
    bow_neighbors = data.config['matching_bow_neighbors']
    bow_pdr_distance = data.config['matching_bow_pdr_distance']
    bow_index_range = data.config['matching_bow_index_range']
    bow_other_cameras = data.config['matching_bow_other_cameras']

    if not data.reference_lla_exists():
        data.invent_reference_lla()

    # TODO: cren fix this
    #reference = data.load_reference()
    reference = None

    if not all(map(has_gps_info, exifs.values())):
        if gps_neighbors != 0:
            logger.warn("Not all images have GPS info. "
                        "Disabling matching_gps_neighbors.")
        gps_neighbors = 0
        max_distance = 0

    images_ref.sort()

    if max_distance == gps_neighbors == time_neighbors == order_neighbors == pdr_max_distance == bow_neighbors == 0:
        # All pair selection strategies deactivated so we match all pairs
        d = set()
        t = set()
        o = set()
        p = set()
        b = set()
        pairs = set([tuple(sorted([i, j])) for i in images_ref for j in images_cand])
    else:
        d = match_candidates_by_distance(images_ref, images_cand, exifs, reference,
                                         gps_neighbors, max_distance)
        t = match_candidates_by_time(images_ref, images_cand, exifs, time_neighbors)
        o = match_candidates_by_order(images_ref, images_cand, order_neighbors)
        p = match_candidates_by_pdr(images_ref, images_cand, pdr_max_distance, data)
        b = match_candidates_with_bow(data, images_ref, images_cand,
                                      exifs, bow_neighbors, order_neighbors,
                                      bow_pdr_distance, bow_index_range,
                                      bow_other_cameras)
        pairs = d | t | o | p | b

    pairs = ordered_pairs(pairs, images_ref)

    report = {
        "num_pairs_distance": len(d),
        "num_pairs_time": len(t),
        "num_pairs_order": len(o),
        "num_pairs_pdr": len(p),
        "num_pairs_bow": len(b)
    }
    return pairs, report


def bow_distances(image, other_images, histograms):
    """ Compute BoW-based distance (L1 on histogram of words)
        between an image and other images.
    """
    if image not in histograms:
        return image, [], []

    distances = []
    other = []
    h = histograms[image]
    for im2 in other_images:
        if im2 != image and im2 in histograms:
            h2 = histograms[im2]
            distances.append(np.fabs(h - h2).sum())
            other.append(im2)
    return image, np.argsort(distances), other


def load_histograms(data, images):
    """ Load BoW histograms of given images """
    min_num_feature = 8

    histograms = {}
    bows = bow.load_bows(data.config)
    for im in images:
        words = data.load_words(im)
        if words is None:
            logger.error("Could not load words for image {}".format(im))
            continue

        mask = data.load_masks(data, im) if hasattr(data, 'load_masks') else None
        filtered_words = words[mask] if mask else words
        if len(filtered_words) <= min_num_feature:
            logger.warning("Too few filtered features in image {}: {}".format(
                im, len(filtered_words)))
            continue

        histograms[im] = bows.histogram(words[:, 0])
    return histograms


def pairs_from_neighbors(image, exifs, order, other, max_neighbors):
    """Construct matching pairs given closest ordered neighbors.

    Pairs will of form (image, im2), im2 being the closest max_neighbors
    given by (order, other) having the same cameras OR the closest max_neighbors
    having from any other camera.
    """
    same_camera, other_cameras = [], []
    for i in order:
        im2 = other[i]
        if exifs[im2]['camera'] == exifs[image]['camera']:
            if len(same_camera) < max_neighbors:
                same_camera.append(im2)
        else:
            if len(other_cameras) < max_neighbors:
                other_cameras.append(im2)
        if len(same_camera) + len(other_cameras) >= 2 * max_neighbors:
            break

    pairs = set()
    for im2 in same_camera+other_cameras:
        pairs.add(tuple(sorted((image, im2))))
    return pairs


def ordered_pairs(pairs, images_ref):
    """Image pairs that need matching skipping duplicates.

    Returns a list of pairs (im1, im2) such that (im1 in images_ref) is true.
    """
    per_image = defaultdict(list)
    for im1, im2 in pairs:
        per_image[im1].append(im2)
        per_image[im2].append(im1)

    ordered = set()
    remaining = sorted(set(images_ref))
    if len(remaining) > 0:
        next_image = remaining.pop()
        while next_image:
            im1 = next_image
            next_image = None

            for im2 in sorted(per_image[im1]):
                if (im2, im1) not in ordered:
                    ordered.add((im1, im2))
                    if not next_image and im2 in remaining:
                        next_image = im2
                        remaining.remove(im2)

            if not next_image and remaining:
                next_image = remaining.pop()

    return list(ordered)
