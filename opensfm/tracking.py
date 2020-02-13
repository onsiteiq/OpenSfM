import logging
import sys
import cv2
import math

import numpy as np
import networkx as nx

from collections import defaultdict
from itertools import combinations
from six import iteritems
from opensfm.unionfind import UnionFind


logger = logging.getLogger(__name__)


def load_features(dataset, images):
    logging.info('reading features')
    features = {}
    colors = {}
    for im in images:
        p, f, c = dataset.load_features(im)

        if p is None:
            features[im] = []
            colors[im] = []
            continue

        features[im] = p[:, :3]
        colors[im] = c
    return features, colors


def load_matches(dataset, images):
    matches = {}
    for im1 in images:
        try:
            im1_matches = dataset.load_matches(im1)
        except IOError:
            continue
        for im2 in im1_matches:
            if im2 in images:
                matches[im1, im2] = im1_matches[im2]

    return matches


def load_pairwise_transforms(dataset, images):
    pairs = {}
    for im1 in images:
        try:
            im1_transforms = dataset.load_transforms(im1)
        except IOError:
            continue
        for im2 in im1_transforms:
            if im2 in images:
                pairs[im1, im2] = im1_transforms[im2]

    return pairs


def triplet_filter(data, images, matches, pairs):
    """
    find all triplets and see if they are valid. a voting scheme is used to find
    the bad edges. the voting scheme follows Cui, et al, "Efficient Large-Scale
    Structure From Motion by Fusing Auxiliary Imaging Information"
    :param data:
    :param images:
    :param matches:
    :param pairs:
    :return:
    """
    logger.debug("triplet filtering start")
    cnt_good = {}
    cnt_bad = {}

    for (i, j) in pairs:
        for k in images:
            if i != k and j != k:
                if all_edge_exists([i, j, k], matches):
                    if is_triplet_valid(i, j, k, pairs):
                        cnt = cnt_good
                    else:
                        cnt = cnt_bad

                    incr_cnt(pairs, cnt, i, j)
                    incr_cnt(pairs, cnt, j, k)
                    incr_cnt(pairs, cnt, k, i)

    edges_to_remove = []
    for (i, j) in matches:
        good = 0
        bad = 0
        if (i, j) in cnt_good:
            good = cnt_good[i, j]

        if (i, j) in cnt_bad:
            bad = cnt_bad[i, j]

        # we will not remove any edge with sequence number difference less than 3
        # unless there's strong indication.
        # TODO: cren optionize the threshold below
        if abs(_shot_id_to_int(i) - _shot_id_to_int(j)) < 3:
            if (bad == 0 and good == 0) or (bad/(bad+good)) > 0.9:
                edges_to_remove.append((i, j))
                logger.debug("interesting {} {} good={}, bad={}".format(i, j, good, bad))
        else:
            if (bad == 0 and good == 0) or (bad/(bad+good)) > 0.75:
                edges_to_remove.append((i, j))

    for edge in sorted(edges_to_remove):
        logger.debug("triplet removing edge {} -{}".format(edge[0], edge[1]))
        matches.pop(edge)

    logger.debug("triplet filtering end, removed {} edges, {:2.1f}% of all".
                 format(len(edges_to_remove), 100*len(edges_to_remove)/len(pairs)))
    return matches


def loop_filter(data, images, features, matches, pairs):
    """
    if there’s an edge between (i, j) where i and j are sequence numbers far apart, check that
    there also exists an edge (i plus/minus k, j plus/minus k), where k is a small integer,
    and that the loop formed by the four nodes pass the multiplying-to-identity check. if so,
    this is a valid "quad".

    next, we merge quads into clusters. each cluster is a loop candidate. we check the loop
    candidates to filter out bad ones, and remove all edges in them.

    :param data:
    :param images:
    :param matches:
    :param pairs:
    :return:
    """
    logger.debug("quad filtering start")
    # TODO: cren optionize the following thresholds
    gap = 6
    edges_to_remove = []
    all_valid_quads = []
    for (im1, im2) in matches:
        if abs(_shot_id_to_int(im1) - _shot_id_to_int(im2)) > gap:
            valid_quads = get_valid_quads(im1, im2, matches, pairs)
            if valid_quads:
                all_valid_quads.extend(valid_quads)
            else:
                edges_to_remove.append((im1, im2))

    for edge in sorted(edges_to_remove):
        logger.debug("quad removing edge {} -{}".format(edge[0], edge[1]))
        matches.pop(edge)

    logger.debug("quad filtering end, removed {} edges, {:2.1f}% of all".
                 format(len(edges_to_remove), 100*len(edges_to_remove)/len(pairs)))

    radius = gap/2
    valid_quads_set = set(tuple(quad) for quad in all_valid_quads)

    # cluster quads into loop candidates
    loop_candidates = cluster_quads(valid_quads_set, radius)

    # apply various checks to figure out bad loop candidates
    bad_candidates = filter_candidates(loop_candidates, matches, features, pairs)

    # remove matches in bad loop candidates
    edges_to_remove = set()
    for cand in bad_candidates:
        loop_candidates.remove(cand)
        for im1 in cand.get_ids_0():
            for im2 in cand.get_ids_1():
                if abs(_shot_id_to_int(im1) - _shot_id_to_int(im2)) > radius:
                    if (im1, im2) in matches:
                        edges_to_remove.add((im1, im2))
                    elif (im2, im1) in matches:
                        edges_to_remove.add((im2, im1))

    for edge in sorted(edges_to_remove):
        logger.debug("quad removing edge {} -{}".format(edge[0], edge[1]))
        matches.pop(edge)

    logger.debug("loop filtering end, removed {} edges, {:2.1f}% of all".
                 format(len(edges_to_remove), 100*len(edges_to_remove)/len(pairs)))

    return matches #, loop_candidates


def filter_candidates(loop_candidates, matches, features, pairs):
    bad_candidates = []
    for cand in loop_candidates:
        common_ratios = []

        ns = sorted(cand.get_ids_0())
        ms = sorted(cand.get_ids_1())
        for n1, n2 in zip(ns, ns[1:]):
            ratio_max = 0

            fids_n1_n2 = common_fids(n1, n2, matches)
            if len(fids_n1_n2) < 50:
                continue
            grid_size = math.sqrt(len(fids_n1_n2))
            fo_n1_n2 = feature_occupancy(n1, fids_n1_n2, features, grid_size)
            for m in ms:
                if all_edge_exists([n1, n2, m], matches):
                    fids_n1_m = common_fids(n1, m, matches)
                    fo_n1_m = feature_occupancy(n1, fids_n1_m, features, grid_size)
                    ratio = common_ratio(n1, n2, m, matches) * fo_n1_m / fo_n1_n2
                    if ratio > ratio_max:
                        ratio_max = ratio

            if ratio_max > 0:
                common_ratios.append(ratio_max)

        for m1, m2 in zip(ms, ms[1:]):
            ratio_max = 0

            fids_m1_m2 = common_fids(m1, m2, matches)
            if len(fids_m1_m2) < 50:
                continue
            grid_size = math.sqrt(len(fids_m1_m2))
            fo_m1_m2 = feature_occupancy(m1, fids_m1_m2, features, grid_size)
            for n in ns:
                if all_edge_exists([m1, m2, n], matches):
                    fids_m1_n = common_fids(m1, n, matches)
                    fo_m1_n = feature_occupancy(m1, fids_m1_n, features, grid_size)
                    ratio = common_ratio(m1, m2, n, matches) * fo_m1_n / fo_m1_m2
                    if ratio > ratio_max:
                        ratio_max = ratio

            if ratio_max > 0:
                common_ratios.append(ratio_max)

        avg_ratio = 0
        if common_ratios:
            avg_ratio = sum(common_ratios) / len(common_ratios)

        logger.debug("loop candidate center {:4.1f}-{:4.1f}, "
                     "average overlap {}, "
                     "members {} - {}".format(
            cand.get_center_0(), cand.get_center_1(), avg_ratio,
            sorted(cand.get_ids_0()), sorted(cand.get_ids_1())))

        # TODO - cren optionize the ratio threshold below
        if avg_ratio < 0.15:
            bad_candidates.append(cand)
        else:
            chaining_test(cand, matches, pairs)

    return bad_candidates


def chaining_test(loop_candidate, matches, pairs):
    '''
    find a loop that connects any image in ids_0 group to ids_1 group, and test if
    the loop is valid
    :param loop_candidate:
    :param matches:
    :param pairs:
    :return:
    '''
    ids_0 = sorted(loop_candidate.get_ids_0())
    ids_1 = sorted(loop_candidate.get_ids_1(), reverse=True)
    logger.debug("ids_0={}".format(ids_0))
    logger.debug("ids_1={}".format(ids_1))
    for start_id in ids_0:
        logger.debug("start_id={}".format(start_id))
        for end_id in ids_1:
            logger.debug("end_id={}".format(end_id))
            if (start_id, end_id) in matches or (end_id, start_id) in matches:
                path = [start_id]
                curr_id = next_id = start_id
                while _shot_id_to_int(next_id) < _shot_id_to_int(end_id):
                    next_id = _next_shot_id(next_id)
                    if (curr_id, next_id) in matches or (next_id, curr_id) in matches:
                        path.append(next_id)
                        curr_id = next_id

                logger.debug("path={}".format(path))
                if path[-1] == end_id:
                    if is_loop_valid(path, pairs):
                        return True

    return False


def common_fids(im1, im2, matches):
    fids = []
    if (im1, im2) in matches:
        for f1, f2 in matches[im1, im2]:
            fids.append(f1)
    elif (im2, im1) in matches:
        for f1, f2 in matches[im2, im1]:
            fids.append(f2)

    return fids


def feature_occupancy(im, fids, features, grid_size):
    occupied = set()
    for id in fids:
        x, y, s = features[im][id]
        occupied.add((int(x*grid_size), int(y*grid_size)))

    return len(occupied)


def common_ratio(n1, n2, m, matches):
    """
    calculates the ratio of # of common features of the triplet (n1, n2, m) to
    # of common features of the pair (n1, n2). the larger the ratio the more
    likely m is correctly related to n1, n2.
    :param n1:
    :param n2:
    :param m:
    :param matches:
    :return:
    """
    uf = UnionFind()

    if (n1, n2) in matches:
        base_cnt = len(matches[n1, n2])
        for f1, f2 in matches[n1, n2]:
            uf.union((n1, f1), (n2, f2))
    else:
        base_cnt = len(matches[n2, n1])
        for f1, f2 in matches[n2, n1]:
            uf.union((n2, f1), (n1, f2))

    if (n1, m) in matches:
        for f1, f2 in matches[n1, m]:
            uf.union((n1, f1), (m, f2))
    else:
        for f1, f2 in matches[m, n1]:
            uf.union((m, f1), (n1, f2))

    if (n2, m) in matches:
        for f1, f2 in matches[n2, m]:
            uf.union((n2, f1), (m, f2))
    else:
        for f1, f2 in matches[m, n2]:
            uf.union((m, f1), (n2, f2))

    sets = {}
    for i in uf:
        p = uf[i]
        if p in sets:
            sets[p].append(i)
        else:
            sets[p] = [i]

    tracks = [t for t in sets.values() if _good_track(t, 3)]

    cnt = 0
    if (n1, n2) in matches:
        for f1, f2 in matches[n1, n2]:
            for track in tracks:
                if (n1, f1) in track and (n2, f2) in track:
                    cnt += 1
                    break
    else:
        for f1, f2 in matches[n2, n1]:
            for track in tracks:
                if (n2, f1) in track and (n1, f2) in track:
                    cnt += 1
                    break

    return cnt/base_cnt


def cluster_quads(valid_quads, radius):
    """
    merge similar quads into loop candidates

    :param valid_quads:
    :param radius:
    :return:
    """
    loop_candidates = []
    for quad in sorted(valid_quads):
        added = False
        for cand in loop_candidates:
            if cand.is_close_to(quad):
                cand.add(quad)
                added = True
                #break

        if not added:
            new_cand = LoopCandidate(radius)
            new_cand.add(quad)
            loop_candidates.append(new_cand)

    # merge loop candidates that are close together
    while True:
        can_merge = False
        for cand1, cand2 in combinations(loop_candidates, 2):
            if cand1.combine(cand2):
                loop_candidates.remove(cand2)
                can_merge = True
                break

        if not can_merge:
            break

    # if the centers are close together, this is really not a 'loop' but a line
    remove_candidates = []
    for cand in loop_candidates:
        if cand.get_center_1() - cand.get_center_0() < radius:
            remove_candidates.append(cand)

    for cand in remove_candidates:
        loop_candidates.remove(cand)

    return loop_candidates


'''
def quad_to_coords(quad):
    im_indices = []
    for im in quad:
        im_indices.append(_shot_id_to_int(im))

    coords = []
    coords.append((im_indices[0], im_indices[2]))
    coords.append((im_indices[0], im_indices[3]))
    coords.append((im_indices[1], im_indices[2]))
    coords.append((im_indices[1], im_indices[3]))

    return coords

def cluster_quads_kmeans(valid_quads, radius):
    coords_set = set()
    for quad in valid_quads:
        coords = quad_to_coords(quad)
        for coord in coords:
            coords_set.add(coord)

    np_coords = np.asarray(list(coords_set), dtype=np.float32)

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    ret, label, center = cv2.kmeans(np_coords, 50, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)

    # Now separate the data, Note the flatten()
    for i in range(100):
        A = np_coords[label.ravel() == i]
        logger.debug("{} -- {}".format(sorted(set(A[:, 0])), sorted(set(A[:, 1]))))
'''


class LoopCandidate(object):
    """
    Loop candidate
    """

    def __init__(self, r):
        self.radius = r
        self.center_0 = -1
        self.center_1 = -1
        self.ids_0 = set()
        self.ids_1 = set()

    def add(self, quad):
        self.ids_0.add(quad[0])
        self.ids_0.add(quad[1])
        self.ids_1.add(quad[2])
        self.ids_1.add(quad[3])

        # update loop center
        self.center_0 = self.get_average(self.ids_0)
        self.center_1 = self.get_average(self.ids_1)

    def get_average(self, ids):
        total = 0
        for id in ids:
            total += _shot_id_to_int(id)
        return total/len(ids)

    def is_close_to(self, quad):
        return abs(self.center_0 - _shot_id_to_int(quad[0])) < self.radius and \
               abs(self.center_1 - _shot_id_to_int(quad[2])) < self.radius

    def combine(self, another):
        if abs(self.get_center_0() - another.get_center_0()) < self.radius and \
           abs(self.get_center_1() - another.get_center_1()) < self.radius:
            self.ids_0 = self.ids_0 | another.get_ids_0()
            self.ids_1 = self.ids_1 | another.get_ids_1()

            # update loop center
            self.center_0 = self.get_average(self.ids_0)
            self.center_1 = self.get_average(self.ids_1)
            return True
        else:
            return False

    def get_center_0(self):
        return self.center_0

    def get_center_1(self):
        return self.center_1

    def get_ids_0(self):
        return self.ids_0

    def get_ids_1(self):
        return self.ids_1


def get_valid_quads(im1, im2, matches, pairs):
    k = 3
    quads = []

    ind_im1 = _shot_id_to_int(im1)
    ind_im2 = _shot_id_to_int(im2)

    for i in range(ind_im1-k, ind_im1+k+1):
        if i == ind_im1:
            continue
        for j in range(ind_im2 - k, ind_im2 + k + 1):
            if j == ind_im2:
                continue

            im1_neighbor = _int_to_shot_id(i)
            im2_neighbor = _int_to_shot_id(j)

            if all_edge_exists([im1, im1_neighbor, im2, im2_neighbor], matches):
                if is_triplet_valid(im1, im1_neighbor, im2, pairs) and \
                   is_triplet_valid(im2, im2_neighbor, im1_neighbor, pairs) and \
                   is_triplet_valid(im1, im1_neighbor, im2_neighbor, pairs) and \
                   is_triplet_valid(im2, im2_neighbor, im1, pairs):
                    quads.append(sorted((im1, im1_neighbor, im2, im2_neighbor)))
            '''
            if all_edge_exists([im1, im1_neighbor, im2], matches) and \
               all_edge_exists([im1_neighbor, im2_neighbor, im2], matches):
                quads.append(sorted([im1, im1_neighbor, im2, im2_neighbor]))
            '''

    return quads


def incr_cnt(pairs, cnt, i, j):
    if (i, j) in pairs:
        if (i, j) in cnt:
            cnt[i, j] = cnt[i, j] + 1
        else:
            cnt[i, j] = 1
    else:
        if (j, i) in cnt:
            cnt[j, i] = cnt[j, i] + 1
        else:
            cnt[j, i] = 1


def get_transform(i, j, pairs):
    R = np.array([])
    if (i, j) in pairs:
        R = pairs[i, j][:, :3]
    elif (j, i) in pairs:
        R = pairs[j, i][:, :3].T

    return R


def edge_exists(im1, im2, matches):
    return (im1, im2) in matches or (im2, im1) in matches


def all_edge_exists(node_list, matches):
    for im1, im2 in combinations(node_list, 2):
        if not edge_exists(im1, im2, matches):
            return False

    return True


def is_triplet_valid(i, j, k, pairs):
    Rji = get_transform(i, j, pairs)
    Rkj = get_transform(j, k, pairs)
    Rik = get_transform(k, i, pairs)

    # TODO - cren optionize the degree threshold below
    if np.linalg.norm(cv2.Rodrigues(Rik.dot(Rkj.dot(Rji)))[0].ravel()) < math.pi/12:
        return True
    else:
        return False


def is_loop_valid(images, pairs):
    # for a triplet, the threshold is math.pi/12. for every additional image in the
    # loop we allow on average an additional 1 degree error
    # TODO - cren optionize the degree threshold below
    thresh = math.pi/12 + (len(images) - 3) * math.pi/180

    R = np.identity(3, dtype=float)
    for n1, n2 in zip(images, images[1:]):
        R = get_transform(n1, n2, pairs).dot(R)

    R = get_transform(images[-1], images[0], pairs).dot(R)

    if np.linalg.norm(cv2.Rodrigues(R)[0].ravel()) < thresh:
        logger.debug("valid={} thresh={}".format(np.linalg.norm(cv2.Rodrigues(R)[0].ravel()), thresh))
        return True
    else:
        logger.debug("invalid={} thresh={}".format(np.linalg.norm(cv2.Rodrigues(R)[0].ravel()), thresh))
        return False


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


def create_tracks_graph(features, colors, matches, config):
    """Link matches into tracks."""
    logger.debug('Merging features onto tracks')
    uf = UnionFind()
    for im1, im2 in matches:
        for f1, f2 in matches[im1, im2]:
            uf.union((im1, f1), (im2, f2))

    sets = {}
    for i in uf:
        p = uf[i]
        if p in sets:
            sets[p].append(i)
        else:
            sets[p] = [i]

    min_length = config['min_track_length']
    tracks = [t for t in sets.values() if _good_track(t, min_length)]
    logger.debug('Good tracks: {}'.format(len(tracks)))

    tracks_graph = nx.Graph()
    for track_id, track in enumerate(tracks):
        for image, featureid in track:
            if image not in features:
                continue
            x, y, s = features[image][featureid]
            r, g, b = colors[image][featureid]
            tracks_graph.add_node(str(image), bipartite=0)
            tracks_graph.add_node(str(track_id), bipartite=1)
            tracks_graph.add_edge(str(image),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(featureid),
                                  feature_color=(float(r), float(g), float(b)))

    return tracks_graph


def tracks_and_images(graph):
    """List of tracks and images in the graph."""
    tracks, images = [], []
    for n in graph.nodes(data=True):
        if n[1]['bipartite'] == 0:
            images.append(n[0])
        else:
            tracks.append(n[0])
    return tracks, images


def common_tracks(graph, im1, im2):
    """List of tracks observed in both images.

    Args:
        graph: tracks graph
        im1: name of the first image
        im2: name of the second image

    Returns:
        tuple: tracks, feature from first image, feature from second image
    """
    t1, t2 = graph[im1], graph[im2]
    tracks, p1, p2 = [], [], []
    for track in t1:
        if track in t2:
            p1.append(t1[track]['feature'])
            p2.append(t2[track]['feature'])
            tracks.append(track)
    p1 = np.array(p1)
    p2 = np.array(p2)
    return tracks, p1, p2


def all_common_tracks(graph, tracks, include_features=True, min_common=50):
    """List of tracks observed by each image pair.

    Args:
        graph: tracks graph
        tracks: list of track identifiers
        include_features: whether to include the features from the images
        min_common: the minimum number of tracks the two images need to have
            in common

    Returns:
        tuple: im1, im2 -> tuple: tracks, features from first image, features
        from second image
    """
    track_dict = defaultdict(list)
    for track in tracks:
        track_images = sorted(graph[track].keys())
        for im1, im2 in combinations(track_images, 2):
            track_dict[im1, im2].append(track)

    common_tracks = {}
    for k, v in iteritems(track_dict):
        if len(v) < min_common:
            continue
        im1, im2 = k
        if include_features:
            p1 = np.array([graph[im1][tr]['feature'] for tr in v])
            p2 = np.array([graph[im2][tr]['feature'] for tr in v])
            common_tracks[im1, im2] = (v, p1, p2)
        else:
            common_tracks[im1, im2] = v
    return common_tracks


def _good_track(track, min_length):
    if len(track) < min_length:
        return False
    images = [f[0] for f in track]
    if len(images) != len(set(images)):
        return False
    return True


TRACKS_VERSION = 1
TRACKS_HEADER = u'OPENSFM_TRACKS_VERSION'


def load_tracks_graph(fileobj):
    """ Load a tracks graph from file object """
    version = _tracks_file_version(fileobj)
    return getattr(sys.modules[__name__], '_load_tracks_graph_v%d' % version)(fileobj)


def save_tracks_graph(fileobj, graph):
    """ Save a tracks graph to some file object """
    fileobj.write((TRACKS_HEADER + u'_v%d\n') % TRACKS_VERSION)
    getattr(sys.modules[__name__], '_save_tracks_graph_v%d' % TRACKS_VERSION)(fileobj, graph)


def _tracks_file_version(fileobj):
    """ Extract tracks file version by reading header.

    Return 0 version if no vrsion/header was red
    """
    current_position = fileobj.tell()
    line = fileobj.readline()
    if line.startswith(TRACKS_HEADER):
        version = int(line.split('_v')[1])
    else:
        fileobj.seek(current_position)
        version = 0
    return version


def _load_tracks_graph_v0(fileobj):
    """ Tracks graph file base version reading

    Uses some default scale for compliancy
    """
    default_scale = 0.004  # old default reprojection_sd config
    g = nx.Graph()
    for line in fileobj:
        image, track, observation, x, y, R, G, B = line.split('\t')
        g.add_node(image, bipartite=0)
        g.add_node(track, bipartite=1)
        g.add_edge(
            image, track,
            feature=(float(x), float(y)),
            feature_scale=float(default_scale),
            feature_id=int(observation),
            feature_color=(float(R), float(G), float(B)))
    return g


def _save_tracks_graph_v0(fileobj, graph):
    """ Tracks graph file base version saving """
    for node, data in graph.nodes(data=True):
        if data['bipartite'] == 0:
            image = node
            for track, data in graph[image].items():
                x, y = data['feature']
                fid = data['feature_id']
                r, g, b = data['feature_color']
                fileobj.write(u'%s\t%s\t%d\t%g\t%g\t%g\t%g\t%g\n' % (
                    str(image), str(track), fid, x, y, r, g, b))


def _load_tracks_graph_v1(fileobj):
    """ Version 1 of tracks graph file loading

    Feature scale was added
    """
    g = nx.Graph()
    for line in fileobj:
        image, track, observation, x, y, scale, R, G, B = line.split('\t')
        g.add_node(image, bipartite=0)
        g.add_node(track, bipartite=1)
        g.add_edge(
            image, track,
            feature=(float(x), float(y)),
            feature_scale=float(scale),
            feature_id=int(observation),
            feature_color=(float(R), float(G), float(B)))
    return g


def _save_tracks_graph_v1(fileobj, graph):
    """ Version 1 of tracks graph file saving

    Feature scale was added
    """
    for node, data in graph.nodes(data=True):
        if data['bipartite'] == 0:
            image = node
            for track, data in graph[image].items():
                x, y = data['feature']
                s = data['feature_scale']
                fid = data['feature_id']
                r, g, b = data['feature_color']
                fileobj.write(u'%s\t%s\t%d\t%g\t%g\t%g\t%g\t%g\t%g\n' % (
                    str(image), str(track), fid, x, y, s, r, g, b))
