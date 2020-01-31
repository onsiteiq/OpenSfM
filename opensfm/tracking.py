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
                Rij = get_transform(i, j, pairs)
                Rjk = get_transform(j, k, pairs)
                Rki = get_transform(k, i, pairs)

                if Rij.size == 0 or Rjk.size == 0 or Rki.size == 0:
                    continue

                if is_triplet_valid(Rij, Rjk, Rki):
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
        # TODO: cren optionize the gap threshold below
        gap = abs(_shot_id_to_int(i) - _shot_id_to_int(j))
        if gap < 3:
            if (bad + good) >= 3 and (bad / (bad + good)) > 0.75:
                edges_to_remove.append((i, j))
                logger.debug("interesting {} {} gap={}, good={}, bad={}".format(i, j, gap, good, bad))
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
    and that the loop formed by the four nodes pass the multiplying-to-identity check.
    :param data:
    :param images:
    :param matches:
    :param pairs:
    :return:
    """
    logger.debug("quad filtering start")
    # TODO: cren optionize the following thresholds
    gap = 10
    edges_to_remove = []
    valid_quads = []
    for (im1, im2) in matches:
        if abs(_shot_id_to_int(im1) - _shot_id_to_int(im2)) > gap:
            valid = False
            possible_quads = get_quads(im1, im2, matches)
            for (i, j, k, l) in possible_quads:
                Rij = get_transform(i, j, pairs)
                Rjk = get_transform(j, k, pairs)
                Rkl = get_transform(k, l, pairs)
                Rli = get_transform(l, i, pairs)
                if is_quad_valid(Rij, Rjk, Rkl, Rli):
                    valid_quads.append((i, j, k, l))
                    valid = True

            if not valid:
                edges_to_remove.append((im1, im2))

    for edge in sorted(edges_to_remove):
        logger.debug("quad removing edge {} -{}".format(edge[0], edge[1]))
        matches.pop(edge)

    logger.debug("quad filtering end, removed {} edges, {:2.1f}% of all".
                 format(len(edges_to_remove), 100*len(edges_to_remove)/len(pairs)))

    loop_candidates = merge_quads(valid_quads)

    for cand in sorted(loop_candidates, key=lambda cand: cand.get_center_0()):
        common_ratios_0 = []
        common_ratios_1 = []
        weights_0 = []
        weights_1 = []

        ns = list(cand.get_ids_0())
        ms = list(cand.get_ids_1())
        for n1, n2 in zip(ns, ns[1:]):
            ratio_max = 0
            weight = 0
            for m in ms:
                if all_edge_exists([n1, n2, m], matches):
                    ratio, w = get_common_ratio(n1, n2, m, features, matches)
                    if ratio > ratio_max:
                        ratio_max = ratio
                        weight = w

            if ratio_max > 0:
                common_ratios_0.append(ratio_max)
                weights_0.append(weight)

        for m1, m2 in zip(ms, ms[1:]):
            ratio_max = 0
            for n in ns:
                if all_edge_exists([m1, m2, n], matches):
                    ratio, weight = get_common_ratio(m1, m2, n, features, matches)
                    if ratio > ratio_max:
                        ratio_max = ratio
                        weight = w

            if ratio_max > 0:
                common_ratios_1.append(ratio_max)
                weights_1.append(weight)

        avg_0 = avg_1 = 0
        w0 = w1 = 0
        if common_ratios_0:
            avg_0 = sum(common_ratios_0) / len(common_ratios_0)
            w0 = sum(weights_0) / len(weights_0)

        if common_ratios_1:
            avg_1 = sum(common_ratios_1) / len(common_ratios_1)
            w1 = sum(weights_1) / len(weights_1)

        logger.debug("loop candidate center {:4.1f}-{:4.1f}, "
                     "average overlap {}, {}, "
                     "weight {}, {}, "
                     "members {} - {}".format(
            cand.get_center_0(), cand.get_center_1(),
            avg_0, avg_1, w0, w1,
            sorted(cand.get_ids_0()), sorted(cand.get_ids_1())))

    logger.debug("quad filtering end")
    return matches


def calc_feature_distribution(im, fids, features, grid_size):
    occupied = set()

    for id in fids:
        x, y, s = features[im][id]
        occupied.add((int(x*grid_size), int(y*grid_size)))

    return len(occupied)


def get_common_ratio(n1, n2, m, features, matches):
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
    fids_n1_n2 = []

    if (n1, n2) in matches:
        base_cnt = len(matches[n1, n2])
        for f1, f2 in matches[n1, n2]:
            uf.union((n1, f1), (n2, f2))
            fids_n1_n2.append(f1)
    else:
        base_cnt = len(matches[n2, n1])
        for f1, f2 in matches[n2, n1]:
            uf.union((n2, f1), (n1, f2))
            fids_n1_n2.append(f2)

    if base_cnt < 50:
        return 0, 0

    fids_n1_m = []
    if (n1, m) in matches:
        for f1, f2 in matches[n1, m]:
            uf.union((n1, f1), (m, f2))
            fids_n1_m.append(f1)
    else:
        for f1, f2 in matches[m, n1]:
            uf.union((m, f1), (n1, f2))
            fids_n1_m.append(f2)

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

    grid_size = math.sqrt(len(fids_n1_n2))
    o_n1n2 = calc_feature_distribution(n1, fids_n1_n2, features, grid_size)
    o_n1m = calc_feature_distribution(n1, fids_n1_m, features, grid_size)
    weight = o_n1m / o_n1n2
    #logger.debug("dist n1n2={} n1m={}, weight={}".format(dist_n1_n2, dist_n1_m, weight))

    return cnt/base_cnt, weight


def merge_quads(valid_quads):
    """
    merge similar quads into loop candidates

    :param valid_quads:
    :return:
    """
    loop_candidates = []
    for quad in valid_quads:
        added = False
        for cand in loop_candidates:
            if cand.is_close_to(quad):
                cand.add(quad)
                added = True
                break

        if not added:
            # TODO: cren optionize the threshold below
            new_cand = LoopCandidate(10)
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

    return loop_candidates


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


def get_quads(im1, im2, matches):
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
                quads.append(sorted((im1, im1_neighbor, im2, im2_neighbor)))

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


def is_triplet_valid(R0, R1, R2):
    # TODO - cren optionize the degree threshold below
    if np.linalg.norm(cv2.Rodrigues(R2.dot(R1.dot(R0)))[0].ravel()) < math.pi/6:
        return True
    else:
        return False


def is_quad_valid(R0, R1, R2, R3):
    # TODO - cren optionize the degree threshold below
    if np.linalg.norm(cv2.Rodrigues(R3.dot(R2.dot(R1.dot(R0))))[0].ravel()) < math.pi/6:
        return True
    else:
        return False


def triplet_filter_2(matches, pairs):
    """
    find triplets of the form (i,j,j+k), where i, j, j+k are image sequence numbers, and
    difference between i and j is larger than some threshold, and k is some small integer.
    we will check if the pairwise rotations satisfy triplet constraint. if not, the edge
    (i,j) is removed. if we find an isolated (i,j), with no edges between (i,j+k) then we
    will remove the edge (i,j) since it is likely an erroneous epiploar geometry
    :param matches:
    :return:
    """
    #TODO: cren optionize the following parameters
    thresh = 20
    k = 3

    filtered_matches = []
    for im1, im1_matches in matches:
        im2s = sorted(im1_matches.keys())

        if len(im2s) == 0:
            continue

        im2s_to_remove = set()
        last_result_valid = False
        for j in range(len(im2s)-1):
            if _shot_id_to_int(im2s[j]) - _shot_id_to_int(im1) > thresh:
                if _shot_id_to_int(im2s[j+1]) - _shot_id_to_int(im2s[j]) <= k:
                    logger.debug("triplet {} {} {}".format(im1, im2s[j], im2s[j+1]))

                    if (im2s[j], im2s[j+1]) in pairs:
                        T0 = pairs[im1, im2s[j]][0]
                        T1 = pairs[im1, im2s[j+1]][0]
                        T2 = pairs[im2s[j], im2s[j+1]][0]

                        if is_triplet_valid_2(T0, T1, T2):
                            last_result_valid = True
                            continue

                if not last_result_valid:
                    im2s_to_remove.add(im2s[j])

                last_result_valid = False

        # handle boundary condition
        if not last_result_valid and _shot_id_to_int(im2s[-1]) - _shot_id_to_int(im1) > thresh:
            im2s_to_remove.add(im2s[-1])

        logger.debug("triplet edges of {} removing {}".format(im1, sorted(im2s_to_remove)))
        for im2 in im2s_to_remove:
            im1_matches.pop(im2)

        filtered_matches.append((im1, im1_matches))

    return filtered_matches


def is_triplet_valid_2(T0, T1, T2):
    r0 = T0[:, :3]
    r1 = T1[:, :3]
    r2 = T2[:, :3]

    # TODO - cren optionize the degree threshold below
    if np.linalg.norm(cv2.Rodrigues(r2.dot(r1.T.dot(r0)))[0].ravel()) < math.pi/12:
        return True
    else:
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
