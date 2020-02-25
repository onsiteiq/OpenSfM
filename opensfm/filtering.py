import logging
import cv2
import math
import random

import numpy as np

from collections import defaultdict
from itertools import combinations
from opensfm.unionfind import UnionFind


logger = logging.getLogger(__name__)


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

    gap = 6
    for i in range(len(images)):
        for j in range(i+1, i+1+gap):
            for k in range(len(images)):
                if i != k and j != k:
                    im1 = _int_to_shot_id(i)
                    im2 = _int_to_shot_id(j)
                    im3 = _int_to_shot_id(k)
                    if edge_exists_all([im1, im2, im3], matches):
                        if is_triplet_valid([im1, im2, im3], pairs):
                            cnt = cnt_good
                        else:
                            cnt = cnt_bad

                        #incr_cnt(pairs, cnt, im1, im2)
                        incr_cnt(pairs, cnt, im2, im3)
                        incr_cnt(pairs, cnt, im3, im1)

    edges_to_remove = []
    for (im1, im2) in matches:
        good = 0
        bad = 0
        if (im1, im2) in cnt_good:
            good = cnt_good[im1, im2]

        if (im1, im2) in cnt_bad:
            bad = cnt_bad[im1, im2]

        # we will not remove any edge with small sequence number difference unless there's
        # stronger evidence.
        if abs(_shot_id_to_int(im1) - _shot_id_to_int(im2)) < gap:
            if bad+good > 3 and (bad/(bad+good)) > data.config['filtering_triplet_bad_ratio']:
                #logger.debug("removing close edge {} {} good={}, bad={}".format(im1, im2, good, bad))
                edges_to_remove.append((im1, im2))
        else:
            if bad+good == 0 or (bad/(bad+good)) > data.config['filtering_triplet_bad_ratio']:
                #logger.debug("removing {} {} good={}, bad={}".format(im1, im2, good, bad))
                edges_to_remove.append((im1, im2))

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

    we then merge quads into clusters. each cluster is a loop candidate. we perform checks on
    the candidates to filter out bad ones, and remove all edges in them.

    :param data:
    :param images:
    :param matches:
    :param pairs:
    :return:
    """
    logger.debug("loop pass 1 filtering start")

    common_feature_thresh = data.config['filtering_common_feature_thresh']

    # TODO: cren optionize the following thresholds
    gap = 6
    edges_to_remove = []
    all_valid_triplets = []
    for (im1, im2) in matches:
        if abs(_shot_id_to_int(im1) - _shot_id_to_int(im2)) > gap:
            valid_triplets = get_valid_triplets(im1, im2, matches, pairs)
            if valid_triplets:
                all_valid_triplets.extend(valid_triplets)
            else:
                edges_to_remove.append((im1, im2))

    for edge in sorted(edges_to_remove):
        logger.debug("loop pass 1 removing edge {} -{}".format(edge[0], edge[1]))
        matches.pop(edge)

    logger.debug("loop pass 1 filtering end, removed {} edges, {:2.1f}% of all".
                 format(len(edges_to_remove), 100*len(edges_to_remove)/len(pairs)))

    logger.debug("loop pass 2 filtering start")
    radius = gap/2
    valid_triplets_set = set(tuple(triplet) for triplet in all_valid_triplets)

    # cluster quads into loop candidates
    loop_candidates = cluster_triplets(valid_triplets_set, radius)

    # apply various checks to figure out bad loop candidates
    bad_candidates = filter_candidates(images, loop_candidates, matches, features, pairs, common_feature_thresh)

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
        #logger.debug("loop removing edge {} -{}".format(edge[0], edge[1]))
        matches.pop(edge)

    logger.debug("loop pass 2 filtering end, removed {} edges, {:2.1f}% of all".
                 format(len(edges_to_remove), 100*len(edges_to_remove)/len(pairs)))

    return matches #, loop_candidates


def filter_candidates(images, loop_candidates, matches, features, pairs, common_feature_thresh):
    """
    two types of filtering are performed:
    1. based on rotation
    2. based on common features
    :param images:
    :param loop_candidates:
    :param matches:
    :param features:
    :param pairs:
    :param common_feature_thresh:
    :return:
    """
    path_finder = PathFinder(images, matches, pairs)

    bad_candidates = []
    for cand in loop_candidates:
        if validate_loop_rotations(cand, matches, pairs, path_finder) == 'badloop':
            bad_candidates.append(cand)
            logger.debug("invalid loop: only bad loops found")
        elif validate_loop_features(cand, matches, features) < common_feature_thresh:
            bad_candidates.append(cand)
            logger.debug("invalid loop: missing features")

    return bad_candidates


def validate_loop_features(loop_candidate, matches, features):
    '''
    take two images n1, n2 from one side of the loop candidate, and one image m from the other side. the two images
    n1 and n2 presumably have a valid match. we compare how much features n1/n2/m have in common vs n1/n2 have in
    common. if this is a true loop, then the ratio of common features should be large. conversely, if a significant
    portion of features is missing, then this is likely to be a false loop. we also take feature distribution into
    consideration - if this is a false loop, then common features n1/n2 should be more evenly distributed in the
    image than the n1/m common features.

    this algorithm is inspired by Zach "What Can Missing Correspondences Tell Us About 3D Structure and Motion",
    although the Bayesian formulation considered there is not really necessary in our case

    :param loop_candidate:
    :param matches:
    :param features:
    :return:
    '''
    common_ratios = []

    ns = sorted(loop_candidate.get_ids_0())
    ms = sorted(loop_candidate.get_ids_1())
    for n1, n2 in zip(ns, ns[1:]):
        ratio_max = 0

        fids_n1_n2 = common_fids(n1, n2, matches)
        if len(fids_n1_n2) < 50:
            continue
        grid_size = math.sqrt(len(fids_n1_n2))
        fo_n1_n2 = feature_occupancy(n1, fids_n1_n2, features, grid_size)
        for m in ms:
            if edge_exists_all([n1, n2, m], matches):
                ratio, fids_n1_n2_m = common_ratio(n1, n2, m, matches)

                fo_n1_n2_m = feature_occupancy(n1, fids_n1_n2_m, features, grid_size)
                feature_distribution_ratio = fo_n1_n2_m/fo_n1_n2

                ratio = ratio * feature_distribution_ratio

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
            if edge_exists_all([m1, m2, n], matches):
                ratio, fids_m1_m2_n = common_ratio(m1, m2, n, matches)

                fo_m1_m2_n = feature_occupancy(m1, fids_m1_m2_n, features, grid_size)
                feature_distribution_ratio = fo_m1_m2_n/fo_m1_m2

                ratio = ratio * feature_distribution_ratio

                if ratio > ratio_max:
                    ratio_max = ratio

        if ratio_max > 0:
            common_ratios.append(ratio_max)

    avg_common_ratio = 0
    if common_ratios:
        avg_common_ratio = sum(common_ratios) / len(common_ratios)

    logger.debug("average overlap {}".format(avg_common_ratio))

    return avg_common_ratio


def validate_loop_rotations(loop_candidate, matches, pairs, path_finder):
    """
    this method returns:
        'goodloop' if a valid loop has been found
        'badloop' if all we found are invalid loops
        'noloop' if there is no loop found (different from bad loop - the loop candidate may still be valid)
    :param loop_candidate:
    :param matches:
    :param pairs:
    :param path_finder:
    :return:
    """
    ret_val = 'noloop'

    center_0 = int(loop_candidate.get_center_0())
    center_1 = int(loop_candidate.get_center_1())

    ids_0 = sorted(loop_candidate.get_ids_0())
    ids_1 = sorted(loop_candidate.get_ids_1(), reverse=True)

    logger.debug("loop candidate center {:4.1f}-{:4.1f}, "
                 "members {} - {}".format(center_0, center_1, ids_0, ids_1))

    # we make a weak assumption that our path is generally free of cycles, e.g., the path wouldn't loop
    # inside an apartment for more than once. this translates into an additional constraint in camera
    # orientations. that is, if two images are far apart in index, and their relative rotation is close
    # to zero, then their match is regarded as highly suspicious and thus rejected as bad loop.
    if center_1 - center_0 > 10:
        rs = []
        for i in range(-1, 2):
            n1 = _int_to_shot_id(center_0 + i)
            n2 = _int_to_shot_id(center_1 + i)
            if (n1, n2) in matches or (n2, n1) in matches:
                r = get_transform(n1, n2, pairs)
                rs.append(np.linalg.norm(cv2.Rodrigues(r)[0].ravel()))
                #logger.debug("{} - {} rotation {}".format(n1, n2, rs[-1]))

        if len(rs) > 0:
            avg_rotation  = sum(rs) / len(rs)
            #logger.debug("average rotation {}".format(avg_rotation))
            if avg_rotation < math.pi/9:
                return 'badloop'

    for start_id in ids_0:
        for end_id in ids_1:
            if start_id >= end_id:
                continue

            if (start_id, end_id) in matches or (end_id, start_id) in matches:
                max_retries = 100
                retries = 0
                while retries < max_retries:
                    result = path_finder.findPath(start_id, end_id)
                    if result == 'goodloop':
                        return 'goodloop'
                    elif result == 'badloop':
                        # if loop was found but bad, keep retrying. remember we found bad loop
                        ret_val = 'badloop'
                    else:
                        # if no loop is found, break and try different start/end point
                        break

                    retries += 1

    return ret_val


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
    fids = []
    if (n1, n2) in matches:
        for f1, f2 in matches[n1, n2]:
            for track in tracks:
                if (n1, f1) in track and (n2, f2) in track:
                    fids.append(f1)
                    cnt += 1
                    break
    else:
        for f1, f2 in matches[n2, n1]:
            for track in tracks:
                if (n2, f1) in track and (n1, f2) in track:
                    fids.append(f2)
                    cnt += 1
                    break

    return cnt/base_cnt, fids


def cluster_triplets(valid_triplets, radius):
    """
    merge similar triplets into loop candidates

    :param valid_triplets:
    :param radius:
    :return:
    """
    loop_candidates = []
    for triplet in sorted(valid_triplets):
        added = False
        for cand in loop_candidates:
            if cand.is_close_to(triplet):
                cand.add(triplet)
                added = True
                #break

        if not added:
            new_cand = LoopCandidate(radius)
            new_cand.add(triplet)
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
        if cand.get_center_1() - cand.get_center_0() < 6:
            remove_candidates.append(cand)

    for cand in remove_candidates:
        loop_candidates.remove(cand)

    return loop_candidates


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
        if cand.get_center_1() - cand.get_center_0() < 6:
            remove_candidates.append(cand)

    for cand in remove_candidates:
        loop_candidates.remove(cand)

    return loop_candidates


class PathFinder:
    """
    at initialization, we construct a directed graph that consists of
    'trusted' edges. an edge is considered trusted if it is part of a
    valid triplet.

    the main utility of this class is to return a 'path' between any
    two images, start_id/end_id. a 'path' is set of images in ascending
    order, with a trusted edge between each neighboring pair. each path
    generally goes from one image to a close neighbors at each leg, but
    (occasionally and at random places) jumps over some images. this is
    designed to skip a small subset of images in each path, in the event
    they have bad epipolar geometry.
    """

    def __init__(self, images, matches, pairs, max_jump=5):
        self.path = []
        self.numVertices = 0  # No. of vertices
        self.start = self.finish = 0
        self.pairs = pairs
        self.graph = defaultdict(set)  # default dictionary to store graph

        for im1 in sorted(images):
            for i in range(1, max_jump):
                im2 = _int_to_shot_id(_shot_id_to_int(im1) + i)
                for j in range(1, max_jump):
                    im3 = _int_to_shot_id(_shot_id_to_int(im2) + j)
                    if edge_exists_all([im1, im2, im3], matches):
                        if is_triplet_valid([im1, im2, im3], pairs):
                            #logger.debug("adding edge {} - {} - {}".format(im1, im2, im3))
                            self.addEdge(_shot_id_to_int(im1), _shot_id_to_int(im2))
                            self.addEdge(_shot_id_to_int(im1), _shot_id_to_int(im3))

    # function to add an edge to graph
    def addEdge(self, v, w):
        self.graph[v].add(w)  # Add w to v_s list

    # A recursive function that uses visited[] to detect valid path
    def findPathUtil(self, v, visited, recStack, random_skip):
        # push the current node to stack
        visited[v-self.start] = True
        recStack[v-self.start] = True

        '''
        curr_path = []
        for k, is_in_stack in enumerate(recStack):
            if is_in_stack:
                curr_path.append(_int_to_shot_id(k + self.start))
        logger.debug("on stack {}".format(curr_path))
        '''

        # Recur until we reach the end_id. if random_skip is true, most of the time we sort the
        # neighboring nodes with closest indexed image first, so that we tend to find the longest
        # path. however we occasionally flip the sorting order, in order to randomly skip some
        # vertices (in case they are bad)
        if random_skip:
            isReversed = random.choices(population=[True, False], weights=[0.1, 0.9], k=1)[0]
        else:
            isReversed = False

        for i in sorted(self.graph[v], reverse=isReversed):
            if i < self.finish:
                if not visited[i-self.start]:
                    if self.findPathUtil(i, visited, recStack, random_skip):
                        return True
            elif i == self.finish:
                self.path = []
                for j, is_in_stack in enumerate(recStack):
                    if is_in_stack:
                        self.path.append(_int_to_shot_id(j+self.start))
                self.path.append(_int_to_shot_id(self.finish))
                return True

        # pop this node from stack
        recStack[v-self.start] = False
        return False

    # Returns true if the graph contains a path from start id to end id, else false.
    def findPath(self, start_id, end_id, random_skip=True):
        self.start = _shot_id_to_int(start_id)
        self.finish = _shot_id_to_int(end_id)
        self.numVertices = self.finish - self.start + 1

        # Mark all the vertices as not visited
        visited = [False] * self.numVertices
        recStack = [False] * self.numVertices

        # Call the recursive helper function to detect valid path in different DFS trees
        if self.findPathUtil(self.start, visited, recStack, random_skip):
            #logger.debug("path {}".format(self.path))
            if is_loop_valid(self.path, self.pairs):
                return 'goodloop'
            else:
                return 'badloop'
        else:
            return 'noloop'


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

    def add(self, triplet):
        self.ids_0.add(triplet[0])
        self.ids_1.add(triplet[2])

        i = _shot_id_to_int(triplet[0])
        j = _shot_id_to_int(triplet[1])
        k = _shot_id_to_int(triplet[2])
        if j < (i+k)/2:
            self.ids_0.add(triplet[1])
        else:
            self.ids_1.add(triplet[1])

        # update loop center
        self.center_0 = self.get_average(self.ids_0)
        self.center_1 = self.get_average(self.ids_1)

    def get_average(self, ids):
        total = 0
        for id in ids:
            total += _shot_id_to_int(id)
        return total/len(ids)

    def is_close_to(self, triplet):
        return abs(self.center_0 - _shot_id_to_int(triplet[0])) < self.radius and \
               abs(self.center_1 - _shot_id_to_int(triplet[2])) < self.radius

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


def get_valid_triplets(im1, im2, matches, pairs):
    k = 3
    triplets = []

    ind_im1 = _shot_id_to_int(im1)
    ind_im2 = _shot_id_to_int(im2)

    for i in range(-k, k+1):
        if i == 0:
            continue
        im1_neighbor = _int_to_shot_id(ind_im1+i)
        im2_neighbor = _int_to_shot_id(ind_im2+i)

        if edge_exists_all([im1, im1_neighbor, im2], matches):
            if is_triplet_valid([im1, im1_neighbor, im2], pairs):
                triplets.append(sorted((im1, im1_neighbor, im2)))

        if edge_exists_all([im1, im2_neighbor, im2], matches):
            if is_triplet_valid([im1, im2_neighbor, im2], pairs):
                triplets.append(sorted((im1, im2_neighbor, im2)))

    return triplets


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

            if edge_exists_all([im1, im1_neighbor, im2, im2_neighbor], matches):
                if is_triplet_valid([im1, im1_neighbor, im2], pairs) and \
                        is_triplet_valid([im2, im2_neighbor, im1_neighbor], pairs) and \
                        is_triplet_valid([im1, im1_neighbor, im2_neighbor], pairs) and \
                        is_triplet_valid([im2, im2_neighbor, im1], pairs):
                    quads.append(sorted((im1, im1_neighbor, im2, im2_neighbor)))
            '''
            if edge_exists_all([im1, im1_neighbor, im2], matches) and \
               edge_exists_all([im1_neighbor, im2_neighbor, im2], matches):
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


def edge_exists_all(node_list, matches):
    for im1, im2 in combinations(node_list, 2):
        if not edge_exists(im1, im2, matches):
            return False

    return True


def is_triplet_valid(triplet, pairs):
    '''
    Rji = get_transform(i, j, pairs)
    Rkj = get_transform(j, k, pairs)
    Rik = get_transform(k, i, pairs)

    if np.linalg.norm(cv2.Rodrigues(Rik.dot(Rkj.dot(Rji)))[0].ravel()) < math.pi/12:
        return True
    else:
        return False
    '''

    return is_loop_valid(triplet, pairs, thresh=math.pi/18)


def is_loop_valid(path, pairs, thresh=math.pi/9):
    R = np.identity(3, dtype=float)
    for n1, n2 in zip(path, path[1:]):
        r = get_transform(n1, n2, pairs)
        if r.size == 0:
            return False
        R = r.dot(R)

    r = get_transform(path[-1], path[0], pairs)
    if r.size == 0:
        return False
    R = r.dot(R)

    #logger.debug("error={} thresh={}".format(np.linalg.norm(cv2.Rodrigues(R)[0].ravel()), thresh))
    if np.linalg.norm(cv2.Rodrigues(R)[0].ravel()) < thresh:
        return True
    else:
        return False


def rotation_close_to_preint(im1, im2, T, pdr_shots_dict):
    """
    compare relative rotation of robust matching to that of imu gyro preintegration,
    if they are not close, it is considered to be an erroneous epipoar geometry
    """
    if abs(_shot_id_to_int(im1) - _shot_id_to_int(im2)) >= 5:
        # because of drift, we don't perform pre-integration check if im1 and im2 are
        # far apart in sequence number
        return True

    # calculate relative rotation from preintegrated gyro input
    preint_im1_rot = cv2.Rodrigues(np.asarray([pdr_shots_dict[im1][7], pdr_shots_dict[im1][8], pdr_shots_dict[im1][9]]))[0]
    preint_im2_rot = cv2.Rodrigues(np.asarray([pdr_shots_dict[im2][7], pdr_shots_dict[im2][8], pdr_shots_dict[im2][9]]))[0]
    preint_rel_rot = np.dot(preint_im2_rot, preint_im1_rot.T)

    # convert this rotation from sensor frame to camera frame
    b_to_c = np.asarray([1, 0, 0, 0, 0, -1, 0, 1, 0]).reshape(3, 3)
    preint_rel_rot = cv2.Rodrigues(b_to_c.dot(cv2.Rodrigues(preint_rel_rot)[0].ravel()))[0]

    # get relative rotation from T obtained from robust matching
    robust_match_rel_rot = T[:, :3]

    # calculate difference between the two relative rotations. this is the geodesic distance
    # see D. Huynh "Metrics for 3D Rotations: Comparison and Analysis" equation 23
    diff_rot = np.dot(preint_rel_rot, robust_match_rel_rot.T)
    geo_diff = np.linalg.norm(cv2.Rodrigues(diff_rot)[0].ravel())

    if geo_diff < math.pi/6.0:
        logger.debug("{} {} preint/robust geodesic {} within threshold".format(im1, im2, geo_diff))
        return True
    else:
        #logger.debug("preint rel rot axis/angle = {}".format(_get_axis_angle(preint_rel_rot)))
        #logger.debug("robust rel rot axis/angle = {}".format(_get_axis_angle(robust_match_rel_rot)))
        logger.debug("{} {} preint/robust geodesic {} exceeds threshold".format(im1, im2, geo_diff))
        return False


def _get_axis_angle(rot_mat):
    axis_angle = cv2.Rodrigues(rot_mat)[0].ravel()
    angle = np.linalg.norm(axis_angle)
    axis = axis_angle / angle
    return axis, angle


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


def _good_track(track, min_length):
    if len(track) < min_length:
        return False
    images = [f[0] for f in track]
    if len(images) != len(set(images)):
        return False
    return True

