import logging
import numpy as np
from timeit import default_timer as timer

from networkx.algorithms import bipartite

from opensfm import dataset
from opensfm import io
from opensfm import matching
from opensfm.commands import superpoint

logger = logging.getLogger(__name__)


class Command:
    name = 'create_tracks'
    help = "Link matches pair-wise matches into tracks"

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        data = dataset.DataSet(args.dataset)

        start = timer()
        features, colors, reg_counts = self.load_features(data)
        features_end = timer()
        matches = self.load_matches(data)
        matches_end = timer()
        tracks_graph, tracks_superpoint = matching.create_tracks_graph(features, colors, reg_counts, matches,
                                                    data.config)
        tracks_end = timer()
        data.save_tracks_graph(tracks_graph)
        data.save_tracks_superpoint(tracks_superpoint)
        end = timer()

        with open(data.profile_log(), 'a') as fout:
            fout.write('create_tracks: {0}\n'.format(end - start))

        self.write_report(data,
                          tracks_graph,
                          features_end - start,
                          matches_end - features_end,
                          tracks_end - matches_end)

    def load_features(self, data):
        logging.info('reading features')
        features = {}
        colors = {}
        reg_counts = {}
        for im in data.images():
            p, f, c = data.load_features(im)

            if p is None:
                features[im] = []
                colors[im] = []
                reg_counts[im] = 0
                continue

            reg_counts[im] = len(p)

            p_s, f_s, c_s = superpoint.load_features(im)
            if p_s is not None:
                p = np.concatenate((p, p_s), axis=0)
                c = np.concatenate((c, c_s), axis=0)

            features[im] = p[:, :2]
            colors[im] = c

        return features, colors, reg_counts

    def load_matches(self, data):
        matches = {}
        for im1 in data.images():
            try:
                im1_matches = data.load_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                matches[im1, im2] = im1_matches[im2]
        return matches

    def write_report(self, data, graph,
                     features_time, matches_time, tracks_time):
        tracks, images = matching.tracks_and_images(graph)
        image_graph = bipartite.weighted_projected_graph(graph, images)
        view_graph = []
        for im1 in data.images():
            for im2 in data.images():
                if im1 in image_graph and im2 in image_graph[im1]:
                    weight = image_graph[im1][im2]['weight']
                    view_graph.append((im1, im2, weight))

        report = {
            "wall_times": {
                "load_features": features_time,
                "load_matches": matches_time,
                "compute_tracks": tracks_time,
            },
            "wall_time": features_time + matches_time + tracks_time,
            "num_images": len(images),
            "num_tracks": len(tracks),
            "view_graph": view_graph
        }
        data.save_report(io.json_dumps(report), 'tracks.json')
