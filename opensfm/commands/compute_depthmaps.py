import sys
import logging
import subprocess

from opensfm import dataset
from opensfm import dense

logger = logging.getLogger(__name__)


class Command:
    name = 'compute_depthmaps'
    help = "Compute depthmap"

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')
        parser.add_argument('--interactive',
                            help='plot results as they are being computed',
                            action='store_true')

    def run(self, args):
        data = dataset.DataSet(args.dataset)
        data.config['interactive'] = args.interactive
        reconstructions = data.load_undistorted_reconstruction()
        graph = data.load_undistorted_tracks_graph()

        for reconstruction in reconstructions:
            dense.compute_depthmaps(data, graph, reconstruction)

        # generate merged.ply
        dense.merge_depthmaps(data, reconstructions)

        # create densified version of reconstruction.json and tracks.csv.
        # note these operations are not parallelized to limit memory usage
        reconstructions = data.load_reconstruction()

        '''
        dense.densify_tracks(data, reconstructions)
        dense.densify_reconstructions(data, reconstructions)
        '''

        # generate poses.csv
        dense.save_poses(data, reconstructions)

        # these will be moved to config files later
        visible_angle_thresh = 10.0
        max_distance_thresh = 999999
        angular_resolution_x = 0.5
        angular_resolution_y = 0.5

        try:
            subprocess.call(['frame_point_cloud', 'merged.ply', 'poses.csv',
                             '-rx', str(angular_resolution_x), '-ry', str(angular_resolution_y),
                             '-at', str(visible_angle_thresh), '-dt', str(max_distance_thresh)],
                            cwd=data._densified_path())
        except OSError as e:
            sys.stderr.write("frame_point_cloud execution failed: {}\n".format(e))

        # clean up intermediate files
        data.densification_cleanup()
