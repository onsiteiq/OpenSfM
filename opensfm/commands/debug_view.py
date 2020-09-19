import os
import sys
import json
import logging
import numpy as np
import webbrowser

from opensfm import io

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

    # by default the viewer shows a grid that measures 80x80 and we want to fit into that
    s = 80.0/max([maxx-minx, maxy-miny])

    # Our floorplan/gps coordinate system: x point right, y point back, z point down
    #
    # OpenSfM 3D viewer coordinate system: x point left, y point back, z point up (or equivalently it can be
    # viewed as x point right, y point forward, z point up)
    #
    # Since our floorplan/gps uses a different coordinate system than the OpenSfM 3D viewer, reconstructions
    # would look upside down in the 3D viewer. We therefore perform a transformation below to correct that.
    #
    A = np.array([[1, 0, 0],
                  [0, -1, 0],
                  [0, 0, -1]])
    b = np.array([
        -(minx+maxx)/2.0*s,
        (miny+maxy)/2.0*s,
        meanz*s
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

    # save scaled recon
    with io.open_wt('reconstruction_debug_view.json') as fout:
        io.json_dump(io.reconstructions_to_json(recons), fout, False)

    # create symbolic links
    images_dir = os.path.join(os.getcwd(), "images")
    debug_view_file = os.path.join(os.getcwd(), "reconstruction_debug_view.json")
    os.chdir(os.path.join(os.environ['HOME'], 'source/OpenSfM'))
    os.system("rm -fr data/reconstruction_debug_view.json")
    os.system("ln -s " + debug_view_file + " data/reconstruction_debug_view.json")
    os.system("rm -fr data/images")
    os.system("ln -s " + images_dir + " data/images")
    os.system("touch data/images/*")

    # launch http.server and show scaled recons. if http.server is already running, the following
    # will show an error, but with no harm
    os.system("python3 -m http.server &")
    webbrowser.open('http://localhost:8000/viewer/reconstruction.html#file=/data/reconstruction_debug_view.json', new=2)


# Entry point
if __name__ == "__main__":
    """
    rescale recon(s), then launch browser for viewing
    
    syntax: python3 debug_view.py show_num [recon_file]
        show_num == -1: all recons
        show_num == n: n is index of recon to be viewed
    """

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

    if len(sys.argv) > 1:
        show_num = int(sys.argv[1])

        if len(sys.argv) > 2:
            recon_file = sys.argv[2]

    if recon_file is None:
        if show_num == -1:
            recon_file = 'aligned_reconstructions.json'
        else:
            recon_file = 'reconstruction.json'

    with open(recon_file) as fin:
        recons = io.reconstructions_from_json(json.load(fin))
    recons = sorted(recons, key=lambda x: -len(x.shots))

    if show_num == -1:
        debug_rescale_reconstructions(recons)
    else:
        debug_rescale_reconstructions([recons[show_num]])
