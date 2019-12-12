import logging
from timeit import default_timer as timer

import numpy as np

from opensfm import dataset
from opensfm import io
from opensfm import log
from opensfm import matching
from opensfm import pairs_selection
from opensfm.context import parallel_map
from opensfm.commands import superpoint


logger = logging.getLogger(__name__)


class Command:
    name = 'match_features'
    help = 'Match features between image pairs'
    
    def __init__(self):
        self.checkpoint_callback = None

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        data = dataset.DataSet( args.dataset )
        images = data.images()
        exifs = {im: data.load_exif(im) for im in images}
        
        preport = {}
        
        try:
            # Attempt to load the report from any previous runs
            report_str = data.load_report('matches.json' )
            preport = io.json_loads( report_str )
        except:
            pass
            
        use_report_pairs = True
        if 'parameters' in preport:
            params = preport['parameters']
            use_report_pairs = matching_config_unchanged( data.config, params )
            
        if 'pairs' in preport and use_report_pairs:
            pairs = preport['pairs']
        else:
            pairs, preport = pairs_selection.match_candidates_from_metadata(images, exifs, data)
            
            # Append a field for indicating processing status (processed or not processed)
            for img in pairs:
                candidates = pairs[img]
                pairs[img] = [ candidates, False ]

        num_pairs = sum(len(c) for c in pairs.values())
        logger.info('Matching {} image pairs'.format(num_pairs))

        ctx = Context()
        ctx.data = data
        ctx.cameras = ctx.data.load_camera_models()
        ctx.exifs = exifs

        # Group processing inputs into 150 images at a time.
        
        pairs_groups = [ {} ]
        
        pg_ind = 0
        for ind,img in enumerate(pairs):
            cur_pair_group = pairs_groups[ pg_ind ]
            cur_pair_group[img] = pairs[img]
            if len(cur_pair_group) == 150:
                pairs_groups.append( {} )
                pg_ind += 1
        
        processes = ctx.data.config['processes']
        
        run_time = 0
        for pg in pairs_groups:
        
            pargs = list(match_arguments(pg, ctx))
        
            if len(pargs) > 0:
        
                start = timer()
            
                parallel_map(match, pargs, processes)
                
                end = timer()
            
                run_time += end - start
                
                # Mark processing as complete for these pairs.
                
                for img in pg:
                    pairs[img][1] = True
                    
                with open(ctx.data.profile_log(), 'a') as fout:
                    fout.write('match_features: {0}\n'.format(run_time))
                self.write_report(data, preport, pairs, run_time)
    
                if self.checkpoint_callback is not None:
                    self.checkpoint_callback( args.dataset )
                    
    
    def set_checkpoint_callback( self, callback ):
        self.checkpoint_callback = callback

    def write_report(self, data, preport, pairs, wall_time):
        pair_list = []
        for im1, others in pairs.items():
            for im2 in others:
                pair_list.append((im1, im2))
                
        proc_params = {}
        
        g_match = proc_params['general_matching'] = {}
        
        g_match['matcher_type'] = data.config['matcher_type']
        g_match['matching_gps_distance'] = data.config['matching_gps_distance']
        g_match['matching_gps_neighbors'] = data.config['matching_gps_neighbors']
        g_match['matching_time_neighbors'] = data.config['matching_time_neighbors']
        g_match['matching_order_neighbors'] = data.config['matching_order_neighbors']
        g_match['matching_pdr_distance'] = data.config['matching_pdr_distance']
        g_match['robust_matching_threshold'] = data.config['robust_matching_threshold']
        g_match['robust_matching_calib_threshold'] = data.config['robust_matching_calib_threshold']

        flann = proc_params['flann'] = {}
        
        flann['flann_branching'] = data.config['flann_branching']
        flann['flann_iterations'] = data.config['flann_iterations']
        flann['flann_checks'] = data.config['flann_checks']

        bow = proc_params['bow'] = {}

        bow['bow_num_checks'] = data.config['bow_num_checks']

        report = {
            "wall_time": wall_time,
            "num_pairs": len(pair_list),
            "parameters": proc_params,
            "pairs": pairs,
        }
        report.update(preport)
        data.save_report(io.json_dumps(report), 'matches.json')


class Context:
    pass


def matching_config_unchanged( config, match_params ):

    g_match = match_params['general_matching']
    flann = match_params['flann']
    bow = match_params['bow']
    
    return (
           g_match['matcher_type'] == config['matcher_type'] and
           g_match['matching_gps_distance'] == config['matching_gps_distance']  and
           g_match['matching_gps_neighbors'] == config['matching_gps_neighbors']  and
           g_match['matching_time_neighbors'] == config['matching_time_neighbors']  and
           g_match['matching_order_neighbors'] == config['matching_order_neighbors']  and
           g_match['matching_pdr_distance'] == config['matching_pdr_distance']  and
           g_match['robust_matching_threshold'] == config['robust_matching_threshold']  and
           g_match['robust_matching_calib_threshold'] == config['robust_matching_calib_threshold']  and
           flann['flann_branching'] == config['flann_branching']  and
           flann['flann_iterations'] == config['flann_iterations']  and
           flann['flann_checks'] == config['flann_checks']  and
           bow['bow_num_checks'] == bow['bow_num_checks'] )


def match_candidates_from_metadata(images, exifs, data):
    """Compute candidate matching pairs"""
    max_distance = data.config['matching_gps_distance']
    gps_neighbors = data.config['matching_gps_neighbors']
    time_neighbors = data.config['matching_time_neighbors']
    order_neighbors = data.config['matching_order_neighbors']
    pdr_max_distance = data.config['matching_pdr_distance']

    if not data.reference_lla_exists():
        data.invent_reference_lla()
    reference = data.load_reference_lla()

    if not all(map(has_gps_info, exifs.values())):
        if gps_neighbors != 0:
            logger.warn("Not all images have GPS info. "
                        "Disabling matching_gps_neighbors.")
        gps_neighbors = 0
        max_distance = 0

    images.sort()

    logger.info('match_candidates_from_metadata: pdr_max_distance {}'.format(pdr_max_distance))
    if max_distance == gps_neighbors == time_neighbors == order_neighbors == pdr_max_distance == 0:
        # All pair selection strategies deactivated so we match all pairs
        d = set()
        t = set()
        o = set()
        p = set()
        pairs = combinations(images, 2)
    else:
        d = match_candidates_by_distance(images, exifs, reference,
                                         gps_neighbors, max_distance)
        t = match_candidates_by_time(images, exifs, time_neighbors)
        o = match_candidates_by_order(images, order_neighbors, data)
        p = match_candidates_by_pdr(images, pdr_max_distance, data)
        pairs = d | t | o | p
        logger.info("num_pairs_pdr={}".format(len(p)))

    res = {im: [] for im in images}
    for im1, im2 in pairs:
        res[im1].append(im2)

    report = {
        "num_pairs_distance": len(d),
        "num_pairs_time": len(t),
        "num_pairs_order": len(o),
        "num_pairs_pdr": len(p)
    }
    return res, report


def match_arguments(pairs, ctx):
    for i, (im, ( candidates, processed )) in enumerate(pairs.items()):
        if not processed:
            yield im, candidates, i, len(pairs), ctx


def match(args):
    """Compute all matches for a single image"""
    log.setup()

    im1, candidates, i, n, ctx = args
    logger.info('Matching {}  -  {} / {}'.format(im1, i + 1, n))

    config = ctx.data.config
    matcher_type = config['matcher_type']
    robust_matching_min_match = config['robust_matching_min_match']

    im1_matches = {}

    for im2 in candidates:
        # symmetric matching
        t = timer()
        p1, f1, c1 = ctx.data.load_features(im1)
        p2, f2, c2 = ctx.data.load_features(im2)

        if p1 is None or p2 is None:
            im1_matches[im2] = []
            continue

        p1_s, f1_s, c1_s = superpoint.load_features(im1)
        p2_s, f2_s, c2_s = superpoint.load_features(im2)

        if p1_s is not None and p2_s is not None:
            p1 = np.concatenate((p1, p1_s), axis=0)
            f1 = np.concatenate((f1, f1_s), axis=0)
            p2 = np.concatenate((p2, p2_s), axis=0)
            f2 = np.concatenate((f2, f2_s), axis=0)

        if matcher_type == 'WORDS':
            w1 = ctx.data.load_words(im1)
            w2 = ctx.data.load_words(im2)
            matches = matching.match_words_symmetric(f1, w1, f2, w2, config)
        elif matcher_type == 'FLANN':
            i1 = ctx.data.load_feature_index(im1, f1)
            i2 = ctx.data.load_feature_index(im2, f2)
            matches = matching.match_flann_symmetric(f1, i1, f2, i2, config)
        elif matcher_type == 'BRUTEFORCE':
            matches = matching.match_brute_force_symmetric(f1, f2, config)
        else:
            raise ValueError("Invalid matcher_type: {}".format(matcher_type))

        logger.debug('{} - {} has {} candidate matches'.format(
            im1, im2, len(matches)))
        if len(matches) < robust_matching_min_match:
            im1_matches[im2] = []
            continue

        # robust matching
        t_robust_matching = timer()
        camera1 = ctx.cameras[ctx.exifs[im1]['camera']]
        camera2 = ctx.cameras[ctx.exifs[im2]['camera']]

        rmatches = matching.robust_match(p1, p2, camera1, camera2, matches,
                                         config)

        if len(rmatches) < robust_matching_min_match:
            im1_matches[im2] = []
            continue
        im1_matches[im2] = rmatches
        logger.debug('Robust matching time : {0}s'.format(
            timer() - t_robust_matching))

        logger.debug("{} - {} Full matching {} / {}, time: {}s".format(
            im1, im2, len(rmatches), len(matches), timer() - t))
    ctx.data.save_matches(im1, im1_matches)
