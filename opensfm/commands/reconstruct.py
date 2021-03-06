import logging
import time
import json

from opensfm import dataset
from opensfm import io
from opensfm import reconstruction

logger = logging.getLogger(__name__)


class Command:
    name = 'reconstruct'
    help = "Compute the reconstruction"

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')
        parser.add_argument('--partials-only', nargs = '*', help = 'indices of partial reconstructions to reconstruct' )
        parser.add_argument('--excluded-images', nargs = '*', help = 'ids/names of images to exclude' )
        parser.add_argument('--remove-image-subset', help = 'JSON file specifying a subset of images to remove' )
        parser.add_argument('--image-subset', help = 'JSON file specifying a subset of images to reprocess' )
        parser.add_argument('--direct-align', help = 'Alignment to gps points using PDR', action='store_true' )
        parser.add_argument('--hybrid-align', help = 'Alignment to gps points using SfM+PDR', action='store_true' )

    def run(self, args):
        
        start = time.time()

        partials = []
        if args.partials_only:
            partials = [ int(p) for p in args.partials_only ]
        
        excluded_images = []
        if args.excluded_images:
            excluded_images = args.excluded_images

        remove_image_subset = []
        if args.remove_image_subset is not None:
            with open( args.remove_image_subset, 'r' ) as fin:
                remove_image_subset = json.load( fin )

        image_subset = []
        if args.image_subset is not None:
            with open( args.image_subset, 'r' ) as fin:
                image_subset = json.load( fin )
        
        data = dataset.DataSet( args.dataset )
        
        if len(partials) > 0:
            
            target_images = []
            
            # Validate that the partial reconstruction in question exists and
            # save a backup of the current reconstruction
        
            reconstructions = data.load_reconstruction()
            for p_ind in partials:
                if p_ind < 0 or p_ind > len(reconstructions)-1:
                    logger.debug('Partial reconstruction {0} does not exist.'.format( p_ind ) )
                    return
                
                recon = reconstructions[p_ind]
                
                shot_ids = [ shot.id for shot in recon.shots.values() ]
                
                target_images.extend( shot_ids )
            
            try:
                for shot_id in excluded_images:
                    target_images.remove( shot_id )
            except ValueError as e:
                logger.debug('Excluded image {0} does not exist.'.format( shot_id ) )
                return
                
            data.config['target_images'] = target_images
            
            data.save_reconstruction( reconstructions , "reconstruction.json.bak" )
        
        elif image_subset:
        
            if data.reconstruction_exists():
                reconstructions = data.load_reconstruction()
           
                data.save_reconstruction( reconstructions , "reconstruction.json.bak" )
            
            else:
                reconstructions = []

            data.config['target_images'] = image_subset

        elif remove_image_subset:

            if data.reconstruction_exists():
                reconstructions = data.load_reconstruction()

                data.save_reconstruction( reconstructions , "reconstruction.json.bak" )

            else:
                logger.debug('Reconstructions do not exist, cannot remove')
                return

            for id in remove_image_subset:
                for recon in reconstructions:

                    to_remove = [ s for s in recon.shots.values() if s.id == id ]

                    for s in to_remove:
                        del recon.shots[s.id]

            # remove any recon that becomes empty
            reconstructions[:] = [recon for recon in reconstructions if recon.shots]

            data.save_reconstruction( reconstructions )
            return

        if args.direct_align:
            # Run alignment, pdr only
            if data.pdr_shots_exist():
                report = reconstruction.direct_align_reconstruction_pdr( data )
            else:
                # legacy code
                report = reconstruction.direct_align_reconstruction( data )

        elif args.hybrid_align:
            # Run alignment, sfm+pdr
            if data.reconstruction_exists() and data.pdr_shots_exist():
                report = reconstruction.hybrid_align_reconstruction_pdr( data )

        else:
            # Run incremental reconstruction
            graph = data.load_tracks_graph()
            if data.is_sequential():
                report, reconstructions = reconstruction. \
                    incremental_reconstruction_sequential(data, graph)
            else:
                report, reconstructions = reconstruction. \
                    incremental_reconstruction(data, graph)

        # If we are re-processing partial reconstructions only then merge the
        # new results with the original reconstruction. 
        
        new_recons = data.load_reconstruction()

        if len(partials) > 0:
            
            merged_recons = []
            
            for ind, recon in enumerate(reconstructions):
                if ind not in partials:
                    merged_recons.append( recon )
                    
            for recon in new_recons:
                merged_recons.append( recon )
    
        elif image_subset:
            
            merged_recons = []
            
            for id in image_subset:
                for recon in reconstructions:
                    
                    to_remove = [ s for s in recon.shots.values() if s.id == id ]
                    
                    for s in to_remove:
                        del recon.shots[s.id]
            
            for recon in reconstructions:
                if recon.shots:
                    merged_recons.append( recon )
            
            for recon in new_recons:
                merged_recons.append( recon )
    
        else:
            merged_recons = new_recons

        # Subsets and splits don't prevent duplicates across partial reconstructions
        # The first reconstruction with a particular image gets to keep it.
        
        num_recons = len(merged_recons)
        for ind,mrecon in enumerate(merged_recons):
            for shot in mrecon.shots.values():
                sid = shot.id
                for subsequent_recon in merged_recons[ind+1:num_recons]:
                    
                    to_remove = [ s for s in subsequent_recon.shots.values() if s.id == sid ]
                    
                    for s in to_remove:
                        del subsequent_recon.shots[s.id]

        data.save_reconstruction( merged_recons )

        end = time.time()
        with open(data.profile_log(), 'a') as fout:
            fout.write('reconstruct: {0}\n'.format(end - start))

        data.save_report(io.json_dumps(report), 'reconstruction.json')
