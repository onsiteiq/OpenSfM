# -*- coding: utf-8 -*-

import os
import json
import logging
import pickle
import gzip
import glob

import numpy as np
import six
import shutil

from opensfm import io
from opensfm import config
from opensfm import context
from opensfm import geo
from opensfm import tracking
from opensfm import features
from opensfm import upright


logger = logging.getLogger(__name__)


class DataSet(object):
    """Accessors to the main input and output data.

    Data include input images, masks, and segmentation as well
    temporary data such as features and matches and the final
    reconstructions.

    All data is stored inside a single folder with a specific subfolder
    structure.

    It is possible to store data remotely or in different formats
    by subclassing this class and overloading its methods.
    """
    def __init__(self, data_path):
        """Init dataset associated to a folder."""
        self.data_path = data_path
        self.gps_points_dict = {}
        self.pdr_shots_dict = {}
        self.topocentric_gps_points_dict = {}
        self.pdr_predictions_dict = {}
        self.hlf_list = []
        self.hlf_det_list = []
        self._load_config()
        self._load_image_list()
        self._load_mask_list()

    def _load_config(self):
        config_file = os.path.join(self.data_path, 'config.yaml')
        self.config = config.load_config(config_file)

    def _load_image_list(self):
        """Load image list from image_list.txt or list image/ folder."""
        image_list_file = os.path.join(self.data_path, 'image_list.txt')
        if os.path.isfile(image_list_file):
            with io.open_rt(image_list_file) as fin:
                lines = fin.read().splitlines()
            self._set_image_list(lines)
        else:
            self._set_image_path(os.path.join(self.data_path, 'images'))

    def images(self):
        """List of file names of all images in the dataset."""
        return self.image_list

    def is_sequential(self):
        for file_name in self.image_list:
            file_name_no_ext = file_name.split(".")[0]
            if len(file_name_no_ext) != 10:
                return False

            for c in file_name_no_ext:
                if not c.isdigit():
                    return False

        return True

    def _image_file(self, image):
        """Path to the image file."""
        return self.image_files[image]

    def open_image_file(self, image):
        """Open image file and return file object."""
        return open(self._image_file(image), 'rb')

    def load_image(self, image):
        """Load image pixels as numpy array.

        The array is 3D, indexed by y-coord, x-coord, channel.
        The channels are in RGB order.
        """
        return io.imread(self._image_file(image))

    def image_size(self, image):
        """Height and width of the image."""
        return io.image_size(self._image_file(image))

    def _undistorted_image_path(self):
        return os.path.join(self.data_path, 'undistorted')

    def _undistorted_image_file(self, image):
        """Path of undistorted version of an image."""
        return os.path.join(self._undistorted_image_path(), image + '.jpg')

    def load_undistorted_image(self, image):
        """Load undistorted image pixels as a numpy array."""
        return io.imread(self._undistorted_image_file(image))

    def save_undistorted_image(self, image, array):
        """Save undistorted image pixels."""
        io.mkdir_p(self._undistorted_image_path())
        io.imwrite(self._undistorted_image_file(image), array)
        
    def undistorted_image_size(self, image):
        """Height and width of the undistorted image."""
        return io.image_size(self._undistorted_image_file(image))

    def _full_mosaic_path(self):
        return os.path.join(self.data_path, 'full_mosaics')

    def _full_mosaic_file(self, image):
        """Path of undistorted version of an image."""
        return os.path.join(self._full_mosaic_path(), image + '.jpg')

    def save_full_mosaic_image(self, image, array):
        io.mkdir_p(self._full_mosaic_path())
        io.imwrite(self._full_mosaic_file(image))


    def _load_mask_list(self):
        """Load mask list from mask_list.txt or list masks/ folder."""
        mask_list_file = os.path.join(self.data_path, 'mask_list.txt')
        if os.path.isfile(mask_list_file):
            with io.open_rt(mask_list_file) as fin:
                lines = fin.read().splitlines()
            self._set_mask_list(lines)
        else:
            self._set_mask_path(os.path.join(self.data_path, 'masks'))

    def load_mask(self, image):
        """Load image mask if it exists, otherwise return None."""
        if image in self.mask_files:
            mask_path = self.mask_files[image]
            mask = io.imread(mask_path, grayscale=True)
            if mask is None:
                raise IOError("Unable to load mask for image {} "
                              "from file {}".format(image, mask_path))
        else:
            mask = None
        return mask

    def load_features_mask(self, image, feat):
        """Load a feature-wise mask if it exists, otherwise return None.

        This mask is used when performing features matching.
        """
        if feat is None or len(feat) == 0:
            return np.array([], dtype=bool)

        mask_image = self.load_combined_mask(image)
        if mask_image is None:
            logger.debug('No segmentation for {}, no features masked.'.format(image))
            return np.ones((feat.shape[0],), dtype=bool)

        exif = self.load_exif(image)
        width = exif["width"]
        height = exif["height"]
        orientation = exif["orientation"]

        new_height, new_width = mask_image.shape
        ps = upright.opensfm_to_upright(feat, width, height, orientation,
                                        new_width=new_width, new_height=new_height).astype(int)
        mask = mask_image[ps[:, 1], ps[:, 0]]

        n_removed = len(mask) - np.sum(mask)
        logger.debug('Masking {} / {} ({:.2f}) features for {}'.format(
                    n_removed, len(mask), n_removed / len(mask), image))

        return np.array(mask, dtype=bool)

    def _undistorted_mask_path(self):
        return os.path.join(self.data_path, 'undistorted_masks')

    def _undistorted_mask_file(self, image):
        """Path of undistorted version of a mask."""
        return os.path.join(self._undistorted_mask_path(), image + '.png')

    def undistorted_mask_exists(self, image):
        """Check if the undistorted mask file exists."""
        return os.path.isfile(self._undistorted_mask_file(image))

    def load_undistorted_mask(self, image):
        """Load undistorted mask pixels as a numpy array."""
        return io.imread(self._undistorted_mask_file(image), grayscale=True)

    def save_undistorted_mask(self, image, array):
        """Save the undistorted image mask."""
        io.mkdir_p(self._undistorted_mask_path())
        io.imwrite(self._undistorted_mask_file(image), array)

    def _detection_path(self):
        return os.path.join(self.data_path, 'detections')

    def _detection_file(self, image):
        return os.path.join(self._detection_path(), image + '.png')

    def load_detection(self, image):
        """Load image detection if it exists, otherwise return None."""
        detection_file = self._detection_file(image)
        if os.path.isfile(detection_file):
            detection = io.imread(detection_file, grayscale=True)
        else:
            detection = None
        return detection

    def _undistorted_detection_path(self):
        return os.path.join(self.data_path, 'undistorted_detections')

    def _undistorted_detection_file(self, image):
        """Path of undistorted version of a detection."""
        return os.path.join(self._undistorted_detection_path(), image + '.png')

    def undistorted_detection_exists(self, image):
        """Check if the undistorted detection file exists."""
        return os.path.isfile(self._undistorted_detection_file(image))

    def load_undistorted_detection(self, image):
        """Load an undistorted image detection."""
        detection = io.imread(self._undistorted_detection_file(image),
                              grayscale=True)
        return detection

    def save_undistorted_detection(self, image, array):
        """Save the undistorted image detection."""
        io.mkdir_p(self._undistorted_detection_path())
        io.imwrite(self._undistorted_detection_file(image), array)

    def _segmentation_path(self):
        return os.path.join(self.data_path, 'segmentations')

    def _segmentation_file(self, image):
        return os.path.join(self._segmentation_path(), image + '.png')

    def load_segmentation(self, image):
        """Load image segmentation if it exists, otherwise return None."""
        segmentation_file = self._segmentation_file(image)
        if os.path.isfile(segmentation_file):
            segmentation = io.imread(segmentation_file, grayscale=True)
        else:
            segmentation = None
        return segmentation

    def _undistorted_segmentation_path(self):
        return os.path.join(self.data_path, 'undistorted_segmentations')

    def _undistorted_segmentation_file(self, image):
        """Path of undistorted version of a segmentation."""
        return os.path.join(self._undistorted_segmentation_path(), image + '.png')

    def undistorted_segmentation_exists(self, image):
        """Check if the undistorted segmentation file exists."""
        return os.path.isfile(self._undistorted_segmentation_file(image))

    def load_undistorted_segmentation(self, image):
        """Load an undistorted image segmentation."""
        segmentation = io.imread(self._undistorted_segmentation_file(image),
                                 grayscale=True)
        return segmentation

    def save_undistorted_segmentation(self, image, array):
        """Save the undistorted image segmentation."""
        io.mkdir_p(self._undistorted_segmentation_path())
        io.imwrite(self._undistorted_segmentation_file(image), array)

    def segmentation_ignore_values(self, image):
        """List of label values to ignore.

        Pixels with this labels values will be masked out and won't be
        processed when extracting features or computing depthmaps.
        """
        return self.config.get('segmentation_ignore_values', [])

    def load_segmentation_mask(self, image):
        """Build a mask from segmentation ignore values.

        The mask is non-zero only for pixels with segmentation
        labels not in segmentation_ignore_values.
        """
        ignore_values = self.segmentation_ignore_values(image)
        if not ignore_values:
            return None

        segmentation = self.load_segmentation(image)
        if segmentation is None:
            return None

        return self._mask_from_segmentation(segmentation, ignore_values)

    def load_undistorted_segmentation_mask(self, image):
        """Build a mask from the undistorted segmentation.

        The mask is non-zero only for pixels with segmentation
        labels not in segmentation_ignore_values.
        """
        ignore_values = self.segmentation_ignore_values(image)
        if not ignore_values:
            return None

        segmentation = self.load_undistorted_segmentation(image)
        if segmentation is None:
            return None

        return self._mask_from_segmentation(segmentation, ignore_values)

    def _mask_from_segmentation(self, segmentation, ignore_values):
        mask = np.ones(segmentation.shape, dtype=np.uint8)
        for value in ignore_values:
            mask &= (segmentation != value)
        return mask

    def load_combined_mask(self, image):
        """Combine binary mask with segmentation mask.

        Return a mask that is non-zero only where the binary
        mask and the segmentation mask are non-zero.
        """
        mask = self.load_mask(image)
        smask = self.load_segmentation_mask(image)
        return self._combine_masks(mask, smask)

    def load_undistorted_combined_mask(self, image):
        """Combine undistorted binary mask with segmentation mask.

        Return a mask that is non-zero only where the binary
        mask and the segmentation mask are non-zero.
        """
        mask = None
        if self.undistorted_mask_exists(image):
            mask = self.load_undistorted_mask(image)
        smask = None
        if self.undistorted_segmentation_exists(image):
            smask = self.load_undistorted_segmentation_mask(image)
        return self._combine_masks(mask, smask)

    def _combine_masks(self, mask, smask):
        if mask is None:
            if smask is None:
                return None
            else:
                return smask
        else:
            if smask is None:
                return mask
            else:
                return mask & smask

    def _depthmap_path(self):
        return os.path.join(self.data_path, 'depthmaps')

    def _depthmap_file(self, image, suffix):
        """Path to the depthmap file"""
        return os.path.join(self._depthmap_path(), image + '.' + suffix)

    def _image_tracks_path(self):
        """Path to save densified tracks fro each subshot"""
        return os.path.join(self.data_path, 'depthmaps')

    def _image_tracks_file(self, image, suffix):
        """Path to the densified tracks file for each subshot"""
        return os.path.join(self._image_tracks_path(), image + '.' + suffix)

    def raw_depthmap_exists(self, image):
        return os.path.isfile(self._depthmap_file(image, 'raw.npz'))

    def save_raw_depthmap(self, image, depth, plane, score, nghbr, nghbrs):
        io.mkdir_p(self._depthmap_path())
        filepath = self._depthmap_file(image, 'raw.npz')
        np.savez_compressed(filepath, depth=depth, plane=plane, score=score, nghbr=nghbr, nghbrs=nghbrs)

    def load_raw_depthmap(self, image):
        o = np.load(self._depthmap_file(image, 'raw.npz'))
        return o['depth'], o['plane'], o['score'], o['nghbr'], o['nghbrs']

    def clean_depthmap_exists(self, image):
        return os.path.isfile(self._depthmap_file(image, 'clean.npz'))

    def save_clean_depthmap(self, image, depth, plane, score):
        io.mkdir_p(self._depthmap_path())
        filepath = self._depthmap_file(image, 'clean.npz')
        np.savez_compressed(filepath, depth=depth, plane=plane, score=score)

    def load_clean_depthmap(self, image):
        o = np.load(self._depthmap_file(image, 'clean.npz'))
        return o['depth'], o['plane'], o['score']

    def pruned_depthmap_exists(self, image):
        return os.path.isfile(self._depthmap_file(image, 'pruned.npz'))

    def save_pruned_depthmap(self, image, points, track_ids, normals, colors, labels, detections):
        io.mkdir_p(self._depthmap_path())
        filepath = self._depthmap_file(image, 'pruned.npz')
        np.savez_compressed(filepath,
                            points=points, track_ids=track_ids,
                            normals=normals, colors=colors,
                            labels=labels, detections=detections)

    def load_pruned_depthmap(self, image):
        o = np.load(self._depthmap_file(image, 'pruned.npz'))
        if 'detections' not in o:
            return o['points'], o['track_ids'], o['normals'], o['colors'], o['labels'], np.zeros(o['labels'].shape)
        else:
            return o['points'], o['track_ids'], o['normals'], o['colors'], o['labels'], o['detections']

    def _is_image_file(self, filename):
        extensions = {'jpg', 'jpeg', 'png', 'tif', 'tiff', 'pgm', 'pnm', 'gif'}
        return filename.split('.')[-1].lower() in extensions

    def _set_image_path(self, path):
        """Set image path and find all images in there"""
        self.image_list = []
        self.image_files = {}
        if os.path.exists(path):
            for name in os.listdir(path):
                name = six.text_type(name)
                if self._is_image_file(name):
                    self.image_list.append(name)
                    self.image_files[name] = os.path.join(path, name)

    def _set_image_list(self, image_list):
        self.image_list = []
        self.image_files = {}
        for line in image_list:
            path = os.path.join(self.data_path, line)
            name = os.path.basename(path)
            self.image_list.append(name)
            self.image_files[name] = path

    def _set_mask_path(self, path):
        """Set mask path and find all masks in there"""
        self.mask_files = {}
        for image in self.images():
            filepath = os.path.join(path, image + '.png')
            if os.path.isfile(filepath):
                self.mask_files[image] = filepath

    def _set_mask_list(self, mask_list_lines):
        self.mask_files = {}
        for line in mask_list_lines:
            image, relpath = line.split(None, 1)
            path = os.path.join(self.data_path, relpath.strip())
            self.mask_files[image.strip()] = path

    def _exif_template_path(self):
        return self.data_path;

    def _exif_path(self):
        """Return path of extracted exif directory"""
        return os.path.join(self.data_path, 'exif')

    def _exif_file(self, image):
        """
        Return path of exif information for given image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self._exif_path(), image + '.exif')

    def get_exif_files(self):
        """
        Return file names for all exif files
        """
        exif_files = glob.glob(os.path.join(self._exif_path(), '*.exif') )
        
        return list( map( lambda x: os.path.basename(x), exif_files ) )

    def load_exif_template(self):
        
        tfile = os.path.join(self._exif_template_path(), 'template.exif')
        
        with io.open_rt( tfile ) as fin:
            return json.load(fin)

    def image_names_from_exif(self):
        exif_files = self.get_exif_files()
        return list( map( lambda x: os.path.splitext(x)[0], exif_files ) )
    
    def load_exif(self, image):
        """
        Return extracted exif information, as dictionary, usually with fields:

        ================  =====  ===================================
        Field             Type   Description
        ================  =====  ===================================
        width             int    Width of image, in pixels
        height            int    Height of image, in pixels
        focal_prior       float  Focal length (real) / sensor width
        ================  =====  ===================================

        :param image: Image name, with extension (i.e. 123.jpg)
        """
        with io.open_rt(self._exif_file(image)) as fin:
            return json.load(fin)

    def save_exif(self, image, data):
        io.mkdir_p(self._exif_path())
        with io.open_wt(self._exif_file(image)) as fout:
            io.json_dump(data, fout)

    def exif_exists(self, image):
        return os.path.isfile(self._exif_file(image))

    def feature_type(self):
        """Return the type of local features (e.g. AKAZE, SURF, SIFT)"""
        feature_name = self.config['feature_type'].lower()
        if self.config['feature_root']:
            feature_name = 'root_' + feature_name
        return feature_name

    def _feature_path(self):
        """Return path of feature descriptors and FLANN indices directory"""
        return os.path.join(self.data_path, "features")

    def _feature_file(self, image):
        """
        Return path of feature file for specified image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self._feature_path(), image + '.features.npz')

    def _feature_file_legacy(self, image):
        """
        Return path of a legacy feature file for specified image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self._feature_path(), image + '.npz')

    def _save_features(self, filepath, points, descriptors, colors=None):
        io.mkdir_p(self._feature_path())
        features.save_features(filepath, points, descriptors, colors, self.config)

    def features_exist(self, image):
        return os.path.isfile(self._feature_file(image)) or\
            os.path.isfile(self._feature_file_legacy(image))

    def load_features(self, image):
        if self.features_exist(image):
            if os.path.isfile(self._feature_file_legacy(image)):
                return features.load_features(self._feature_file_legacy(image), self.config)
            return features.load_features(self._feature_file(image), self.config)
        else:
            return None, None, None

    def save_features(self, image, points, descriptors, colors):
        self._save_features(self._feature_file(image), points, descriptors, colors)

    def feature_index_exists(self, image):
        return os.path.isfile(self._feature_index_file(image))

    def _feature_index_file(self, image):
        """
        Return path of FLANN index file for specified image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self._feature_path(), image + '.flann')

    def load_feature_index(self, image, features):
        if not self.feature_index_exists(image):
            raise IOError("FLANN index file for {} doesn't exists.".format(image))
        index = context.flann_Index()
        index.load(features, self._feature_index_file(image))
        return index

    def save_feature_index(self, image, index):
        index.save(self._feature_index_file(image))

    def _words_file(self, image):
        return os.path.join(self._feature_path(), image + '.words.npz')

    def words_exist(self, image):
        return os.path.isfile(self._words_file(image))

    def load_words(self, image):
        if self.words_exist(image):
            s = np.load(self._words_file(image))
            return s['words'].astype(np.int32)
        else:
            return None

    def save_words(self, image, words):
        np.savez_compressed(self._words_file(image), words=words.astype(np.uint16))

    def _matches_path(self):
        """Return path of matches directory"""
        return os.path.join(self.data_path, 'matches')

    def _matches_file(self, image):
        """File for matches for an image"""
        return os.path.join(self._matches_path(), '{}_matches.pkl.gz'.format(image))

    def matches_exists(self, image):
        return os.path.isfile(self._matches_file(image))

    def load_matches(self, image):
        with gzip.open(self._matches_file(image), 'rb') as fin:
            m = pickle.load(fin)

        matches = {}
        for image, image_matches in m.items():
            matches[image] = image_matches[1]

        return matches

    def save_matches(self, image, matches):
        io.mkdir_p(self._matches_path())
        with gzip.open(self._matches_file(image), 'wb') as fout:
            pickle.dump(matches, fout)

    def load_transforms(self, image):
        with gzip.open(self._matches_file(image), 'rb') as fin:
            m = pickle.load(fin)

        transforms = {}
        for image, image_transforms in m.items():
            transforms[image] = image_transforms[0]

        return transforms

    def find_matches(self, im1, im2):
        if self.matches_exists(im1):
            im1_matches = self.load_matches(im1)
            if im2 in im1_matches:
                return im1_matches[im2]
        if self.matches_exists(im2):
            im2_matches = self.load_matches(im2)
            if im1 in im2_matches:
                if len(im2_matches[im1]):
                    return im2_matches[im1][:, [1, 0]]
        return []

    def _tracks_graph_file(self, filename=None):
        """Return path of tracks file"""
        return os.path.join(self.data_path, filename or 'tracks.csv')

    def load_tracks_graph(self, filename=None):
        """Return graph (networkx data structure) of tracks"""
        with io.open_rt(self._tracks_graph_file(filename)) as fin:
            return tracking.load_tracks_graph(fin)

    def tracks_exists(self, filename=None):
        return os.path.isfile(self._tracks_graph_file(filename))

    def save_tracks_graph(self, graph, filename=None):
        with io.open_wt(self._tracks_graph_file(filename)) as fout:
            tracking.save_tracks_graph(fout, graph)

    def load_undistorted_tracks_graph(self):
        return self.load_tracks_graph('undistorted_tracks.csv')

    def save_undistorted_tracks_graph(self, graph):
        return self.save_tracks_graph(graph, 'undistorted_tracks.csv')

    def _densified_path(self):
        return os.path.join(self.data_path, 'densified')

    def _densified_reconstruction_file(self, recon_num):
        """Path to the densified recon file"""
        return os.path.join(self._densified_path(), 'densified_reconstruction_' + str(recon_num) + '.json')

    def _densified_tracks_graph_file(self, recon_num):
        """Path to the densified tracks file"""
        return os.path.join(self._densified_path(), 'densified_tracks_' + str(recon_num) + '.csv')

    def save_image_tracks(self, image, tracks):
        """Save densified tracks for each subshot image"""
        with io.open_wt(self._image_tracks_file(image, 'tracks.csv')) as fout:
            for track in tracks:
                fout.write(u'%s\t%s\t%d\t%g\t%g\t%g\t%g\t%g\t%g\n' % (
                    str(track[0]), str(track[1]), 0, float(track[2]), float(track[3]), 0.0004, 0, 0, 0))

    def save_densified_tracks_graph(self, recon_num, subshot_ids):
        io.mkdir_p(self._densified_path())
        with io.open_wt(self._densified_tracks_graph_file(recon_num)) as fout:
            tracking.save_tracks_graph_header(fout)
            for subshot_id in subshot_ids:
                with io.open_rt(self._image_tracks_file(subshot_id, 'tracks.csv')) as fin:
                    shutil.copyfileobj(fin, fout)

    def load_densified_tracks_graph(self, recon_num):
        """Return graph (networkx data structure) of densififed tracks"""
        with io.open_rt(self._densified_tracks_graph_file(recon_num)) as fin:
            return tracking.load_tracks_graph(fin)

    def densification_cleanup(self):
        shutil.rmtree(self._depthmap_path(), True)
        shutil.rmtree(self._undistorted_image_path(), True)
        if os.path.exists(self._reconstruction_file('undistorted_reconstruction.json')):
            os.remove(self._reconstruction_file('undistorted_reconstruction.json'))
        if os.path.exists(self._tracks_graph_file('undistorted_tracks.csv')):
            os.remove(self._tracks_graph_file('undistorted_tracks.csv'))

    def save_match_counts(self, matches, filename=None, minify=False ):
        with io.open_wt( self._match_counts_file(filename) ) as fout:
            io.json_dump(matches, fout, minify)
    
    def _match_counts_file(self, filename):
        """Return path of match counts json file"""
        return os.path.join(self.data_path, filename or 'match_counts.json')
        
    def _reconstruction_file(self, filename):
        """Return path of reconstruction file"""
        return os.path.join(self.data_path, filename or 'reconstruction.json')

    def reconstruction_exists(self, filename=None):
        return os.path.isfile(self._reconstruction_file(filename))

    def load_reconstruction(self, filename=None):
        with io.open_rt(self._reconstruction_file(filename)) as fin:
            reconstructions = io.reconstructions_from_json(io.json_load(fin))
        return reconstructions

    def save_reconstruction(self, reconstruction, filename=None, minify=False):
        with io.open_wt(self._reconstruction_file(filename)) as fout:
            io.json_dump(io.reconstructions_to_json(reconstruction), fout, minify)

    def save_reconstruction_no_point(self, reconstruction, minify=False):
        with io.open_wt(self._reconstruction_file('reconstruction_no_point.json')) as fout:
            objs = io.reconstructions_to_json(reconstruction)
            for obj in objs:
                obj['points'] = {}
            io.json_dump(objs, fout, minify)

    def _recon_quality_file(self):
        return os.path.join(self.data_path, 'recon_quality.txt')

    def save_recon_quality(self, recon_quality, avg_segment_size, ratio_shots_in_min_recon, speedup):
        """Save recon quality 0-100 to a file."""
        with io.open_wt(self._recon_quality_file()) as fout:
            fout.write("recon_quality_factor {0}\n".format(recon_quality))
            fout.write("avg_segment_size {0}\n".format(int(avg_segment_size)))
            fout.write("ratio_shots_in_min_recon {0}%\n".format(int(100*ratio_shots_in_min_recon)))
            fout.write("speedup {:2.1f}x\n".format(speedup))

    def load_undistorted_reconstruction(self):
        return self.load_reconstruction(
            filename='undistorted_reconstruction.json')

    def save_undistorted_reconstruction(self, reconstruction):
        self.save_reconstruction(reconstruction, filename='undistorted_reconstruction.json')

    def load_densified_reconstruction(self, recon_num):
        with io.open_rt(self._densified_reconstruction_file(recon_num)) as fin:
            reconstructions = io.reconstructions_from_json(io.json_load(fin))
        return reconstructions

    def save_densified_reconstruction(self, recon_num, reconstruction, minify=False):
        io.mkdir_p(self._densified_path())
        with io.open_wt(self._densified_reconstruction_file(recon_num)) as fout:
            io.json_dump(io.reconstructions_to_json([reconstruction]), fout, minify)

    def save_image_recon_num_dict(self, image_recon_num_dict):
        with io.open_wt(os.path.join(self._densified_path(), 'image_recon_num.json')) as fout:
            io.json_dump(image_recon_num_dict, fout, False)

    def get_recon_num(self, image):
        with io.open_rt(os.path.join(self._densified_path(), 'image_recon_num.json')) as fin:
            image_recon_num_dict = io.json_load(fin)

        if image in image_recon_num_dict:
            return image_recon_num_dict[image]
        else:
            return -1

    def get_recon_list(self):
        reconstructions = self.load_reconstruction()
        recon_list = []
        for recon_num, reconstruction in enumerate(reconstructions):
            if len(reconstruction.points) == 0:
                continue

            recon_list.append(recon_num)

        return recon_list

    def _reference_lla_path(self):
        return os.path.join(self.data_path, 'reference_lla.json')

    def invent_reference_lla(self, images=None):
        lat, lon, alt = 0.0, 0.0, 0.0
        wlat, wlon, walt = 0.0, 0.0, 0.0
        if images is None: images = self.images()
        for image in images:
            d = self.load_exif(image)
            if 'gps' in d and 'latitude' in d['gps'] and 'longitude' in d['gps']:
                w = 1.0 / max(0.01, d['gps'].get('dop', 15))
                lat += w * d['gps']['latitude']
                lon += w * d['gps']['longitude']
                wlat += w
                wlon += w
                if 'altitude' in d['gps']:
                    alt += w * d['gps']['altitude']
                    walt += w
        if wlat: lat /= wlat
        if wlon: lon /= wlon
        if walt: alt /= walt
        reference = {'latitude': lat, 'longitude': lon, 'altitude': 0}  # Set altitude manually.
        self.save_reference_lla(reference)
        return reference

    def save_reference_lla(self, reference):
        with io.open_wt(self._reference_lla_path()) as fout:
            io.json_dump(reference, fout)

    def load_reference_lla(self):
        with io.open_rt(self._reference_lla_path()) as fin:
            return io.json_load(fin)

    def load_reference(self):
        """Load reference as a topocentric converter."""
        lla = self.load_reference_lla()
        return geo.TopocentricConverter(
            lla['latitude'], lla['longitude'], lla['altitude'])

    def reference_lla_exists(self):
        return os.path.isfile(self._reference_lla_path())

    def _camera_models_file(self):
        """Return path of camera model file"""
        return os.path.join(self.data_path, 'camera_models.json')

    def load_camera_models(self):
        """Return camera models data"""
        with io.open_rt(self._camera_models_file()) as fin:
            obj = json.load(fin)
            return io.cameras_from_json(obj)

    def save_camera_models(self, camera_models):
        """Save camera models data"""
        with io.open_wt(self._camera_models_file()) as fout:
            obj = io.cameras_to_json(camera_models)
            io.json_dump(obj, fout)

    def _camera_models_overrides_file(self):
        """Path to the camera model overrides file."""
        return os.path.join(self.data_path, 'camera_models_overrides.json')

    def camera_models_overrides_exists(self):
        """Check if camera overrides file exists."""
        return os.path.isfile(self._camera_models_overrides_file())

    def load_camera_models_overrides(self):
        """Load camera models overrides data."""
        with io.open_rt(self._camera_models_overrides_file()) as fin:
            obj = json.load(fin)
            return io.cameras_from_json(obj)

    def save_camera_models_overrides(self, camera_models):
        """Save camera models overrides data"""
        with io.open_wt(self._camera_models_overrides_file()) as fout:
            obj = io.cameras_to_json(camera_models)
            io.json_dump(obj, fout)

    def _exif_overrides_file(self):
        """Path to the EXIF overrides file."""
        return os.path.join(self.data_path, 'exif_overrides.json')

    def exif_overrides_exists(self):
        """Check if EXIF overrides file exists."""
        return os.path.isfile(self._exif_overrides_file())

    def load_exif_overrides(self):
        """Load EXIF overrides data."""
        with io.open_rt(self._exif_overrides_file()) as fin:
            return json.load(fin)

    def profile_log(self):
        "Filename where to write timings."
        return os.path.join(self.data_path, 'profile.log')

    def _report_path(self):
        return os.path.join(self.data_path, 'reports')

    def load_report(self, path):
        """Load a report file as a string."""
        with io.open_rt(os.path.join(self._report_path(), path)) as fin:
            return fin.read()

    def save_report(self, report_str, path):
        """Save report string to a file."""
        filepath = os.path.join(self._report_path(), path)
        io.mkdir_p(os.path.dirname(filepath))
        with io.open_wt(filepath) as fout:
            return fout.write(report_str)

    def _navigation_graph_file(self):
        "Return the path of the navigation graph."
        return os.path.join(self.data_path, 'navigation_graph.json')

    def save_navigation_graph(self, navigation_graphs):
        with io.open_wt(self._navigation_graph_file()) as fout:
            io.json_dump(navigation_graphs, fout)

    def _ply_file(self, filename):
        return os.path.join(self.data_path, filename or 'reconstruction.ply')

    def save_ply(self, reconstruction, filename=None,
                 no_cameras=False, no_points=False):
        """Save a reconstruction in PLY format."""
        ply = io.reconstruction_to_ply(reconstruction, no_cameras, no_points)
        with io.open_wt(self._ply_file(filename)) as fout:
            fout.write(ply)

    def _gps_points_file(self):
        return os.path.join(self.data_path, 'gps_list.txt')        
        
    def gps_points_exist(self):
        return os.path.isfile(self._gps_points_file())
    
    def load_gps_points(self):
        """Load explicit points as opposed to GPS derived from EXIF metadata"""

        if not self.gps_points_dict:
            with open(self._gps_points_file()) as fin:
                self.gps_points_dict = io.read_gps_points_list(fin)

        return self.gps_points_dict

    def load_topocentric_gps_points(self):
        return self.topocentric_gps_points_dict

    def save_topocentric_gps_points(self, topocentric_gps_points_dict):
        """save a local copy of topocentric gps points"""
        self.topocentric_gps_points_dict = topocentric_gps_points_dict

    def _culling_dict_file(self):
        return os.path.join(os.path.dirname(self.data_path), 'frames_redundant/undo.txt')

    def culling_dict_exist(self):
        return os.path.isfile(self._culling_dict_file())

    def load_culling_dict(self):
        culling_dict = {}
        if self.culling_dict_exist():
            with open(self._culling_dict_file()) as fin:
                for line in fin:
                    (new_shot_id, orig_shot_id) = line.split()
                    culling_dict[new_shot_id] = orig_shot_id
        return culling_dict

    def _pdr_shots_file(self):
        return os.path.join(self.data_path, 'pdr_shots.txt')

    def pdr_shots_exist(self):
        return os.path.isfile(self._pdr_shots_file())

    def load_pdr_shots(self):
        """
        Load PDR shots
        """

        if not self.pdr_shots_dict:
            with open(self._pdr_shots_file()) as fin:
                for line in fin:
                    (shot_id, x, y, z, roll, pitch, heading, omega_0, omega_1, omega_2, delta_distance) = line.split()
                    self.pdr_shots_dict[shot_id] = (float(x), float(y), float(z),
                                                    float(roll), float(pitch), float(heading),
                                                    float(omega_0), float(omega_1), float(omega_2),
                                                    float(delta_distance))

        return self.pdr_shots_dict

    def load_pdr_predictions(self):
        return self.pdr_predictions_dict

    def save_pdr_predictions(self, pdr_predictions_dict):
        """save a copy of aligned pdr shots"""
        self.pdr_predictions_dict = pdr_predictions_dict

    def _ground_control_points_file(self):
        return os.path.join(self.data_path, 'ground_control_points.json')

    def ground_control_points_exist(self):
        return os.path.isfile(self._ground_control_points_file())

    def _gcp_list_file(self):
        return os.path.join(self.data_path, 'gcp_list.txt')

    def load_ground_control_points(self):
        """Load ground control points.

        It uses reference_lla to convert the coordinates
        to topocentric reference frame.
        """
        exif = {image: self.load_exif(image) for image in self.images()}
        reference = self.load_reference()

        gcp = []
        if os.path.isfile(self._gcp_list_file()):
            with io.open_rt(self._gcp_list_file()) as fin:
                gcp = io.read_gcp_list(fin, reference, exif)

        pcs = []
        if os.path.isfile(self._ground_control_points_file()):
            with io.open_rt(self._ground_control_points_file()) as fin:
                pcs = io.read_ground_control_points(fin, reference)

        return gcp + pcs

    def _hlf_list_file(self):
        return os.path.join(self.data_path, 'hlf_list.txt')

    def hlf_list_exist(self):
        return os.path.isfile(self._hlf_list_file())

    def load_hlf_list(self):
        """
        Load high level features list
        """
        if not self.hlf_list:
            with open(self._hlf_list_file()) as fin:
                next(fin)
                for line in fin:
                    (x, y, type) = line.split()
                    self.hlf_list.append([float(x), float(y)])

        return self.hlf_list

    def _hlf_det_list_file(self):
        return os.path.join(self.data_path, 'tp_doors.txt')

    def hlf_det_list_exist(self):
        return os.path.isfile(self._hlf_det_list_file())

    def load_hlf_det_list(self):
        """
        Load list of images where high level feature is detected
        """
        if not self.hlf_det_list:
            with open(self._hlf_det_list_file()) as fin:
                for line in fin:
                    self.hlf_det_list.append(line.rstrip('\n'))

        return self.hlf_det_list

    def image_as_array(self, image):
        logger.warning("image_as_array() is deprecated. Use load_image() instead.")
        return self.load_image(image)

    def undistorted_image_as_array(self, image):
        logger.warning("undistorted_image_as_array() is deprecated. "
                       "Use load_undistorted_image() instead.")
        return self.load_undistorted_image(image)

    def mask_as_array(self, image):
        logger.warning("mask_as_array() is deprecated. Use load_mask() instead.")
        return self.load_mask(image)
