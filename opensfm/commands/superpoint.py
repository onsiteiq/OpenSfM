# superpoints implemented on Torch for all resolution images
# It also works with single images as well
# does not require GPU for feature extraction
# By: Mohammad Nahangi- June 2019
# For ONSITEIQ to improve feature detection and matching

# TODO: Torch is required. Installation command to be added to the Dockerfile!


import argparse
import glob
import numpy as np
import os
import time
import math

import cv2
import torch

from opensfm import types
from opensfm.commands import undistort
from opensfm.context import parallel_map

# global variables
size_w = 1280
size_h = 640

display = False
show_extra = False
win = None

write = True
write_dir = './osfm/superpoints'
input_dir = './frames'
weights_file = './data/superpoint.pth'
img_glob = '*.jpg'

skip = 1
display_scale = 1
min_length = 2
max_length = 5
nms_dist = 4
conf_thresh = 0.15
nn_thresh = 0.5
waitkey = 1
cuda = False

# Font parameters for visualizaton.
font = cv2.FONT_HERSHEY_DUPLEX
font_clr = (255, 255, 255)
font_pt = (4, 10)
font_sc = 0.4


# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  # training is not used. it is only coded for future use.
  def __init__(self):
    super(SuperPointNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)
    # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return semi, desc


class SuperPointFrontend(object):
  """ Wrapper around pytorch net to help with pre and post image processing. """
  def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
               cuda=False):
    self.name = 'SuperPoint'
    self.cuda = cuda
    self.nms_dist = nms_dist
    self.conf_thresh = conf_thresh
    self.nn_thresh = nn_thresh # L2 descriptor distance for good match.
    self.cell = 8 # Size of each output cell. Keep this fixed.
    self.border_remove = 4 # Remove points this close to the border.

    # Load the network in inference mode.
    self.net = SuperPointNet()
    if cuda:
      # Train on GPU, deploy on GPU.
      # for superpoints deployed on ONSITEIQ trianing is not relevant.
      # We are using a pretrained network
      self.net.load_state_dict(torch.load(weights_path))
      self.net = self.net.cuda()
    else:
      # Train on GPU, deploy on CPU.
      self.net.load_state_dict(torch.load(weights_path,
                               map_location=lambda storage, loc: storage))
    self.net.eval()

  def nms_fast(self, in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
  
    Algorithm summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
  
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
  
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
  
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

  def run(self, img):
    """ Process a numpy image to extract points and descriptors.
    Input
      img - HxW numpy float32 input image in range [0,1].
    Output
      corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      desc - 256xN numpy array of corresponding unit normalized descriptors.
      heatmap - HxW numpy heatmap in range [0,1] of point confidences.
      """
    assert img.ndim == 2, 'Image must be grayscale.'
    assert img.dtype == np.float32, 'Image must be float32.'
    H, W = img.shape[0], img.shape[1]
    inp = img.copy()
    inp = (inp.reshape(1, H, W))
    inp = torch.from_numpy(inp)
    inp = torch.autograd.Variable(inp).view(1, 1, H, W)
    if self.cuda:
      inp = inp.cuda()
    # Forward pass of network.
    outs = self.net.forward(inp)
    semi, coarse_desc = outs[0], outs[1]
    # Convert pytorch -> numpy.
    semi = semi.data.cpu().numpy().squeeze()
    # --- Process points.
    dense = np.exp(semi) # Softmax.
    dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
    # Remove dustbin.
    nodust = dense[:-1, :, :]
    # Reshape to get full resolution heatmap.
    Hc = int(H / self.cell)
    Wc = int(W / self.cell)
    nodust = nodust.transpose(1, 2, 0)
    heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
    xs, ys = np.where(heatmap >= self.conf_thresh) # Confidence threshold.
    if len(xs) == 0:
      return np.zeros((3, 0)), None, None
    pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist) # Apply NMS.
    inds = np.argsort(pts[2,:])
    pts = pts[:,inds[::-1]] # Sort by confidence.
    # Remove points along border.
    bord = self.border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    # --- Process descriptor.
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
      desc = np.zeros((D, 0))
    else:
      # Interpolate into descriptor map using 2D point locations.
      samp_pts = torch.from_numpy(pts[:2, :].copy())
      samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
      samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
      samp_pts = samp_pts.transpose(0, 1).contiguous()
      samp_pts = samp_pts.view(1, 1, -1, 2)
      samp_pts = samp_pts.float()
      if self.cuda:
        samp_pts = samp_pts.cuda()
      desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
      desc = desc.data.cpu().numpy().reshape(D, -1)
      desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

    # reduce dimensions of each descriptor from 256 to 128 (to match other feature types)
    even_rows = [x for x in range(256) if x % 2 == 0]
    return pts, desc[even_rows, :], heatmap


class PointTracker(object):
  """ Class to manage a fixed memory of points and descriptors that enables
  sparse optical flow point tracking.

  Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
  tracks with maximum length L, where each row corresponds to:
  row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
  """

  def __init__(self, max_length, nn_thresh):
    if max_length < 2:
      raise ValueError('max_length must be greater than or equal to 2.')
    self.maxl = max_length
    self.nn_thresh = nn_thresh
    self.all_pts = []
    for n in range(self.maxl):
      self.all_pts.append(np.zeros((2, 0)))
    self.last_desc = None
    self.tracks = np.zeros((0, self.maxl+2))
    self.track_count = 0
    self.max_score = 9999

  def nn_match_two_way(self, desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches

  def get_offsets(self):
    """ Iterate through list of points and accumulate an offset value. Used to
    index the global point IDs into the list of points.

    Returns
      offsets - N length array with integer offset locations.
    """
    # Compute id offsets.
    offsets = []
    offsets.append(0)
    for i in range(len(self.all_pts)-1): # Skip last camera size, not needed.
      offsets.append(self.all_pts[i].shape[1])
    offsets = np.array(offsets)
    offsets = np.cumsum(offsets)
    return offsets

  def update(self, pts, desc):
    """ Add a new set of point and descriptor observations to the tracker.

    Inputs
      pts - 3xN numpy array of 2D point observations.
      desc - DxN numpy array of corresponding D dimensional descriptors.
    """
    if pts is None or desc is None:
      print('PointTracker: Warning, no points were added to tracker.')
      return
    assert pts.shape[1] == desc.shape[1]
    # Initialize last_desc.
    if self.last_desc is None:
      self.last_desc = np.zeros((desc.shape[0], 0))
    # Remove oldest points, store its size to update ids later.
    remove_size = self.all_pts[0].shape[1]
    self.all_pts.pop(0)
    self.all_pts.append(pts)
    # Remove oldest point in track.
    self.tracks = np.delete(self.tracks, 2, axis=1)
    # Update track offsets.
    for i in range(2, self.tracks.shape[1]):
      self.tracks[:, i] -= remove_size
    self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
    offsets = self.get_offsets()
    # Add a new -1 column.
    self.tracks = np.hstack((self.tracks, -1*np.ones((self.tracks.shape[0], 1))))
    # Try to append to existing tracks.
    matched = np.zeros((pts.shape[1])).astype(bool)
    matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
    for match in matches.T:
      # Add a new point to it's matched track.
      id1 = int(match[0]) + offsets[-2]
      id2 = int(match[1]) + offsets[-1]
      found = np.argwhere(self.tracks[:, -2] == id1)
      if found.shape[0] > 0:
        matched[int(match[1])] = True
        row = int(found)
        self.tracks[row, -1] = id2
        if self.tracks[row, 1] == self.max_score:
          # Initialize track score.
          self.tracks[row, 1] = match[2]
        else:
          # Update track score with running average.
          # NOTE(dd): this running average can contain scores from old matches
          #           not contained in last max_length track points.
          track_len = (self.tracks[row, 2:] != -1).sum() - 1.
          frac = 1. / float(track_len)
          self.tracks[row, 1] = (1.-frac)*self.tracks[row, 1] + frac*match[2]
    # Add unmatched tracks.
    new_ids = np.arange(pts.shape[1]) + offsets[-1]
    new_ids = new_ids[~matched]
    new_tracks = -1*np.ones((new_ids.shape[0], self.maxl + 2))
    new_tracks[:, -1] = new_ids
    new_num = new_ids.shape[0]
    new_trackids = self.track_count + np.arange(new_num)
    new_tracks[:, 0] = new_trackids
    new_tracks[:, 1] = self.max_score*np.ones(new_ids.shape[0])
    self.tracks = np.vstack((self.tracks, new_tracks))
    self.track_count += new_num # Update the track count.
    # Remove empty tracks.
    keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
    self.tracks = self.tracks[keep_rows, :]
    # Store the last descriptors.
    self.last_desc = desc.copy()
    return

  def get_tracks(self, min_length):
    """ Retrieve point tracks of a given minimum length.
    Input
      min_length - integer >= 1 with minimum track length
    Output
      returned_tracks - M x (2+L) sized matrix storing track indices, where
        M is the number of tracks and L is the maximum track length.
    """
    if min_length < 1:
      raise ValueError('\'min_length\' too small.')
    valid = np.ones((self.tracks.shape[0])).astype(bool)
    good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
    # Remove tracks which do not have an observation in most recent frame.
    not_headless = (self.tracks[:, -1] != -1)
    keepers = np.logical_and.reduce((valid, good_len, not_headless))
    returned_tracks = self.tracks[keepers, :].copy()
    return returned_tracks

  def draw_tracks(self, out, tracks):
    """ Visualize tracks all overlayed on a single image.
    Inputs
      out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
      tracks - M x (2+L) sized matrix storing track info.
    """
    # Store the number of points per camera.
    pts_mem = self.all_pts
    N = len(pts_mem) # Number of cameras/images.
    # Get offset ids needed to reference into pts_mem.
    offsets = self.get_offsets()
    # Width of track and point circles to be drawn.
    stroke = 1
    # Iterate through each track and draw it.
    for track in tracks:
      clr = myjet[int(np.clip(np.floor(track[1]*10), 0, 9)), :]*255
      for i in range(N-1):
        if track[i+2] == -1 or track[i+3] == -1:
          continue
        offset1 = offsets[i]
        offset2 = offsets[i+1]
        idx1 = int(track[i+2]-offset1)
        idx2 = int(track[i+3]-offset2)
        pt1 = pts_mem[i][:2, idx1]
        pt2 = pts_mem[i+1][:2, idx2]
        p1 = (int(round(pt1[0])), int(round(pt1[1])))
        p2 = (int(round(pt2[0])), int(round(pt2[1])))
        cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
        # Draw end points of each track.
        if i == N-2:
          clr2 = (255, 0, 0)
          cv2.circle(out, p2, stroke, clr2, -1, lineType=16)

class VideoStreamer(object):
  """ Class to help process image streams. Three types of possible inputs:"
    1.) USB Webcam.
    2.) A directory of images (files in directory matching 'img_glob').
    3.) A video file, such as an .mp4 or .avi file.
  """
  def __init__(self, basedir, height, width, skip, img_glob):
    self.listing = []
    self.sizer = [height, width]
    self.skip = skip
    self.maxlen = 1000000
    print('==> Processing Image Directory Input.')
    search = os.path.join(basedir, img_glob)
    self.listing = glob.glob(search)
    self.listing.sort()
    self.listing = self.listing[::self.skip]
    self.maxlen = len(self.listing)
    if self.maxlen == 0:
      raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')

  def read_image(self, impath, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    img = cv2.imread(impath)
    if img is None:
      raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=interp)

    # unfold img
    undist_img = self.convert_image(impath[-14:], img, img_size[1])

    grayim = cv2.cvtColor(undist_img, cv2.COLOR_BGR2GRAY)
    grayim = (grayim.astype('float32') / 255.)
    return undist_img, grayim

  def convert_image(self, image, img, max_size):
    image_camera_model = types.SphericalCamera()
    image_camera_model.id = "v2 nctech pulsar 11000 5500 equirectangular 0.1666"
    image_camera_model.width = 11000
    image_camera_model.height = 5500

    undist_tile_size = max_size // 4

    undist_img = np.zeros((max_size // 2, max_size, 3), np.uint8)
    undist_mask = np.full((max_size // 2, max_size, 1), 255, np.uint8)

    undist_mask[undist_tile_size:2 * undist_tile_size, 2 * undist_tile_size:3 * undist_tile_size] = 0
    undist_mask[undist_tile_size:2 * undist_tile_size, undist_tile_size:2 * undist_tile_size] = 0

    spherical_shot = types.Shot()
    spherical_shot.pose = types.Pose()
    spherical_shot.id = image
    spherical_shot.camera = image_camera_model

    perspective_shots = undistort.perspective_views_of_a_panorama(spherical_shot, undist_tile_size)

    for subshot in perspective_shots:

      undistorted = undistort.render_perspective_view_of_a_panorama(img, spherical_shot, subshot)

      subshot_id_prefix = '{}_perspective_view_'.format(spherical_shot.id)

      subshot_name = subshot.id[len(subshot_id_prefix):] if subshot.id.startswith(subshot_id_prefix) else subshot.id
      (subshot_name, ext) = os.path.splitext(subshot_name)

      if subshot_name == 'front':
        undist_img[:undist_tile_size, :undist_tile_size] = undistorted
        # print( 'front')
      elif subshot_name == 'left':
        undist_img[:undist_tile_size, undist_tile_size:2 * undist_tile_size] = undistorted
        # print( 'left')
      elif subshot_name == 'back':
        undist_img[:undist_tile_size, 2 * undist_tile_size:3 * undist_tile_size] = undistorted
        # print( 'back')
      elif subshot_name == 'right':
        undist_img[:undist_tile_size, 3 * undist_tile_size:4 * undist_tile_size] = undistorted
        # print( 'right')
      elif subshot_name == 'top':
        undist_img[undist_tile_size:2 * undist_tile_size, 3 * undist_tile_size:4 * undist_tile_size] = undistorted
        # print( 'top')
      elif subshot_name == 'bottom':
        undist_img[undist_tile_size:2 * undist_tile_size, :undist_tile_size] = undistorted
        # print( 'bottom')

    # data.save_undistorted_image(subshot.id, undist_img)
    return undist_img

  def convert_points(self, p_unsorted, f_unsorted, c_unsorted):
    image_camera_model = types.SphericalCamera()
    image_camera_model.id = "v2 nctech pulsar 11000 5500 equirectangular 0.1666"
    image_camera_model.width = 11000
    image_camera_model.height = 5500

    if len(p_unsorted) > 0:
      # Mask pixels that are out of valid image bounds before converting to equirectangular image coordinates

      bearings = image_camera_model.unfolded_pixel_bearings(p_unsorted[:, :2])

      p_mask = np.array([point is not None for point in bearings])

      p_unsorted = p_unsorted[p_mask]
      f_unsorted = f_unsorted[p_mask]
      c_unsorted = c_unsorted[p_mask]

      p_unsorted[:, :2] = self.unfolded_cube_to_equi_normalized_image_coordinates(p_unsorted[:, :2], image_camera_model)
      return p_unsorted, f_unsorted, c_unsorted

  def unfolded_cube_to_equi_normalized_image_coordinates(self, pixel_coords, image_camera_model):

    bearings = image_camera_model.unfolded_pixel_bearings(pixel_coords)

    norm_pix_x, norm_pix_y = image_camera_model.project((bearings[:, 0], bearings[:, 1], bearings[:, 2]))

    norm_pixels = np.column_stack([norm_pix_x.ravel(), norm_pix_y.ravel()])

    return norm_pixels

  def get_frame(self, idx):
    """ Return frame with index idx
    Returns
       image: H x W image.
       status: True or False depending whether image was loaded.
    """
    if idx >= self.maxlen:
      return (None, False)

    image_file = self.listing[idx]
    input_image, gray_image = self.read_image(image_file, self.sizer)
    gray_image = gray_image.astype('float32')
    return (input_image, gray_image, True)


def mag_and_dir(grayim, x, y):
  '''
  :param grayim: input image in grayscale
  :param x: x-coord of the feature
  :param y: y-coord of the feature
  :return: size and angle = magnitude and orientation of the feature
           based on Lowe's paper for SIFT features
  '''
  l = cv2.Laplacian(grayim, cv2.CV_64FC1)
  del_y = l[x, y+1] - l[x, y-1]
  del_x = l[x+1, y] - l[x-1, y]

  size = math.sqrt(del_x**2 + del_y **2)

  if del_x == 0:
    angle = 90
  else:
    angle = math.atan(del_y / del_x)
    angle  = math.degrees(angle)

  return size, angle


def save_features(feature_outfile, points, desc, colors):
  np.savez_compressed(feature_outfile,
           points=points.astype(np.float32),
           descriptors=desc.astype(np.float32),
           colors=colors)


def load_features(image):
  filepath = './osfm/superpoints'

  try:
    s = np.load(os.path.join(filepath, image + '.npz'))
    return s['points'], s['descriptors'], s['colors'].astype(float)
  except FileNotFoundError:
    return None, None, None


def save_feature_index(flann_outfile, desc):
  flann_params = dict(algorithm=2,
                      branching=16,
                      iterations=10)

  index = cv2.flann_Index(desc, flann_params)
  index.save(flann_outfile)


def load_feature_index(image, features):
  filepath = './osfm/superpoints'

  index = cv2.flann_Index()
  index.load(features, os.path.join(filepath, image + '.flann'))
  return index


def remove_border_points(img, pts, desc, border_size=5):
  # find the first row in which all pixels are black
  idx = img.shape[0]
  for i in range(img.shape[0]):
    if np.all(img[i][:] == [0, 0, 0]):
      idx = i
      break

  # then remove the sp's close to the corders
  selection = (idx - pts[1] > border_size)
  return pts[:, selection], desc[:, selection]


def normalized_image_coordinates(pixel_coords, width, height):
  size = max(width, height)
  p = np.empty((len(pixel_coords), 2))
  p[:, 0] = (pixel_coords[:, 0] + 0.5 - width / 2.0) / size
  p[:, 1] = (pixel_coords[:, 1] + 0.5 - height / 2.0) / size

  return p.T

def denormalized_image_coordinates(norm_coords, width, height):
  size = max(width, height)
  p = np.empty((len(norm_coords), 2))
  p[:, 0] = norm_coords[:, 0] * size - 0.5 + width / 2.0
  p[:, 1] = norm_coords[:, 1] * size - 0.5 + height / 2.0

  return p.T


def detect(args):
    idx, vs, fe, tracker = args

    start = time.time()
    # get the filename to be used for filepath to store the features.
    fname = vs.listing[idx][-14:]

    # Get a new image.
    img, grayimg, status = vs.get_frame(idx)
    if status is False:
        return

    # Get points and descriptors.
    pts, desc, heatmap = fe.run(grayimg)

    if pts is None or desc is None:
        return

    pts, desc = remove_border_points(img, pts, desc, border_size=6)

    # Add points and descriptors to the tracker.
    tracker.update(pts, desc)

    norm_pts = normalized_image_coordinates(pts.T, img.shape[1], img.shape[0])

    # get the color at the feature extracted to be stored for matching
    colors = [img[int(round(pt[1])), int(round(pt[0]))] for pt in pts.T]
    colors = np.array(colors)

    # compute magnitude and direction for feature matching using Opensfm
    # grayimg is normalized to 0-1 and laplacian will return an error if used.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mag = [mag_and_dir(gray, int(round(pt[1])), int(round(pt[0])))[0] for pt in pts.T]
    mag = np.array(mag)
    dir = [mag_and_dir(gray, int(round(pt[1])), int(round(pt[0])))[1] for pt in pts.T]
    dir = np.array(dir)

    # adding mag and dir to the pts(x,y)
    points = []
    for i in range(len(mag)):
        points.append([norm_pts.T[i][0], norm_pts.T[i][1], mag[i], dir[i]])
    #points = np.array(points, dtype=np.uint8)
    points = np.array(points)

    desc = desc.T
    points, desc, colors = vs.convert_points(points, desc, colors)

    end1 = time.time()

    # Get tracks for points which were match successfully across all frames.
    tracks = tracker.get_tracks(min_length)

    # Primary output - Show point tracks overlayed on top of input image.
    out1 = img.copy()

    for pt in pts.T:
        pt1 = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(out1, pt1, 1, (0, 255, 0), -1, lineType=16)
    # cv2.putText(out1, 'Raw Point Detections', font_pt, font, font_sc, font_clr, lineType=16)

    # Extra output -- Show current point detections.
    out2 = img.copy()
    tracks[:, 1] /= float(fe.nn_thresh)  # Normalize track scores to [0,1].
    tracker.draw_tracks(out2, tracks)
    if show_extra:
        cv2.putText(out2, 'Point Tracks', font_pt, font, font_sc, font_clr, lineType=16)

    # Extra output -- Show the point confidence heatmap.
    if heatmap is not None:
        min_conf = 0.001
        heatmap[heatmap < min_conf] = min_conf
        heatmap = -np.log(heatmap)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
        out3 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
        out3 = (out3*255).astype('uint8')
    else:
        out3 = np.zeros_like(out2)
    cv2.putText(out3, 'Raw Point Confidences', font_pt, font, font_sc, font_clr, lineType=16)

    # Resize final output.
    if show_extra:
        out = np.hstack((out1, out2, out3))
        out = cv2.resize(out, (3*display_scale*size_w, display_scale*size_h))
    else:
        out = cv2.resize(out1, (display_scale*size_w, display_scale*size_h))

    # Display visualization image to screen.
    if display:
        cv2.imshow(win, out)
        #key = cv2.waitKey(waitkey) & 0xFF
        #if key == ord('q'):
            #print('Quitting, \'q\' pressed.')
            #break

    # save the features
    if write:
        npz_file = fname + '.npz'
        npz_outfile = os.path.join(write_dir, npz_file)
        save_features(npz_outfile, points, desc, colors)
        print('Saving features to %s' % npz_outfile)

        #preemptive_npz_file = fname + '_preemptive' + '.npz'
        #preemptive_npz_outfile = os.path.join(write_dir, preemptive_npz_file)
        #save_features(preemptive_npz_outfile, points[-200:], desc[-200:], None)

        #idx_file = fname + '.flann'
        #idx_outfile = os.path.join(write_dir, idx_file)
        #save_feature_index(idx_outfile, desc)
        #print('Saving feature index to %s' % idx_outfile)

        # uncomment the line below to save annotated frames w superpoints
        out_file = os.path.join(write_dir, fname)
        cv2.imwrite(out_file, out)

    end = time.time()
    net_t = (1./ float(end1 - start))
    total_t = (1./ float(end - start))
    if show_extra:
        print('Processed image %d (net+post_process: %.2f FPS, total: %.2f FPS).'\
              % (vs.i, net_t, total_t))


def gen_ss(W, processes):
  size_w = W
  size_h = W//2

  # This class helps load input images from different sources.
  vs = VideoStreamer(input_dir, size_h, size_w, skip, img_glob)

  print('==> Loading pre-trained network.')
  current_dir = os.path.dirname(__file__)
  weights_path = os.path.join(os.path.dirname(current_dir), weights_file)

  # This class runs the SuperPoint network and processes its outputs.
  fe = SuperPointFrontend(weights_path=weights_path,
                          nms_dist=nms_dist,
                          conf_thresh=conf_thresh,
                          nn_thresh=nn_thresh,
                          cuda=cuda)
  print('==> Successfully loaded pre-trained network.')

  # This class helps merge consecutive point matches into tracks.
  tracker = PointTracker(max_length, nn_thresh=fe.nn_thresh)

  # Create a window to display the demo.
  if display:
    win = 'SuperPoint Tracker'
    cv2.namedWindow(win)
  else:
    print('Skipping visualization, will not show a GUI.')

  # Create output directory if desired.

  if write:
    print('==> Will write outputs to %s' % write_dir)
    if not os.path.exists(write_dir):
      os.makedirs(write_dir)

  print('==> Running Demo.')
  arguments = [(idx, vs, fe, tracker) for idx in range(len(vs.listing))]
  parallel_map(detect, arguments, processes)

  # Close any remaining windows.
  cv2.destroyAllWindows()

  print('.. finished Extracting Super Points.')


if __name__ == '__main__':
  gen_ss(1280, 10)

