from nd2reader import ND2Reader
import itk
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import clear_output
import pandas as pd
from joblib import Parallel, delayed
from skimage import exposure, util, feature, draw
from scipy import ndimage
import tifffile
from glob import glob
from matplotlib.patches import Polygon
from matplotlib.path import Path
plt.rcParams['font.sans-serif'] = 'Arial'

name_from_path = lambda name: os.path.splitext(os.path.basename(name))[0]

def make_scratch_dir(input_nd2, scratch_path):
	exp_name = name_from_path(input_nd2)
	scratch_dir = os.path.join(scratch_path, exp_name + '-scratch')
	os.makedirs(scratch_dir, exist_ok=True)
	return scratch_dir

def make_rgb(rfp, gfp):
	rgb = np.zeros((*rfp.shape, 3))
	rgb[..., 0] = exposure.equalize_hist(rfp)
	rgb[..., 1] = exposure.equalize_hist(gfp)
	return util.img_as_ubyte(rgb)

def align_cameras(input_nd2, scratch_dir, plot=True, save=True):
	with ND2Reader(input_nd2) as f:
		rfp, gfp = f.get_frame_2D(0, 0), f.get_frame_2D(1, 0)
	parameter_object = itk.ParameterObject.New()
	default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
	parameter_object.AddParameterMap(default_rigid_parameter_map)
	gfp_registered, params = itk.elastix_registration_method(rfp, gfp, parameter_object=parameter_object)
	rgb, rgb_registered = make_rgb(rfp, gfp), make_rgb(rfp, gfp_registered)
	if plot:
		plt.figure(figsize=(10, 5))
		plt.subplot(121); plt.title('Raw')
		plt.imshow(rgb); plt.axis('off')
		plt.subplot(122); plt.title('Registered')
		plt.imshow(rgb_registered); plt.axis('off')
		plt.tight_layout()
	if save:
		exp_name = name_from_path(input_nd2)
		patch_path = os.path.join(scratch_dir, 'patch.tif')
		tifffile.imwrite(patch_path, rgb_registered)
		param_path = os.path.join(scratch_dir, 'params.txt')
		itk.ParameterObject.WriteParameterFile(params, param_path)
	return params

def make_tif(input_nd2, scratch_dir, params):
	tif_path = os.path.join(scratch_dir, 'registered.tif')
	if os.path.isfile(tif_path):
		print('File exists!')
		return tif_path
	with ND2Reader(input_nd2) as f, tifffile.TiffWriter(tif_path, bigtiff=True, imagej=True) as tif:
		max_frame = f.metadata['num_frames']
		for i in range(1, max_frame):
			rfp, gfp = f.get_frame_2D(0, i), f.get_frame_2D(1, i)
			gfp_registered = itk.transformix_filter(gfp, params)
			tif.write(np.array([rfp, gfp_registered]), contiguous=True)
			clear_output(wait=True)
			print('%i/%i' % (i, max_frame - 1))
	return tif_path

def get_max_frame(tif_path):
	with tifffile.TiffFile(tif_path) as tif:
		return len(tif.pages) // 2

def get_frame(tif_path, index):
	with tifffile.TiffFile(tif_path) as tif:
		return tif.pages[2 * index].asarray(), tif.pages[2 * index + 1].asarray()

def make_mask(scratch_dir, fps=10):
	tif_path = os.path.join(scratch_dir, 'registered.tif')
	max_t = get_max_frame(tif_path)
	arr = np.zeros((max_t // fps, 2, 1024, 1024), np.uint16)
	for i in range(max_t // fps):
		arr[i] = get_frame(tif_path, i * fps)
	mask = np.median(arr, axis=0)
	mask_path = os.path.join(scratch_dir, 'mask.tif')
	tifffile.imwrite(mask_path, mask)
	return mask

def get_blobs(frame, mask, sigma, thresh, inner_rad, outer_rad):
	#output columns are ['x', 'y', 'laplace', 'inner_gfp/px', 'outer_gfp/px', 'inner_rfp/px', 'outer_rfp/px', 'ratio']
	rfp_frame, gfp_frame = frame
	rfp_mask = mask[0]
	masked_frame = (rfp_frame / np.mean(rfp_frame) * np.mean(rfp_mask)) - rfp_mask
	laplace_frame = -ndimage.gaussian_laplace(masked_frame, sigma=sigma)
	blobs = feature.peak_local_max(laplace_frame, threshold_abs=thresh)
	out_arr = np.zeros((len(blobs), 8))
	for i in range(len(blobs)):
		x, y = blobs[i, 0], blobs[i, 1]
		laplace = laplace_frame[x, y]
		inner_circle = draw.disk((x, y), inner_rad, shape=(1024, 1024))
		outer_circle = draw.disk((x, y), outer_rad, shape=(1024, 1024))
		inner_count, outer_count = len(inner_circle[0]), len(outer_circle[0])
		inner_gfp = np.sum(gfp_frame[inner_circle]) / inner_count
		outer_gfp = (np.sum(gfp_frame[outer_circle]) - np.sum(gfp_frame[inner_circle])) / (outer_count - inner_count)
		inner_rfp = np.sum(rfp_frame[inner_circle]) / inner_count
		outer_rfp = (np.sum(rfp_frame[outer_circle]) - np.sum(rfp_frame[inner_circle])) / (outer_count - inner_count)
		ratio = (inner_gfp - outer_gfp) / (inner_rfp - outer_rfp)
		out_arr[i] = [x, y, laplace, inner_gfp, outer_gfp, inner_rfp, outer_rfp, ratio]
	return out_arr

def plot_blobs(rfp_frame, blobs, ax):
	ax.imshow(rfp_frame)
	for i in blobs:
		ax.scatter(i[1], i[0])
	ax.axis('off')

def check_params(scratch_dir, mask, sigma, thresh, inner_rad, outer_rad):
	tif_path = os.path.join(scratch_dir, 'registered.tif')
	fig, axs = plt.subplots(4, 5, figsize=(15, 12))
	axs = axs.flatten()
	indexes = np.linspace(0, get_max_frame(tif_path) - 1, len(axs), dtype=int)
	for i in range(len(axs)):
		frame = get_frame(tif_path, indexes[i])
		blobs = get_blobs(frame, mask, sigma, thresh, inner_rad, outer_rad)
		axs[i].set_title('Frame %i' % indexes[i])
		plot_blobs(frame[0], blobs, ax=axs[i])
	plt.tight_layout()

def write_blobs(scratch_dir, mask, sigma, thresh, inner_rad, outer_rad, max_frame):
	tif_path = os.path.join(scratch_dir, 'registered.tif')
	if max_frame == None:
		max_frame = get_max_frame(tif_path)
	track_dir = os.path.join(scratch_dir, 'worms')
	os.makedirs(track_dir, exist_ok=True)
	
	def parallel(i):
		frame = get_frame(tif_path, i)
		blobs = get_blobs(frame, mask, sigma, thresh, inner_rad, outer_rad)
		df = pd.DataFrame()
		df['frame'] = [i] * len(blobs)
		df[['x', 'y', 'laplace', 'inner_gfp/px', 'outer_gfp/px', 'inner_rfp/px', 'outer_rfp/px', 'ratio']] = blobs
		df['id'] = range(len(df))
		df.to_csv(os.path.join(track_dir, '%05d.csv' % i), index=False)
	Parallel(n_jobs=-1)(delayed(parallel)(i) for i in range(1, max_frame))

def link_worms(scratch_dir, seperation):
	track_dir = os.path.join(scratch_dir, 'tracks')
	worm_dir = os.path.join(scratch_dir, 'worms')
	os.makedirs(track_dir, exist_ok=True)
	os.system('cd %s && tar -cf %s/worms.tar *.csv' % (worm_dir, scratch_dir))
	os.system('tar -xf %s/worms.tar -C %s' % (scratch_dir, track_dir))

	def link(index, track_list):
		#renumbers ids so that all ids are unique
		df1, df2 = pd.read_csv(track_list[index]), pd.read_csv(track_list[index+1])
		ids1, ids2 = sorted(set(df1['id'])), sorted(set(df2['id']))
		df1['id'] = df1['id'].replace(dict(zip(ids1, range(len(ids1)))))
		df2['id'] = df2['id'].replace(dict(zip(ids2, range(len(ids1), len(ids1) + len(ids2)))))

		#if one has no worms just ends
		if len(df1) == 0 or len(df2) == 0:
			df = pd.concat([df1, df2])
			df.to_csv(track_list[index], index=False)
			os.remove(track_list[index+1])
			return

		#get last frame of first df and first frame of second df
		sub_df1 = df1[df1['frame'] == np.max(df1['frame'])]
		arr1 = np.array(sub_df1[['x', 'y']])
		sub_df2 = df2[df2['frame'] == np.min(df2['frame'])]
		arr2 = np.array(sub_df2[['x', 'y']])

		#get a dict of ids to be replaced, every worm from the first df should only link to one worm of the second df
		diff = arr1[:, np.newaxis, :] - arr2[np.newaxis, :, :]
		D = np.sqrt(np.sum(diff ** 2, axis=2))
		replace_dict = {}
		while True:
			if np.min(D) > seperation:
				break
			index2 = np.argmin(np.min(D, axis=0))
			index1 = np.argmin(D[:, index2])
			replace_dict[sub_df2['id'].iloc[index2]] = sub_df1['id'].iloc[index1]
			D[index1], D[:, index2] = np.inf, np.inf
		
		#replaces ids and overwrites files
		df2['id'] = df2['id'].replace(replace_dict)
		df = pd.concat([df1, df2])
		df.to_csv(track_list[index], index=False)
		os.remove(track_list[index+1])

	while True:
		track_list = sorted(glob(os.path.join(track_dir, '*.csv')))
		if len(track_list) == 1:
			break
		Parallel(n_jobs=-1)(delayed(link)(i, track_list) for i in range(0, (len(track_list) // 2) * 2, 2))
	untrimmed_path = os.path.join(scratch_dir, 'untrimmed_tracks.csv')
	os.system('mv %s %s' % (track_list[0], untrimmed_path))
	os.rmdir(track_dir)

def plot_tracks(scratch_dir, mask, csv, attr='id'):
	#csv takes either the input 'untrimmed' or 'trimmed'
	csv_path = os.path.join(scratch_dir, csv + '_tracks.csv')
	df = pd.read_csv(csv_path, index_col=0)
	plt.figure(figsize=(5, 5))
	plt.imshow(mask[0], cmap='Greys_r')
	if attr == 'laplace':
		plt.scatter(df['y'], df['x'], c=df['laplace'], lw=0, s=1, vmax=np.percentile(df['laplace'], 90))
		plt.colorbar()
	else:
		plt.scatter(df['y'], df['x'], c=df['id'] % 20, lw=0, s=1, cmap='tab20')
	plt.axis('off'); plt.tight_layout()
	plt.savefig(os.path.join(scratch_dir, csv + '_tracks.png'), dpi=300)

def select_tracks(scratch_dir, mask):
	plot_tracks(scratch_dir, mask, 'untrimmed', 'laplace')
	pts = np.array(plt.ginput(-1, timeout=0))
	plt.close()
	return pts

def trim(scratch_dir, pts, save=True):
	untrimmed_path = os.path.join(scratch_dir, 'untrimmed_tracks.csv')
	untrimmed_df = pd.read_csv(untrimmed_path, index_col=0)
	good_ids = []
	for i in pts:
		id = untrimmed_df.iloc[np.argmin((untrimmed_df['x'] - i[1]) ** 2 + (untrimmed_df['y'] - i[0]) ** 2)]['id']
		good_ids.append(id)
	trimmed_df = pd.concat([untrimmed_df[untrimmed_df['id'] == i] for i in good_ids])
	trimmed_path = os.path.join(scratch_dir, 'trimmed_tracks.csv')
	trimmed_df.to_csv(trimmed_path)
	return trimmed_df

def double_check_ids(scratch_dir):
	trimmed_path = os.path.join(scratch_dir, 'trimmed_tracks.csv')
	trimmed_df = pd.read_csv(trimmed_path, index_col=0).sort_index()
	fig, axs = plt.subplots(8, 5, figsize=(15, 24))
	tif_path = os.path.join(scratch_dir, 'registered.tif')
	axs = axs.flatten()
	step = len(trimmed_df.index) // len(axs)
	for i in range(len(axs)):
		index = trimmed_df.index[i * step]
		frame = get_frame(tif_path, index)
		axs[i].set_title('Frame %i' % index)
		sub_df = trimmed_df.loc[index]
		axs[i].imshow(frame[0])
		axs[i].scatter(sub_df['y'], sub_df['x'], c=sub_df['id'] % 20, cmap='tab20')
		axs[i].axis('off')
	plt.tight_layout()

def select_patch(scratch_dir):
    patch = tifffile.imread(os.path.join(scratch_dir, 'patch.tif'))
    plt.imshow(patch)
    patch_pts = np.array(plt.ginput(-1, timeout=0))
    plt.close()
    return patch_pts

def add_patch_to_df(trimmed_df, scratch_dir, patch_pts, save=True):
    path = Path(patch_pts)
    trimmed_df['in_patch'] = [path.contains_point(pt) for pt in zip(trimmed_df['y'], trimmed_df['x'])]
    mask = tifffile.imread(os.path.join(scratch_dir, 'mask.tif'))
    fig, ax = plt.subplots()
    polygon = Polygon(patch_pts, closed=True, edgecolor='r', facecolor='none')
    ax.add_patch(polygon)
    plt.imshow(mask[0])
    plt.scatter(trimmed_df['y'], trimmed_df['x'], c=trimmed_df['in_patch'])
    if save:
        patch_pts_path = os.path.join(scratch_dir, 'patch_pts.csv')
        pd.DataFrame(patch_pts).to_csv(patch_pts_path, header=False, index=False)
        trimmed_path = os.path.join(scratch_dir, 'trimmed_tracks.csv')
        trimmed_df.to_csv(trimmed_path)

def copy_to_output(input_nd2, output_path, scratch_dir, move_tif=True):
	exp_name = name_from_path(input_nd2)
	input_dir = os.path.dirname(input_nd2)
	output_dir = os.path.join(output_path, exp_name)
	os.makedirs(output_dir, exist_ok=True)
	os.system('cp %s/mask.tif %s/%s_mask.tif' % (scratch_dir, output_dir, exp_name))
	os.system('cp %s/params.txt %s/%s_params.txt' % (scratch_dir, output_dir, exp_name))
	os.system('cp %s/patch.tif %s/%s_patch.tif' % (scratch_dir, output_dir, exp_name))
	os.system('cp %s/patch_pts.csv %s/%s_patch_pts.csv' % (scratch_dir, output_dir, exp_name))
	os.system('cp %s/trimmed_tracks.csv %s/%s_trimmed_tracks.csv' % (scratch_dir, output_dir, exp_name))
	os.system('cp %s/trimmed_tracks.png %s/%s_trimmed_tracks.png' % (scratch_dir, output_dir, exp_name))
	if move_tif:
		os.system('mv %s/registered.tif %s/%s_registered.tif' % (scratch_dir, input_dir, exp_name))
