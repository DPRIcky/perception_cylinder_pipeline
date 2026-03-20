from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class PipelineConfig:
	box_min: np.ndarray = field(default_factory=lambda: np.array([-1.0, -0.6, 0.2], dtype=np.float32))
	box_max: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.6, 2.0], dtype=np.float32))
	voxel_size: float = 0.02
	normal_k: int = 20
	floor_dist: float = 0.02
	target_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))
	normal_thresh: float = 0.85
	plane_ransac_iters: int = 100
	cluster_dist_thresh: float = 0.10
	min_cluster_size: int = 20
	max_cluster_size: int = 2000
	cyl_radius: float = 0.055
	cyl_radius_thresh: float = 0.02
	cylinder_ransac_iters: int = 100
	expected_cylinder_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))
	axis_alignment_thresh: float = 0.85
	min_cylinder_inliers: int = 30
	max_cylinders: int = 3


def box_filter(
	xyz: np.ndarray,
	rgb: np.ndarray,
	box_min: np.ndarray,
	box_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	xyz_array = np.asarray(xyz, dtype=np.float32)
	rgb_array = np.asarray(rgb)
	box_min_array = np.asarray(box_min, dtype=np.float32)
	box_max_array = np.asarray(box_max, dtype=np.float32)

	if xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
		raise ValueError('xyz must have shape (N, 3)')
	if rgb_array.ndim != 2 or rgb_array.shape[1] != 3:
		raise ValueError('rgb must have shape (N, 3)')
	if rgb_array.shape[0] != xyz_array.shape[0]:
		raise ValueError('xyz and rgb must have the same number of rows')
	if box_min_array.shape != (3,) or box_max_array.shape != (3,):
		raise ValueError('box_min and box_max must each have shape (3,)')

	in_box_mask = np.logical_and(xyz_array >= box_min_array, xyz_array <= box_max_array).all(axis=1)
	filtered_xyz = xyz_array[in_box_mask]
	filtered_rgb = rgb_array[in_box_mask]
	return filtered_xyz, filtered_rgb


def voxel_downsample(
	xyz: np.ndarray,
	rgb: np.ndarray,
	voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
	xyz_array = np.asarray(xyz, dtype=np.float32)
	rgb_array = np.asarray(rgb)

	if xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
		raise ValueError('xyz must have shape (N, 3)')
	if rgb_array.ndim != 2 or rgb_array.shape[1] != 3:
		raise ValueError('rgb must have shape (N, 3)')
	if rgb_array.shape[0] != xyz_array.shape[0]:
		raise ValueError('xyz and rgb must have the same number of rows')
	if voxel_size <= 0.0:
		raise ValueError('voxel_size must be > 0')
	if xyz_array.shape[0] == 0:
		return xyz_array, rgb_array

	voxel_indices = np.floor(xyz_array / float(voxel_size)).astype(np.int32)
	_, first_indices = np.unique(voxel_indices, axis=0, return_index=True)
	first_indices = np.sort(first_indices)

	down_xyz = xyz_array[first_indices]
	down_rgb = rgb_array[first_indices]
	return down_xyz, down_rgb


def estimate_normals(
	xyz: np.ndarray,
	k: int,
) -> np.ndarray:
	xyz_array = np.asarray(xyz, dtype=np.float32)

	if xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
		raise ValueError('xyz must have shape (N, 3)')
	if k <= 0:
		raise ValueError('k must be > 0')
	if xyz_array.shape[0] == 0:
		return np.empty((0, 3), dtype=np.float32)

	if xyz_array.shape[0] < k:
		k_eff = xyz_array.shape[0]
	else:
		k_eff = k

	tree = cKDTree(xyz_array)
	_, indices = tree.query(xyz_array, k=k_eff + 1)

	normals_list = []
	for i in range(xyz_array.shape[0]):
		if indices.ndim == 1:
			neighbor_indices = indices[1:]
		else:
			neighbor_indices = indices[i, 1:]

		neighbors = xyz_array[neighbor_indices]
		center = xyz_array[i:i+1]
		centered = neighbors - center

		if centered.shape[0] >= 3:
			U, S, Vt = np.linalg.svd(centered, full_matrices=True)
			normal = Vt[-1, :].astype(np.float32)
		else:
			normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

		normals_list.append(normal)

	normals = np.array(normals_list, dtype=np.float32)
	return normals


def find_plane_ransac(
	xyz: np.ndarray,
	target_normal: np.ndarray,
	normal_thresh: float,
	floor_dist: float,
	iters: int,
) -> tuple[np.ndarray, np.ndarray]:
	xyz_array = np.asarray(xyz, dtype=np.float32)
	target_normal_array = np.asarray(target_normal, dtype=np.float32)

	if xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
		raise ValueError('xyz must have shape (N, 3)')
	if xyz_array.shape[0] < 3:
		return np.array([0, 1, 0], dtype=np.float32), np.zeros(xyz_array.shape[0], dtype=bool)

	target_normal_array = target_normal_array / (np.linalg.norm(target_normal_array) + 1e-10)

	best_inlier_mask = np.zeros(xyz_array.shape[0], dtype=bool)
	best_inlier_count = 0
	best_plane_normal = np.array([0, 1, 0], dtype=np.float32)

	for _ in range(iters):
		idx = np.random.choice(xyz_array.shape[0], 3, replace=False)
		p1, p2, p3 = xyz_array[idx]

		v1 = p2 - p1
		v2 = p3 - p1
		normal = np.cross(v1, v2)
		norm = np.linalg.norm(normal)

		if norm < 1e-6:
			continue

		normal = normal / norm

		if np.dot(normal, target_normal_array) < normal_thresh:
			normal = -normal
			if np.dot(normal, target_normal_array) < normal_thresh:
				continue

		d = -np.dot(normal, p1)
		distances = np.abs(np.dot(xyz_array, normal) + d)
		inlier_mask = distances < floor_dist
		inlier_count = inlier_mask.sum()

		if inlier_count > best_inlier_count:
			best_inlier_count = inlier_count
			best_inlier_mask = inlier_mask
			best_plane_normal = normal

	return best_plane_normal, best_inlier_mask


def remove_plane_inliers(
	xyz: np.ndarray,
	rgb: np.ndarray,
	inlier_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	xyz_array = np.asarray(xyz, dtype=np.float32)
	rgb_array = np.asarray(rgb)
	inlier_mask_array = np.asarray(inlier_mask, dtype=bool)

	if xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
		raise ValueError('xyz must have shape (N, 3)')
	if rgb_array.ndim != 2 or rgb_array.shape[1] != 3:
		raise ValueError('rgb must have shape (N, 3)')
	if xyz_array.shape[0] != inlier_mask_array.shape[0]:
		raise ValueError('xyz and inlier_mask must have the same number of rows')

	outlier_mask = ~inlier_mask_array
	filtered_xyz = xyz_array[outlier_mask]
	filtered_rgb = rgb_array[outlier_mask]
	return filtered_xyz, filtered_rgb


def euclidean_clustering(
	xyz: np.ndarray,
	dist_thresh: float,
	min_cluster_size: int,
	max_cluster_size: int,
) -> list[np.ndarray]:
	xyz_array = np.asarray(xyz, dtype=np.float32)

	if xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
		raise ValueError('xyz must have shape (N, 3)')
	if xyz_array.shape[0] == 0:
		return []

	tree = cKDTree(xyz_array)
	visited = np.zeros(xyz_array.shape[0], dtype=bool)
	clusters = []

	for i in range(xyz_array.shape[0]):
		if visited[i]:
			continue

		neighbors_in_radius = tree.query_ball_point(xyz_array[i], dist_thresh)
		stack = [j for j in neighbors_in_radius if not visited[j]]
		cluster = []

		while stack:
			j = stack.pop()
			if visited[j]:
				continue

			visited[j] = True
			cluster.append(j)

			current_neighbors = tree.query_ball_point(xyz_array[j], dist_thresh)
			for k in current_neighbors:
				if not visited[k]:
					stack.append(k)

		if min_cluster_size <= len(cluster) <= max_cluster_size:
			clusters.append(np.array(cluster, dtype=np.int32))

	return clusters


def extract_clusters(
	xyz: np.ndarray,
	rgb: np.ndarray,
	cluster_indices_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	xyz_array = np.asarray(xyz, dtype=np.float32)
	rgb_array = np.asarray(rgb)

	if xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
		raise ValueError('xyz must have shape (N, 3)')
	if rgb_array.ndim != 2 or rgb_array.shape[1] != 3:
		raise ValueError('rgb must have shape (N, 3)')
	if xyz_array.shape[0] != rgb_array.shape[0]:
		raise ValueError('xyz and rgb must have the same number of rows')

	if not cluster_indices_list:
		return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), np.array([], dtype=np.int32)

	clustered_xyz_list = []
	clustered_rgb_list = []
	cluster_sizes = []

	for cluster_idx in cluster_indices_list:
		clustered_xyz_list.append(xyz_array[cluster_idx])
		clustered_rgb_list.append(rgb_array[cluster_idx])
		cluster_sizes.append(len(cluster_idx))

	clustered_xyz = np.vstack(clustered_xyz_list) if clustered_xyz_list else np.empty((0, 3), dtype=np.float32)
	clustered_rgb = np.vstack(clustered_rgb_list) if clustered_rgb_list else np.empty((0, 3), dtype=np.uint8)
	cluster_sizes_array = np.array(cluster_sizes, dtype=np.int32)

	return clustered_xyz, clustered_rgb, cluster_sizes_array

def find_single_cylinder(
	xyz: np.ndarray,
	normals: np.ndarray,
	config: PipelineConfig,
) -> tuple[bool, np.ndarray, np.ndarray]:
	"""
	Fit a cylinder to a single cluster using RANSAC.
	
	Parameters:
	  xyz: cluster points, shape (N, 3), dtype float32
	  normals: surface normals, shape (N, 3), dtype float32
	  config: PipelineConfig with cylinder parameters
	
	Returns:
	  (found: bool, inlier_mask: np.ndarray(dtype=bool, shape N), axis: np.ndarray(shape (3,)))
	  inlier_mask and axis are valid only if found=True
	"""
	xyz_array = np.asarray(xyz, dtype=np.float32)
	normals_array = np.asarray(normals, dtype=np.float32)
	
	if xyz_array.shape[0] < config.min_cylinder_inliers:
		return False, np.array([], dtype=bool), np.zeros(3, dtype=np.float32)
	
	if xyz_array.shape[0] != normals_array.shape[0]:
		raise ValueError('xyz and normals must have same number of points')
	
	best_inlier_mask = np.zeros(xyz_array.shape[0], dtype=bool)
	best_inlier_count = 0
	best_axis = np.zeros(3, dtype=np.float32)
	expected_axis = np.asarray(config.expected_cylinder_axis, dtype=np.float32)
	expected_axis_norm = float(np.linalg.norm(expected_axis))
	if expected_axis_norm < 1e-10:
		expected_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
	else:
		expected_axis = expected_axis / expected_axis_norm
	
	num_iters = min(config.cylinder_ransac_iters, max(1, xyz_array.shape[0] // 2))
	
	for _ in range(num_iters):
		# Sample two point-normal pairs
		idx = np.random.choice(xyz_array.shape[0], 2, replace=False)
		p1, p2 = xyz_array[idx]
		n1, n2 = normals_array[idx]
		
		# Normalize normals
		n1_norm = np.linalg.norm(n1)
		n2_norm = np.linalg.norm(n2)
		
		if n1_norm < 1e-6 or n2_norm < 1e-6:
			continue
		
		n1 = n1 / n1_norm
		n2 = n2 / n2_norm
		
		# Candidate axis from cross product of normals
		axis_candidate = np.cross(n1, n2)
		axis_norm = np.linalg.norm(axis_candidate)
		
		if axis_norm < 1e-6:  # normals too parallel
			continue
		
		axis_candidate = axis_candidate / axis_norm
		
		# Prefer axes aligned with the configured expected axis (sign-insensitive)
		axis_alignment = float(np.abs(np.dot(axis_candidate, expected_axis)))
		if axis_alignment < config.axis_alignment_thresh:
			continue
		
		# Compute perpendicular distances from all points to axis passing through p1
		v = xyz_array - p1
		proj_lengths = np.dot(v, axis_candidate)  # shape (N,)
		projections = proj_lengths[:, np.newaxis] * axis_candidate  # shape (N, 3)
		perpendiculars = v - projections  # shape (N, 3)
		distances = np.linalg.norm(perpendiculars, axis=1)  # shape (N,)
		
		# Count inliers: radius within tolerance
		radius_min = config.cyl_radius - config.cyl_radius_thresh
		radius_max = config.cyl_radius + config.cyl_radius_thresh
		inlier_mask = (distances >= radius_min) & (distances <= radius_max)
		inlier_count = inlier_mask.sum()
		
		if inlier_count > best_inlier_count:
			best_inlier_count = inlier_count
			best_inlier_mask = inlier_mask
			best_axis = axis_candidate.astype(np.float32)
	
	if best_inlier_count >= config.min_cylinder_inliers:
		return True, best_inlier_mask, best_axis
	else:
		return False, np.array([], dtype=bool), np.zeros(3, dtype=np.float32)


def fit_cylinders_in_clusters(
	xyz_points: np.ndarray,
	normals_points: np.ndarray,
	rgb_points: np.ndarray,
	cluster_indices_list: list[np.ndarray],
	config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""
	Fit cylinders in each cluster and gather inlier points.
	
	Parameters:
	  xyz_points: all no_plane points, shape (M, 3)
	  normals_points: all normals for those points, shape (M, 3)
	  rgb_points: all colors, shape (M, 3)
	  cluster_indices_list: list of index arrays from euclidean_clustering
	  config: PipelineConfig
	
	Returns:
	  (
	    cylinder_xyz,
	    cylinder_rgb,
	    cylinder_inlier_counts,
	    cylinder_axes,
	    cylinder_normals,
	    source_cluster_xyz,
	    source_cluster_counts,
	  )
	  cylinder_inlier_counts is an array of inlier counts per accepted cylinder
	  cylinder_axes is an array of fitted axis vectors with shape (num_cylinders, 3)
	  cylinder_normals is an array of inlier normals, stacked like cylinder_xyz
	  source_cluster_xyz is stacked full source clusters for accepted cylinders
	  source_cluster_counts stores per-accepted-cylinder full cluster sizes
	"""
	xyz_array = np.asarray(xyz_points, dtype=np.float32)
	normals_array = np.asarray(normals_points, dtype=np.float32)
	rgb_array = np.asarray(rgb_points)
	
	cylinder_xyz_list = []
	cylinder_rgb_list = []
	cylinder_inlier_counts = []
	cylinder_axes = []
	cylinder_normals_list = []
	source_cluster_xyz_list = []
	source_cluster_counts = []
	
	num_accepted = 0
	
	for cluster_idx_array in cluster_indices_list:
		if num_accepted >= config.max_cylinders:
			break
		
		cluster_xyz = xyz_array[cluster_idx_array]
		cluster_normals = normals_array[cluster_idx_array]
		cluster_rgb = rgb_array[cluster_idx_array]
		
		found, inlier_mask, axis = find_single_cylinder(cluster_xyz, cluster_normals, config)
		
		if found:
			cyl_xyz = cluster_xyz[inlier_mask]
			cyl_rgb = cluster_rgb[inlier_mask]
			cyl_normals = cluster_normals[inlier_mask]
			cylinder_xyz_list.append(cyl_xyz)
			cylinder_rgb_list.append(cyl_rgb)
			cylinder_normals_list.append(cyl_normals)
			source_cluster_xyz_list.append(cluster_xyz)
			source_cluster_counts.append(cluster_xyz.shape[0])
			cylinder_inlier_counts.append(len(cyl_xyz))
			cylinder_axes.append(axis)
			num_accepted += 1
	
	if cylinder_xyz_list:
		combined_xyz = np.vstack(cylinder_xyz_list)
		combined_rgb = np.vstack(cylinder_rgb_list)
		combined_normals = np.vstack(cylinder_normals_list)
		combined_source_cluster_xyz = np.vstack(source_cluster_xyz_list)
	else:
		combined_xyz = np.empty((0, 3), dtype=np.float32)
		combined_rgb = np.empty((0, 3), dtype=np.uint8)
		combined_normals = np.empty((0, 3), dtype=np.float32)
		combined_source_cluster_xyz = np.empty((0, 3), dtype=np.float32)
	
	cylinder_inlier_counts_array = np.array(cylinder_inlier_counts, dtype=np.int32)
	cylinder_axes_array = np.array(cylinder_axes, dtype=np.float32) if cylinder_axes else np.empty((0, 3), dtype=np.float32)
	source_cluster_counts_array = np.array(source_cluster_counts, dtype=np.int32)
	
	return (
		combined_xyz,
		combined_rgb,
		cylinder_inlier_counts_array,
		cylinder_axes_array,
		combined_normals,
		combined_source_cluster_xyz,
		source_cluster_counts_array,
	)


def rgb_to_hsv(
	rgb: np.ndarray,
) -> tuple[float, float, float]:
	"""
	Convert RGB (0-255) to HSV.
	
	Parameters:
	  rgb: shape (3,) with values in [0, 255]
	
	Returns:
	  (h, s, v) where:
	    h in [0, 360) degrees
	    s in [0, 1]
	    v in [0, 1]
	"""
	rgb_array = np.asarray(rgb, dtype=np.float32)
	if rgb_array.shape != (3,):
		raise ValueError('rgb must have shape (3,)')
	
	rgb_norm = rgb_array / 255.0
	r, g, b = rgb_norm[0], rgb_norm[1], rgb_norm[2]
	
	max_val = max(r, g, b)
	min_val = min(r, g, b)
	delta = max_val - min_val
	
	# Hue
	if delta == 0:
		h = 0.0
	elif max_val == r:
		h = (60.0 * ((g - b) / delta % 6.0))
	elif max_val == g:
		h = (60.0 * ((b - r) / delta + 2.0))
	else:
		h = (60.0 * ((r - g) / delta + 4.0))
	
	if h < 0:
		h += 360.0
	
	# Saturation
	if max_val == 0:
		s = 0.0
	else:
		s = delta / max_val
	
	# Value
	v = max_val
	
	return h, s, v


def semantic_label_from_rgb(
	rgb_points: np.ndarray,
) -> tuple[str, np.ndarray, tuple[float, float, float]]:
	"""
	Compute semantic label for a set of cylinder inlier RGB points.
	
	Parameters:
	  rgb_points: shape (N, 3) with values in [0, 255], dtype uint8 or float
	
	Returns:
	  (label, avg_rgb, hsv) where:
	    label: 'red', 'green', 'blue', or 'unknown'
	    avg_rgb: array of shape (3,) with average RGB
	    hsv: tuple (h, s, v) of average color
	"""
	rgb_array = np.asarray(rgb_points, dtype=np.float32)
	
	if rgb_array.ndim != 2 or rgb_array.shape[1] != 3:
		raise ValueError('rgb_points must have shape (N, 3)')
	
	if rgb_array.shape[0] == 0:
		return 'unknown', np.array([0, 0, 0], dtype=np.uint8), (0.0, 0.0, 0.0)
	
	# Compute average RGB
	avg_rgb = np.mean(rgb_array, axis=0)
	avg_rgb_clamped = np.clip(avg_rgb, 0, 255)
	
	# Convert to HSV
	h, s, v = rgb_to_hsv(avg_rgb_clamped)
	
	# Classify based on hue (require some saturation to distinguish colors)
	if s < 0.15:
		# Too desaturated or too dark, consider it neutral
		label = 'unknown'
	elif s < 0.2 and v < 0.2:
		# Very dark
		label = 'unknown'
	elif h < 30 or h >= 330:
		label = 'red'
	elif h >= 60 and h < 180:
		label = 'green'
	elif h >= 180 and h < 270:
		label = 'blue'
	else:
		label = 'unknown'
	
	return label, avg_rgb_clamped.astype(np.uint8), (h, s, v)