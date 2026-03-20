import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray

import numpy as np

from perception_cylinder_pipeline.pc2_utils import build_pointcloud2_from_xyz_rgb, pointcloud2_to_xyz_arrays
from perception_cylinder_pipeline.pipeline import PipelineConfig, box_filter, voxel_downsample, estimate_normals, find_plane_ransac, remove_plane_inliers, euclidean_clustering, extract_clusters, fit_cylinders_in_clusters, semantic_label_from_rgb
from perception_cylinder_pipeline.visualization import CylinderMarkerPublisher


class CylinderProcessorNode(Node):
	def __init__(self) -> None:
		super().__init__('cylinder_processor_node')
		self._subscription = self.create_subscription(
			PointCloud2,
			'/oakd/points',
			self.pointcloud_callback,
			10,
		)
		self._valid_pub = self.create_publisher(PointCloud2, '/pipeline/stage_valid', 10)
		self._box_pub = self.create_publisher(PointCloud2, '/pipeline/stage_box', 10)
		self._downsample_pub = self.create_publisher(PointCloud2, '/pipeline/stage_downsample', 10)
		self._no_plane_pub = self.create_publisher(PointCloud2, '/pipeline/stage_no_plane', 10)
		self._clusters_pub = self.create_publisher(PointCloud2, '/pipeline/stage_clusters', 10)
		self._cylinders_pub = self.create_publisher(PointCloud2, '/pipeline/stage_cylinders', 10)
		self._detections_pub = self.create_publisher(MarkerArray, '/viz/detections', 10)
		self._config = PipelineConfig()
		self._marker_publisher = CylinderMarkerPublisher(self._detections_pub, namespace='detections')
		self._marker_publisher.set_logger(self.get_logger())
		self._log_every_n = 1
		self._msg_count = 0
		self.get_logger().info('Subscribed to /oakd/points (sensor_msgs/msg/PointCloud2).')
		self.get_logger().info('Publishing debug valid cloud to /pipeline/stage_valid.')
		self.get_logger().info('Publishing debug box-filtered cloud to /pipeline/stage_box.')
		self.get_logger().info('Publishing debug downsampled cloud to /pipeline/stage_downsample.')
		self.get_logger().info('Publishing debug plane-removed cloud to /pipeline/stage_no_plane.')
		self.get_logger().info('Publishing debug clustered cloud to /pipeline/stage_clusters.')
		self.get_logger().info('Publishing debug cylinder cloud to /pipeline/stage_cylinders.')
		self.get_logger().info('Publishing detection markers to /viz/detections.')

	def pointcloud_callback(self, msg: PointCloud2) -> None:
		parsed = pointcloud2_to_xyz_arrays(msg)
		valid_cloud_msg = build_pointcloud2_from_xyz_rgb(msg.header, parsed.valid_xyz, parsed.valid_rgb)
		self._valid_pub.publish(valid_cloud_msg)

		box_xyz, box_rgb = box_filter(
			parsed.valid_xyz,
			parsed.valid_rgb,
			self._config.box_min,
			self._config.box_max,
		)
		box_cloud_msg = build_pointcloud2_from_xyz_rgb(msg.header, box_xyz, box_rgb)
		self._box_pub.publish(box_cloud_msg)

		down_xyz, down_rgb = voxel_downsample(
			box_xyz,
			box_rgb,
			self._config.voxel_size,
		)
		downsample_cloud_msg = build_pointcloud2_from_xyz_rgb(msg.header, down_xyz, down_rgb)
		self._downsample_pub.publish(downsample_cloud_msg)

		normals = estimate_normals(down_xyz, self._config.normal_k)

		plane_normal, plane_inlier_mask = find_plane_ransac(
			down_xyz,
			self._config.target_normal,
			self._config.normal_thresh,
			self._config.floor_dist,
			self._config.plane_ransac_iters,
		)
		
		no_plane_xyz, no_plane_rgb = remove_plane_inliers(down_xyz, down_rgb, plane_inlier_mask)
		no_plane_cloud_msg = build_pointcloud2_from_xyz_rgb(msg.header, no_plane_xyz, no_plane_rgb)
		self._no_plane_pub.publish(no_plane_cloud_msg)

		cluster_indices_list = euclidean_clustering(
			no_plane_xyz,
			self._config.cluster_dist_thresh,
			self._config.min_cluster_size,
			self._config.max_cluster_size,
		)

		clustered_xyz, clustered_rgb, cluster_sizes = extract_clusters(no_plane_xyz, no_plane_rgb, cluster_indices_list)
		clusters_cloud_msg = build_pointcloud2_from_xyz_rgb(msg.header, clustered_xyz, clustered_rgb)
		self._clusters_pub.publish(clusters_cloud_msg)
		# Filter normals for no_plane points
		no_plane_normals = normals[~plane_inlier_mask]

		# Fit cylinders in each cluster
		(
			cylinder_xyz,
			cylinder_rgb,
			cylinder_inlier_counts,
			cylinder_axes,
			cylinder_normals,
			source_cluster_xyz,
			source_cluster_counts,
		) = fit_cylinders_in_clusters(
			no_plane_xyz,
			no_plane_normals,
			no_plane_rgb,
			cluster_indices_list,
			self._config,
		)
		cylinders_cloud_msg = build_pointcloud2_from_xyz_rgb(msg.header, cylinder_xyz, cylinder_rgb)
		self._cylinders_pub.publish(cylinders_cloud_msg)

		# Compute semantic labels for each detected cylinder
		cylinder_count = len(cylinder_inlier_counts)
		cylinder_labels = []
		cylinder_avg_rgb_list = []
		cylinder_avg_hsv_list = []
		cylinder_inlier_xyz_list = []
		cylinder_centers = []
		expected_axis_prior = np.asarray(self._config.expected_cylinder_axis, dtype=np.float32)
		expected_axis_prior_norm = float(np.linalg.norm(expected_axis_prior))
		if expected_axis_prior_norm > 1e-10:
			expected_axis_prior = expected_axis_prior / expected_axis_prior_norm
		else:
			expected_axis_prior = np.array([0.0, 1.0, 0.0], dtype=np.float32)
		global_y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
		global_z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
		
		cylinder_normals_list = []
		source_cluster_xyz_list = []
		if cylinder_count > 0:
			# Reconstruct individual cylinder xyz/rgb from combined result
			inlier_point_idx = 0
			source_point_idx = 0
			for cyl_idx, inlier_count in enumerate(cylinder_inlier_counts):
				inlier_end_idx = inlier_point_idx + int(inlier_count)
				cyl_xyz = cylinder_xyz[inlier_point_idx:inlier_end_idx]
				cyl_rgb = cylinder_rgb[inlier_point_idx:inlier_end_idx]
				cyl_normals = cylinder_normals[inlier_point_idx:inlier_end_idx]
				cylinder_inlier_xyz_list.append(cyl_xyz)
				cylinder_normals_list.append(cyl_normals)

				source_count = int(source_cluster_counts[cyl_idx]) if cyl_idx < len(source_cluster_counts) else 0
				source_end_idx = source_point_idx + source_count
				source_xyz = source_cluster_xyz[source_point_idx:source_end_idx]
				source_cluster_xyz_list.append(source_xyz)
				if cyl_xyz.shape[0] > 0:
					cylinder_centers.append(np.mean(cyl_xyz, axis=0))
				else:
					cylinder_centers.append(np.zeros(3, dtype=np.float32))
				label, avg_rgb, hsv = semantic_label_from_rgb(cyl_rgb)
				cylinder_labels.append(label)
				cylinder_avg_rgb_list.append(avg_rgb)
				cylinder_avg_hsv_list.append(hsv)
				inlier_point_idx = inlier_end_idx
				source_point_idx = source_end_idx

		self._marker_publisher.publish(
			frame_id=msg.header.frame_id,
			stamp=msg.header.stamp,
			source_cluster_xyz_list=source_cluster_xyz_list,
			cylinder_inlier_xyz_list=cylinder_inlier_xyz_list,
			cylinder_normals_list=cylinder_normals_list,
			cylinder_inlier_counts=cylinder_inlier_counts,
			cylinder_labels=cylinder_labels,
			cylinder_axes=cylinder_axes,
			marker_axis_prior=self._config.expected_cylinder_axis,
			cyl_radius=self._config.cyl_radius,
		)
		
		raw_count = parsed.raw_xyz.shape[0]
		valid_count = parsed.valid_xyz.shape[0]
		rgb_count = parsed.valid_rgb.shape[0]
		box_count = box_xyz.shape[0]
		downsample_count = down_xyz.shape[0]
		normals_count = normals.shape[0]
		normals_finite_count = np.isfinite(normals).all(axis=1).sum()
		plane_inlier_count = plane_inlier_mask.sum()
		no_plane_count = no_plane_xyz.shape[0]
		cluster_count = len(cluster_indices_list)
		clustered_count = clustered_xyz.shape[0]
		cylinder_points = cylinder_xyz.shape[0]
		self._msg_count += 1
		
		# Build log strings for RGB and HSV
		if cylinder_count > 0:
			cylinder_rgb_str = ', '.join(
				f'r{int(rgb[0])}g{int(rgb[1])}b{int(rgb[2])}'
				for rgb in cylinder_avg_rgb_list
			)
			cylinder_hsv_str = ', '.join(
				f'h{hsv[0]:.1f}s{hsv[1]:.2f}v{hsv[2]:.2f}'
				for hsv in cylinder_avg_hsv_list
			)
		else:
			cylinder_rgb_str = 'none'
			cylinder_hsv_str = 'none'

		if self._msg_count % self._log_every_n == 0:
			if rgb_count > 0:
				rgb_min = parsed.valid_rgb.min(axis=0)
				rgb_max = parsed.valid_rgb.max(axis=0)
				rgb_stats = (
					f'r[{int(rgb_min[0])},{int(rgb_max[0])}], '
					f'g[{int(rgb_min[1])},{int(rgb_max[1])}], '
					f'b[{int(rgb_min[2])},{int(rgb_max[2])}]'
				)
			else:
				rgb_stats = 'r[n/a,n/a], g[n/a,n/a], b[n/a,n/a]'

			self.get_logger().info(
				'Received PointCloud2: '
				f'width={msg.width}, '
				f'height={msg.height}, '
				f'frame_id={parsed.frame_id}, '
				f'raw_points={raw_count}, '
				f'valid_points={valid_count}, '
				f'box_points={box_count}, '
				f'downsample_points={downsample_count}, '
				f'normals_count={normals_count}, '
				f'normals_finite={normals_finite_count}, '
				f'plane_inliers={plane_inlier_count}, '
				f'no_plane_points={no_plane_count}, '
				f'cluster_count={cluster_count}, '
				f'cluster_sizes={list(cluster_sizes)}, '
				f'clustered_points={clustered_count}, '
				f'cylinder_count={cylinder_count}, '
				f'cylinder_inlier_counts={list(cylinder_inlier_counts)}, '
				f'cylinder_points={cylinder_points}, '
				f'cylinder_labels={cylinder_labels}, '
				f'cylinder_avg_rgb=[{cylinder_rgb_str}], '
				f'cylinder_avg_hsv=[{cylinder_hsv_str}], '
				f'cylinder_axis_prior=[{expected_axis_prior[0]:+.2f},{expected_axis_prior[1]:+.2f},{expected_axis_prior[2]:+.2f}], '
				f'cylinder_axis_alignment_thresh={self._config.axis_alignment_thresh:.2f}, '
				f'rgb_count={rgb_count}, '
				f'{rgb_stats}'
			)

			if cylinder_count > 0:
				cylinder_debug_lines = []
				for i in range(cylinder_count):
					axis = cylinder_axes[i] if i < cylinder_axes.shape[0] else np.zeros(3, dtype=np.float32)
					axis_norm = float(np.linalg.norm(axis))
					if axis_norm > 1e-10:
						axis_unit = axis / axis_norm
						abs_dot_prior = float(np.abs(np.dot(axis_unit, expected_axis_prior)))
						abs_dot_y = float(np.abs(np.dot(axis_unit, global_y_axis)))
						abs_dot_z = float(np.abs(np.dot(axis_unit, global_z_axis)))
					else:
						abs_dot_prior = 0.0
						abs_dot_y = 0.0
						abs_dot_z = 0.0
					center = cylinder_centers[i] if i < len(cylinder_centers) else np.zeros(3, dtype=np.float32)
					inlier_count = int(cylinder_inlier_counts[i]) if i < len(cylinder_inlier_counts) else 0
					cylinder_debug_lines.append(
						f'  cyl[{i}] axis=[{axis[0]:+.4f}, {axis[1]:+.4f}, {axis[2]:+.4f}], '
						f'axis_norm={axis_norm:.6f}, '
						f'abs_dot_prior={abs_dot_prior:.6f}, '
						f'abs_dot_y={abs_dot_y:.6f}, '
						f'abs_dot_z={abs_dot_z:.6f}, '
						f'center=[{center[0]:+.3f}, {center[1]:+.3f}, {center[2]:+.3f}], '
						f'inliers={inlier_count}'
					)

				self.get_logger().info(
					'Cylinder fit debug (per detected cylinder): using cylinder alignment prior '
					f'[{expected_axis_prior[0]:+.2f}, {expected_axis_prior[1]:+.2f}, {expected_axis_prior[2]:+.2f}]\n'
					+ '\n'.join(cylinder_debug_lines)
				)
			else:
				self.get_logger().info('Cylinder fit debug: no cylinders detected in this cloud.')


def main(args=None) -> None:
	rclpy.init(args=args)
	node = CylinderProcessorNode()
	try:
		rclpy.spin(node)
	finally:
		node.destroy_node()
		rclpy.shutdown()
