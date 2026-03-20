from __future__ import annotations

import numpy as np
from rclpy.publisher import Publisher
from visualization_msgs.msg import Marker, MarkerArray


def _label_to_color_rgba(label: str) -> tuple[float, float, float, float]:
	if label == 'green':
		return 0.0, 1.0, 0.0, 0.85
	if label == 'red':
		return 1.0, 0.0, 0.0, 0.85
	if label == 'blue':
		return 0.0, 0.0, 1.0, 0.85
	return 0.85, 0.85, 0.85, 0.85


def _safe_normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray | None:
	vec_array = np.asarray(vec, dtype=np.float32)
	vec_norm = float(np.linalg.norm(vec_array))
	if vec_norm < eps:
		return None
	return vec_array / vec_norm


def _orthogonal_unit_vector(vec: np.ndarray) -> np.ndarray:
	"""Return a unit vector orthogonal to vec."""
	v = np.asarray(vec, dtype=np.float32)
	if abs(float(v[0])) <= abs(float(v[1])) and abs(float(v[0])) <= abs(float(v[2])):
		basis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
	elif abs(float(v[1])) <= abs(float(v[0])) and abs(float(v[1])) <= abs(float(v[2])):
		basis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
	else:
		basis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
	ortho = np.cross(v, basis)
	ortho_unit = _safe_normalize(ortho)
	if ortho_unit is None:
		return np.array([1.0, 0.0, 0.0], dtype=np.float32)
	return ortho_unit


def _quaternion_from_two_vectors(source_vec: np.ndarray, target_vec: np.ndarray) -> np.ndarray:
	"""Return quaternion [x, y, z, w] rotating source_vec onto target_vec."""
	a_unit = _safe_normalize(source_vec)
	b_unit = _safe_normalize(target_vec)
	if a_unit is None or b_unit is None:
		return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

	dot_val = float(np.dot(a_unit, b_unit))
	dot_val = max(-1.0, min(1.0, dot_val))

	if dot_val > 1.0 - 1e-8:
		return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

	if dot_val < -1.0 + 1e-8:
		rot_axis = _orthogonal_unit_vector(a_unit)
		quat = np.array([rot_axis[0], rot_axis[1], rot_axis[2], 0.0], dtype=np.float32)
		quat_norm = float(np.linalg.norm(quat))
		if quat_norm > 1e-10:
			quat /= quat_norm
		return quat

	cross_vec = np.cross(a_unit, b_unit)
	quat = np.array([cross_vec[0], cross_vec[1], cross_vec[2], 1.0 + dot_val], dtype=np.float32)
	quat_norm = float(np.linalg.norm(quat))
	if quat_norm < 1e-10:
		return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
	return quat / quat_norm


class CylinderMarkerPublisher:
	def __init__(self, publisher: Publisher, namespace: str = 'detections') -> None:
		self._publisher = publisher
		self._namespace = namespace
		self._logger = None

	def set_logger(self, logger) -> None:
		self._logger = logger

	def publish(
		self,
		frame_id: str,
		stamp,
		source_cluster_xyz_list: list[np.ndarray],
		cylinder_inlier_xyz_list: list[np.ndarray],
		cylinder_normals_list: list[np.ndarray],
		cylinder_inlier_counts: np.ndarray,
		cylinder_labels: list[str],
		cylinder_axes: np.ndarray,
		marker_axis_prior: np.ndarray,
		cyl_radius: float,
	) -> None:
		clear_array = MarkerArray()
		delete_all = Marker()
		delete_all.header.frame_id = frame_id
		delete_all.header.stamp = stamp
		delete_all.ns = self._namespace
		delete_all.id = 0
		delete_all.action = Marker.DELETEALL
		clear_array.markers.append(delete_all)
		self._publisher.publish(clear_array)

		marker_array = MarkerArray()

		diameter = max(2.0 * float(cyl_radius), 0.02)

		for i, source_points in enumerate(source_cluster_xyz_list):
			if source_points.size == 0:
				continue

			source_xyz_array = np.asarray(source_points, dtype=np.float32)
			inlier_xyz_array = (
				np.asarray(cylinder_inlier_xyz_list[i], dtype=np.float32)
				if i < len(cylinder_inlier_xyz_list)
				else np.empty((0, 3), dtype=np.float32)
			)
			normals_array = (
				np.asarray(cylinder_normals_list[i], dtype=np.float32)
				if i < len(cylinder_normals_list)
				else np.empty((0, 3), dtype=np.float32)
			)

			fitted_axis = cylinder_axes[i] if i < cylinder_axes.shape[0] else np.array([0.0, 1.0, 0.0], dtype=np.float32)
			fitted_axis_norm = float(np.linalg.norm(fitted_axis))
			if fitted_axis_norm > 1e-10:
				fitted_axis = fitted_axis / fitted_axis_norm
			else:
				fitted_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)

			marker_axis = _safe_normalize(marker_axis_prior)
			if marker_axis is None:
				marker_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)

			axis_scalars = np.dot(source_xyz_array, marker_axis)
			orth_components = source_xyz_array - axis_scalars[:, np.newaxis] * marker_axis
			orth_center = np.mean(orth_components, axis=0)

			projections = axis_scalars
			if projections.size > 0:
				t_low = float(np.percentile(projections, 5.0))
				t_high = float(np.percentile(projections, 95.0))
				height = max(t_high - t_low, 0.05)
				center = orth_center + 0.5 * (t_low + t_high) * marker_axis
			else:
				height = 0.05
				center = np.mean(source_xyz_array, axis=0)

			label = cylinder_labels[i] if i < len(cylinder_labels) else 'unknown'
			r, g, b, a = _label_to_color_rgba(label)

			marker = Marker()
			marker.header.frame_id = frame_id
			marker.header.stamp = stamp
			marker.ns = self._namespace
			marker.id = i
			marker.type = Marker.CYLINDER
			marker.action = Marker.ADD

			marker.pose.position.x = float(center[0])
			marker.pose.position.y = float(center[1])
			marker.pose.position.z = float(center[2])

			source_axis_a = np.array([0.0, 0.0, 1.0], dtype=np.float32)
			target_axis_b = marker_axis
			quat = _quaternion_from_two_vectors(source_axis_a, target_axis_b)
			dot_a_b = float(np.dot(source_axis_a, target_axis_b))
			marker.pose.orientation.x = float(quat[0])
			marker.pose.orientation.y = float(quat[1])
			marker.pose.orientation.z = float(quat[2])
			marker.pose.orientation.w = float(quat[3])

			marker.scale.x = diameter
			marker.scale.y = diameter
			marker.scale.z = height

			marker.color.r = r
			marker.color.g = g
			marker.color.b = b
			marker.color.a = a

			if self._logger is not None:
				inlier_count = int(cylinder_inlier_counts[i]) if i < len(cylinder_inlier_counts) else int(inlier_xyz_array.shape[0])
				self._logger.info(
					'Published cylinder marker: '
					f'id={i}, '
					f'source_cluster_size={int(source_xyz_array.shape[0])}, '
					f'cylinder_inlier_count={inlier_count}, '
					f'fitted_axis=[{float(fitted_axis[0]):+.4f}, {float(fitted_axis[1]):+.4f}, {float(fitted_axis[2]):+.4f}], '
					f'marker_axis_used=[{float(marker_axis[0]):+.4f}, {float(marker_axis[1]):+.4f}, {float(marker_axis[2]):+.4f}], '
					f'source_axis_A=[{float(source_axis_a[0]):+.1f}, {float(source_axis_a[1]):+.1f}, {float(source_axis_a[2]):+.1f}], '
					f'target_axis_B=[{float(target_axis_b[0]):+.4f}, {float(target_axis_b[1]):+.4f}, {float(target_axis_b[2]):+.4f}], '
					f'dot_A_B={dot_a_b:+.6f}, '
					f'center=[{float(center[0]):+.3f}, {float(center[1]):+.3f}, {float(center[2]):+.3f}], '
					f'height={float(height):.3f}, '
					f'radius={float(cyl_radius):.3f}, '
					f'quat=[{float(quat[0]):+.5f}, {float(quat[1]):+.5f}, {float(quat[2]):+.5f}, {float(quat[3]):+.5f}]'
				)

			marker_array.markers.append(marker)

		self._publisher.publish(marker_array)
