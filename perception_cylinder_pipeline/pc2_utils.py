from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


@dataclass(frozen=True)
class ParsedPointCloud:
	frame_id: str
	raw_xyz: np.ndarray
	valid_xyz: np.ndarray
	valid_rgb: np.ndarray


def _field_offset(msg: PointCloud2, name: str) -> int:
	for field in msg.fields:
		if field.name == name:
			if field.datatype != PointField.FLOAT32:
				raise ValueError(f"PointField '{name}' is not FLOAT32")
			return field.offset
	raise ValueError(f"PointField '{name}' was not found")


def _field_by_name(msg: PointCloud2, name: str) -> PointField:
	for field in msg.fields:
		if field.name == name:
			return field
	raise ValueError(f"PointField '{name}' was not found")


def pointcloud2_to_xyz_arrays(msg: PointCloud2) -> ParsedPointCloud:
	total_points = int(msg.width) * int(msg.height)
	if total_points == 0 or msg.point_step == 0:
		empty = np.empty((0, 3), dtype=np.float32)
		empty_rgb = np.empty((0, 3), dtype=np.uint8)
		return ParsedPointCloud(frame_id=msg.header.frame_id, raw_xyz=empty, valid_xyz=empty, valid_rgb=empty_rgb)

	x_offset = _field_offset(msg, 'x')
	y_offset = _field_offset(msg, 'y')
	z_offset = _field_offset(msg, 'z')
	rgb_field = _field_by_name(msg, 'rgb')
	if rgb_field.offset + 4 > msg.point_step:
		raise ValueError("PointField 'rgb' does not fit inside point_step")

	expected_bytes = total_points * msg.point_step
	data_bytes = np.frombuffer(msg.data, dtype=np.uint8)
	if data_bytes.size < expected_bytes:
		total_points = data_bytes.size // msg.point_step
		expected_bytes = total_points * msg.point_step

	if total_points == 0:
		empty = np.empty((0, 3), dtype=np.float32)
		empty_rgb = np.empty((0, 3), dtype=np.uint8)
		return ParsedPointCloud(frame_id=msg.header.frame_id, raw_xyz=empty, valid_xyz=empty, valid_rgb=empty_rgb)

	point_rows = data_bytes[:expected_bytes].reshape(total_points, msg.point_step)
	float_dtype = np.dtype('>f4' if msg.is_bigendian else '<f4')

	x = point_rows[:, x_offset:x_offset + 4].copy().view(float_dtype).reshape(-1)
	y = point_rows[:, y_offset:y_offset + 4].copy().view(float_dtype).reshape(-1)
	z = point_rows[:, z_offset:z_offset + 4].copy().view(float_dtype).reshape(-1)

	raw_xyz = np.column_stack((x, y, z)).astype(np.float32, copy=False)
	valid_mask = np.isfinite(raw_xyz).all(axis=1)
	valid_xyz = raw_xyz[valid_mask]

	packed_dtype = np.dtype('>u4' if msg.is_bigendian else '<u4')
	packed_rgb = point_rows[:, rgb_field.offset:rgb_field.offset + 4].copy().view(packed_dtype).reshape(-1)
	r = ((packed_rgb >> 16) & 0xFF).astype(np.uint8, copy=False)
	g = ((packed_rgb >> 8) & 0xFF).astype(np.uint8, copy=False)
	b = (packed_rgb & 0xFF).astype(np.uint8, copy=False)
	raw_rgb = np.column_stack((r, g, b))
	valid_rgb = raw_rgb[valid_mask]

	return ParsedPointCloud(
		frame_id=msg.header.frame_id,
		raw_xyz=raw_xyz,
		valid_xyz=valid_xyz,
		valid_rgb=valid_rgb,
	)


def _rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
	rgb_array = np.asarray(rgb)
	if rgb_array.ndim != 2 or rgb_array.shape[1] != 3:
		raise ValueError('rgb must have shape (N, 3)')

	if np.issubdtype(rgb_array.dtype, np.floating):
		if rgb_array.size == 0:
			return rgb_array.astype(np.uint8, copy=False)
		if np.nanmax(rgb_array) <= 1.0:
			rgb_array = rgb_array * 255.0
		rgb_array = np.clip(rgb_array, 0.0, 255.0).astype(np.uint8)
	else:
		rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)

	return rgb_array


def build_pointcloud2_from_xyz_rgb(header: Header, xyz: np.ndarray, rgb: np.ndarray) -> PointCloud2:
	xyz_array = np.asarray(xyz, dtype=np.float32)
	if xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
		raise ValueError('xyz must have shape (N, 3)')

	rgb_array = _rgb_to_uint8(rgb)
	if rgb_array.shape[0] != xyz_array.shape[0]:
		raise ValueError('xyz and rgb must have the same number of rows')

	point_count = xyz_array.shape[0]
	point_step = 16
	row_step = point_count * point_step

	buffer = np.zeros((point_count, point_step), dtype=np.uint8)
	x_bytes = np.ascontiguousarray(xyz_array[:, 0], dtype='<f4').view(np.uint8).reshape(-1, 4)
	y_bytes = np.ascontiguousarray(xyz_array[:, 1], dtype='<f4').view(np.uint8).reshape(-1, 4)
	z_bytes = np.ascontiguousarray(xyz_array[:, 2], dtype='<f4').view(np.uint8).reshape(-1, 4)
	buffer[:, 0:4] = x_bytes
	buffer[:, 4:8] = y_bytes
	buffer[:, 8:12] = z_bytes

	packed_rgb_u32 = (
		(rgb_array[:, 0].astype(np.uint32) << 16)
		| (rgb_array[:, 1].astype(np.uint32) << 8)
		| rgb_array[:, 2].astype(np.uint32)
	)
	packed_rgb_f32 = packed_rgb_u32.astype('<u4', copy=False).view('<f4')
	buffer[:, 12:16] = packed_rgb_f32.view(np.uint8).reshape(-1, 4)

	out = PointCloud2()
	out.header = header
	out.height = 1
	out.width = point_count
	out.fields = [
		PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
		PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
		PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
		PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
	]
	out.is_bigendian = False
	out.point_step = point_step
	out.row_step = row_step
	out.is_dense = bool(point_count == 0 or np.isfinite(xyz_array).all())
	out.data = buffer.tobytes()
	return out
