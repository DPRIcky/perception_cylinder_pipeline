from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description() -> LaunchDescription:
	package_name = 'perception_cylinder_pipeline'
	rviz_default = os.path.join(
		get_package_share_directory(package_name),
		'rviz',
		'perception_pipeline.rviz',
	)

	use_rviz_arg = DeclareLaunchArgument(
		'use_rviz',
		default_value='true',
		description='Launch RViz with the saved configuration.',
	)

	rviz_config_arg = DeclareLaunchArgument(
		'rviz_config',
		default_value=rviz_default,
		description='Absolute path to an RViz config file.',
	)

	perception_node = Node(
		package=package_name,
		executable='cylinder_processor_node',
		name='cylinder_processor_node',
		output='screen',
	)

	rviz_node = Node(
		package='rviz2',
		executable='rviz2',
		name='rviz2',
		arguments=['-d', LaunchConfiguration('rviz_config')],
		output='screen',
		condition=IfCondition(LaunchConfiguration('use_rviz')),
	)

	return LaunchDescription([
		use_rviz_arg,
		rviz_config_arg,
		perception_node,
		rviz_node,
	])
