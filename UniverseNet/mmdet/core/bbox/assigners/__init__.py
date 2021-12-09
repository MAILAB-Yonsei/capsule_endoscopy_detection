# Copyright (c) OpenMMLab. All rights reserved.
from .approx_max_iou_assigner import ApproxMaxIoUAssigner
from .assign_result import AssignResult
from .atss_assigner import ATSSAssigner
from .atss_cost_assigner import ATSSCostAssigner
from .base_assigner import BaseAssigner
from .center_region_assigner import CenterRegionAssigner
from .grid_assigner import GridAssigner
from .hungarian_assigner import HungarianAssigner
from .max_iou_assigner import MaxIoUAssigner
from .point_assigner import PointAssigner
from .point_kpt_assigner import PointKptAssigner
from .region_assigner import RegionAssigner
from .sim_ota_assigner import SimOTAAssigner
from .task_aligned_assign_result import TaskAlignedAssignResult
from .task_aligned_assigner import TaskAlignedAssigner
from .uniform_assigner import UniformAssigner

__all__ = [
    'BaseAssigner', 'MaxIoUAssigner', 'ApproxMaxIoUAssigner', 'AssignResult',
    'PointAssigner', 'ATSSAssigner', 'CenterRegionAssigner', 'GridAssigner',
    'HungarianAssigner', 'RegionAssigner', 'UniformAssigner', 'SimOTAAssigner'
]

__all__ += [
    'PointKptAssigner', 'ATSSCostAssigner', 'TaskAlignedAssignResult',
    'TaskAlignedAssigner'
]
