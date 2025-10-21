# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Dict, Tuple, Type

from judo.tasks.base import Task, TaskConfig
from judo.tasks.caltech_leap_cube import CaltechLeapCube, CaltechLeapCubeConfig
from judo.tasks.cartpole import Cartpole, CartpoleConfig
from judo.tasks.cylinder_push import CylinderPush, CylinderPushConfig
from judo.tasks.fr3_pick import FR3Pick, FR3PickConfig
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig
from judo.tasks.leap_cube_down import LeapCubeDown, LeapCubeDownConfig
from judo.tasks.spot_door_box import SpotDoorBox, SpotDoorBoxConfig
from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_box import SpotBox, SpotBoxConfig
from judo.tasks.spot.spot_yellow_chair import SpotYellowChair, SpotYellowChairConfig
from judo.tasks.spot.spot_yellow_chair_ramp import SpotYellowChairRamp, SpotYellowChairRampConfig

_registered_tasks: Dict[str, Tuple[Type[Task], Type[TaskConfig]]] = {
    "spot_base": (SpotBase, SpotBaseConfig),
    "spot_door_box": (SpotDoorBox, SpotDoorBoxConfig),
    "spot_box": (SpotBox, SpotBoxConfig),
    "spot_yellow_chair": (SpotYellowChair, SpotYellowChairConfig),
    "spot_yellow_chair_ramp": (SpotYellowChairRamp, SpotYellowChairRampConfig),
    "cylinder_push": (CylinderPush, CylinderPushConfig),
    "cartpole": (Cartpole, CartpoleConfig),
    "fr3_pick": (FR3Pick, FR3PickConfig),
    "leap_cube": (LeapCube, LeapCubeConfig),
    "leap_cube_down": (LeapCubeDown, LeapCubeDownConfig),
    "caltech_leap_cube": (CaltechLeapCube, CaltechLeapCubeConfig),
}


def get_registered_tasks() -> Dict[str, Tuple[Type[Task], Type[TaskConfig]]]:
    """Returns a dictionary of registered tasks."""
    return _registered_tasks


def register_task(name: str, task_type: Type[Task], task_config_type: Type[TaskConfig]) -> None:
    """Registers a new task."""
    _registered_tasks[name] = (task_type, task_config_type)


__all__ = [
    "get_registered_tasks",
    "register_task",
    "Task",
    "TaskConfig",
    "CaltechLeapCube",
    "CaltechLeapCubeConfig",
    "Cartpole",
    "CartpoleConfig",
    "CylinderPush",
    "CylinderPushConfig",
    "FR3Pick",
    "FR3PickConfig",
    "LeapCube",
    "LeapCubeConfig",
    "LeapCubeDown",
    "LeapCubeDownConfig",
    "SpotBase",
    "SpotBaseConfig",
    "SpotBox",
    "SpotBoxConfig",
    "SpotYellowChair",
    "SpotYellowChairConfig",
    "SpotYellowChairRamp",
    "SpotYellowChairRampConfig",
]
