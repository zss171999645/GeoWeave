# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Stub distortion helpers for official VGGT geometry utilities.

The original dependency module is not included in this repo. These stubs
raise explicit errors if called so training usage remains unaffected.
"""


def _missing_dependency(name: str):
    raise NotImplementedError(
        f"{name} requires the original VGGT distortion dependency, which is not "
        "vendored in this project."
    )


def apply_distortion(*args, **kwargs):
    _missing_dependency("apply_distortion")


def iterative_undistortion(*args, **kwargs):
    _missing_dependency("iterative_undistortion")


def single_undistortion(*args, **kwargs):
    _missing_dependency("single_undistortion")
