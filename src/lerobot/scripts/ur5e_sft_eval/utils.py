# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Ported from VLA-Precision; see provenance.json for source and changes.

"""Helpers shared by the native OpenPI inference backends."""

import logging
import time


class FpsCounter:
    """Utility class to measure the frequency of a loop"""

    def __init__(self, name: str, log_interval: float = 1.0):
        self.name = name
        self.last_time = time.perf_counter()

    def reset(self):
        """Reset the timer to the current time."""
        self.last_time = time.perf_counter()

    def update(self, show: bool = True):
        """Compute and log the frequency when show is enabled."""
        if not show:
            return
        now = time.perf_counter()
        fps = 1.0 / (now - self.last_time)
        self.last_time = now
        logging.info(f"[DEBUG] {self.name} loop actual frequency: {fps:.2f} Hz")
