# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Record files and latest links for single-robot UR evaluation runs."""

import os
import threading
from pathlib import Path


def publish_latest(target: Path, link: Path) -> None:
    """Atomically point a latest link at an absolute run-file path."""
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.{os.getpid()}.{threading.get_ident()}")
    try:
        temporary.symlink_to(target.resolve())
        temporary.replace(link)
    finally:
        temporary.unlink(missing_ok=True)
