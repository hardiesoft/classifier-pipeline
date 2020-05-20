"""
classifier-pipeline - this is a server side component that manipulates cptv
files and to create a classification model of animals present
Copyright (C) 2018, The Cacophony Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import attr

from config import config
from .defaultconfig import DefaultConfig


@attr.s
class TaggingConfig(DefaultConfig):
    min_track_confidence = attr.ib()
    min_tag_confidence = attr.ib()
    max_tag_novelty = attr.ib()
    min_tag_clarity = attr.ib()
    min_tag_clarity_secondary = attr.ib()
    min_track_frames = attr.ib()
    min_movement = attr.ib()

    @classmethod
    def load(cls, tagging):
        return cls(
            min_track_confidence=tagging["min_track_confidence"],
            min_tag_confidence=tagging["min_tag_confidence"],
            max_tag_novelty=tagging["max_tag_novelty"],
            min_tag_clarity=tagging["min_tag_clarity"],
            min_tag_clarity_secondary=tagging["min_tag_clarity_secondary"],
            min_track_frames=tagging["min_track_frames"],
            min_movement=tagging["min_movement"],
        )

    @classmethod
    def get_defaults(cls):
        return cls(
            min_track_confidence=0.4,
            min_tag_confidence=0.8,
            max_tag_novelty=0.7,
            min_tag_clarity=0.2,
            min_tag_clarity_secondary=0.05,
            min_track_frames=3,
            min_movement=50,
        )

    def validate(self):
        return True

    def as_dict(self):
        return attr.asdict(self)
