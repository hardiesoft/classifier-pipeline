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


import logging
import numpy as np

from cptv import CPTVReader
import cv2
from .trackextractor import DefaultTrackExtractor

from .clip import Clip
import ml_tools.tools as tools
from ml_tools.tools import Rectangle
from track.region import Region
from track.track import Track
from piclassifier.motiondetector import is_affected_by_ffc


class ClipTrackExtractor:
    PREVIEW = "preview"

    def __init__(
        self, config, use_opt_flow, cache_to_disk, keep_frames=True, calc_stats=True
    ):
        self.config = config
        self.use_opt_flow = use_opt_flow
        self.stats = None
        self.cache_to_disk = cache_to_disk
        self.max_tracks = config.max_tracks
        # frame_padding < 3 causes problems when we get small areas...
        self.frame_padding = max(3, self.config.frame_padding)
        # the dilation effectively also pads the frame so take it into consideration.
        self.frame_padding = max(0, self.frame_padding - self.config.dilation_pixels)
        self.keep_frames = keep_frames
        self.calc_stats = calc_stats
        if self.config.dilation_pixels > 0:
            size = self.config.dilation_pixels * 2 + 1
            self.dilate_kernel = np.ones((size, size), np.uint8)

        self.track_extractor = None

    def parse_clip(self, clip):
        """
        Loads a cptv file, and prepares for track extraction.
        """
        clip.set_frame_buffer(
            self.config.high_quality_optical_flow,
            self.cache_to_disk,
            self.use_opt_flow,
            self.keep_frames,
        )
        with open(clip.source_file, "rb") as f:
            reader = CPTVReader(f)
            clip.set_res(reader.x_resolution, reader.y_resolution)
            video_start_time = reader.timestamp.astimezone(Clip.local_tz)
            clip.num_preview_frames = (
                reader.preview_secs * clip.frames_per_second - self.config.ignore_frames
            )
            clip.set_video_stats(video_start_time)

            self.track_extractor = DefaultTrackExtractor(
                clip,
                self.config,
                self.config.high_quality_optical_flow,
                self.cache_to_disk,
                self.use_opt_flow,
                self.keep_frames,
            )
            if clip.background_is_preview and clip.num_preview_frames > 0:
                for frame in reader:
                    self.track_extractor.process_frame(
                        frame.pix, is_affected_by_ffc(frame)
                    )

                if not self.track_extractor.tracking:
                    logging.warn("Clip is all preview frames")
                    if not self.track_extractor.has_non_ffc_frames:
                        logging.warn("Clip only has ffc affected frames")
                        return False

                self.track_extracotr.completed()
            else:
                # we need to load the entire video so we can analyse the background
                clip.background_is_preview = False
                self.process_frames(clip, [frame for frame in reader])

        if not clip.from_metadata:
            self.apply_track_filtering(clip)

        if self.calc_stats:
            clip.stats.completed(clip.frame_on, clip.res_y, clip.res_x)

        return True

    def _whole_clip_stats(self, clip, frames):
        filtered = np.float32(
            [self._get_filtered_frame(clip, frame) for frame in frames]
        )

        delta = np.asarray(frames[1:], dtype=np.float32) - np.asarray(
            frames[:-1], dtype=np.float32
        )
        average_delta = float(np.nanmean(np.abs(delta)))

        # take half the max filtered value as a threshold
        threshold = float(
            np.percentile(
                np.reshape(filtered, [-1]), q=self.config.threshold_percentile
            )
        )

        # cap the threshold to something reasonable
        threshold = max(self.config.min_threshold, threshold)
        threshold = min(self.config.max_threshold, threshold)

        clip.threshold = threshold
        if self.calc_stats:
            clip.stats.threshold = threshold
            clip.stats.temp_thresh = self.config.temp_thresh
            clip.stats.average_delta = float(average_delta)
            clip.stats.filtered_deviation = float(np.mean(np.abs(filtered)))
            clip.stats.is_static_background = (
                clip.stats.filtered_deviation < clip.config.static_background_threshold
            )

            if (
                not clip.stats.is_static_background
                or clip.disable_background_subtraction
            ):
                clip.background = None

    def apply_track_filtering(self, clip):
        self.filter_tracks(clip)
        # apply smoothing if required
        if self.config.track_smoothing and clip.frame_on > 0:
            for track in clip.active_tracks:
                track.smooth(Rectangle(0, 0, clip.res_x, clip.res_y))

    def filter_tracks(self, clip):

        for track in clip.tracks:
            track.trim()
            track.set_end(clip.frames_per_second)

        track_stats = [(track.get_stats(), track) for track in clip.tracks]
        track_stats.sort(reverse=True, key=lambda record: record[0].score)

        if self.config.verbose:
            for stats, track in track_stats:
                start_s, end_s = clip.start_and_end_in_secs(track)
                logging.info(
                    " - track duration: %.1fsec, number of frames:%s, offset:%.1fpx, delta:%.1f, mass:%.1fpx",
                    end_s - start_s,
                    len(track),
                    stats.max_offset,
                    stats.delta_std,
                    stats.average_mass,
                )
        # filter out tracks that probably are just noise.
        good_tracks = []
        self.print_if_verbose(
            "{} {}".format("Number of tracks before filtering", len(clip.tracks))
        )

        for stats, track in track_stats:
            # discard any tracks that overlap too often with other tracks.  This normally means we are tracking the
            # tail of an animal.
            if not self.filter_track(clip, track, stats):
                good_tracks.append(track)

        clip.tracks = good_tracks
        self.print_if_verbose(
            "{} {}".format("Number of 'good' tracks", len(clip.tracks))
        )
        # apply max_tracks filter
        # note, we take the n best tracks.
        if self.max_tracks is not None and self.max_tracks < len(clip.tracks):
            logging.warning(
                " -using only {0} tracks out of {1}".format(
                    self.max_tracks, len(clip.tracks)
                )
            )
            clip.filtered_tracks.extend(
                [("Too many tracks", track) for track in clip.tracks[self.max_tracks :]]
            )
            clip.tracks = clip.tracks[: self.max_tracks]

    def filter_track(self, clip, track, stats):
        # discard any tracks that are less min_duration
        # these are probably glitches anyway, or don't contain enough information.
        if len(track) < self.config.min_duration_secs * 9:
            self.print_if_verbose("Track filtered. Too short, {}".format(len(track)))
            clip.filtered_tracks.append(("Track filtered.  Too much overlap", track))
            return True

        # discard tracks that do not move enough
        if stats.max_offset < self.config.track_min_offset:
            self.print_if_verbose("Track filtered.  Didn't move")
            clip.filtered_tracks.append(("Track filtered.  Didn't move", track))

            return True

        # discard tracks that do not have enough delta within the window (i.e. pixels that change a lot)
        if stats.delta_std < self.config.track_min_delta:
            self.print_if_verbose("Track filtered.  Too static")
            clip.filtered_tracks.append(("Track filtered.  Too static", track))
            return True

        # discard tracks that do not have enough enough average mass.
        if stats.average_mass < self.config.track_min_mass:
            self.print_if_verbose(
                "Track filtered.  Mass too small ({})".format(stats.average_mass)
            )
            clip.filtered_tracks.append(("Track filtered.  Mass too small", track))

            return True

        highest_ratio = 0
        for other in clip.tracks:
            if track == other:
                continue
            highest_ratio = max(track.get_overlap_ratio(other), highest_ratio)

        if highest_ratio > self.config.track_overlap_ratio:
            self.print_if_verbose(
                "Track filtered.  Too much overlap {}".format(highest_ratio)
            )
            clip.filtered_tracks.append(("Track filtered.  Too much overlap", track))
            return True

        return False

    def print_if_verbose(self, info_string):
        if self.config.verbose:
            logging.info(info_string)
