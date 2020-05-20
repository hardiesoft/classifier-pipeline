import ml_tools.tools as tools
from abc import ABC, abstractmethod
from track.framebuffer import FrameBuffer
from track.track import Track
import numpy as np
import cv2


class Background(ABC):
    @abstractmethod
    def add_frame(self, frame):
        """ The function to get default config. """
        ...

    @abstractmethod
    def get_background(self):
        """ The function to get default config. """
        ...

    @abstractmethod
    def get_temp_thresh(self):
        """ The function to get default config. """
        ...

    @property
    @abstractmethod
    def calculated(self):
        """ The function to get default config. """
        ...

    @property
    @abstractmethod
    def num_frames(self):
        """ The function to get default config. """
        ...

    @property
    @abstractmethod
    def frames(self):
        """ The function to get default config. """
        ...


class TrackExtractor(ABC):
    @abstractmethod
    def _process_preview_frames(self, clip):
        """ The function to get default config. """
        ...

    @abstractmethod
    def process_frame(self, clip, frame, ffc_affected=False):
        """ The function to get default config. """
        ...


class DefaultBackground(Background):
    def __init__(self, dynamic_thresh, default_thresh, preview_frames):
        self.temp_thresh = default_thresh
        self.dynamic_thresh = dynamic_thresh
        self.background_frames = 0
        self.preview_frames = []
        self.background = None
        self.num_preview_frames = preview_frames
        self.background_calculated = False
        self.mean_background_value = 0

    @property
    def calculated(self):
        return self.background_calculated

    @property
    def frames(self):
        return self.preview_frames

    @property
    def num_frames(self):
        return len(self.preview_frames)

    def mean_value(self):
        return self.mean_background_value

    def clear_frames(self):
        return self.preview_frames

    def get_background(self):
        return self.background

    def get_temp_thresh(self):
        return self.temp_thresh

    def add_frame(self, frame, ffc_affected):
        print("preview frames", self.num_preview_frames)
        self.preview_frames.append((frame, ffc_affected))
        if ffc_affected:
            return
        if self.background is None:
            self.background = frame
        else:
            self.background = np.minimum(self.background, frame)
        if self.background_frames == (self.num_preview_frames - 1):
            self._set_from_background()
        self.background_frames += 1

    def _set_from_background(self):
        self.mean_background_value = np.average(self.background)
        self.set_temp_thresh()
        self.background_calculated = True

    def set_temp_thresh(self):
        if self.dynamic_thresh:
            self.temp_thresh = min(self.temp_thresh, self.mean_background_value)

    def background_from_whole_clip(self, frames):
        """
        Runs through all provided frames and estimates the background, consuming all the source frames.
        :param frames_list: a list of numpy array frames
        :return: background
        """

        self.background = np.percentile(frames, q=10, axis=0)
        self._set_from_background()


class DefaultTrackExtractor(TrackExtractor):
    def __init__(
        self, clip, config, high_quality_flow, cache_to_disk, use_flow, keep_frames
    ):
        print(config)
        self.clip = clip
        self.background_calculator = DefaultBackground(
            config.dynamic_thresh, config.temp_thresh, clip.num_preview_frames
        )
        # self.frame_buffer = FrameBuffer(
        #     self.clip.source_file,
        #     high_quality_flow,
        #     cache_to_disk,
        #     use_flow,
        #     keep_frames,
        # )
        self.config = config
        self.frame_on = 0
        self.ffc_affected = False
        self.tracking = False
        self.has_non_ffc_frames = False
        Track._track_id = 1
        if self.config.dilation_pixels > 0:
            size = self.config.dilation_pixels * 2 + 1
            self.dilate_kernel = np.ones((size, size), np.uint8)

    def completed():
        self.background_calculator.set_from_background()
        self._process_preview_frames(clip)
        clip.frame_on = self.frame_on

    def process_frames(self, raw_frames):
        # for now just always calculate as we are using the stats...
        # background np.float64[][] filtered calculated here and stats
        non_ffc_frames = [
            frame.pix for frame in raw_frames if not is_affected_by_ffc(frame)
        ]
        if len(non_ffc_frames) == 0:
            logging.warn("Clip only has ffc affected frames")
            return False
        self.background_calculator.background_from_whole_clip(non_ffc_frames)
        # not sure if we want to include ffc frames here or not
        self._whole_clip_stats(clip, non_ffc_frames)
        if self.clip.background_is_preview:
            if clip.num_preview_frames > 0:
                clip.background_from_frames(raw_frames)
            else:
                logging.info(
                    "No preview secs defined for CPTV file - using statistical background measurement"
                )

        # process each frame
        for frame in raw_frames:
            ffc_affected = is_affected_by_ffc(frame)
            self._process_frame(clip, frame.pix, ffc_affected)
            self.frame_on += 1

    def process_frame(self, frame, ffc_affected=False):
        if ffc_affected:
            self.print_if_verbose("{} ffc_affected".format(self.frame_on))
        else:
            has_non_ffc_frames = True
        self.ffc_affected = ffc_affected
        if not self.background_calculator.calculated:
            print(frame)
            self.background_calculator.add_frame(frame, ffc_affected)
            print("addiing background")
            if self.background_calculator.calculated:
                self.frame_on += 1
                self._process_preview_frames()
                return
        else:
            self._process_frame(self.clip, frame, ffc_affected)
        self.frame_on += 1

    def _process_preview_frames(self):
        self.frame_on -= self.background_calculator.num_frames
        for _, back_frame in enumerate(self.background_calculator.frames):
            self._process_frame(self.clip, back_frame[0], back_frame[1])
            self.frame_on += 1

        self.background_calculator.clear_frames()

    def _get_filtered_frame(self, thermal):
        """
        Calculates filtered frame from thermal
        :param thermal: the thermal frame
        :param background: (optional) used for background subtraction
        :return: the filtered frame
        """

        # has to be a signed int so we dont get overflow
        filtered = np.float32(thermal.copy())
        if self.background_calculator.get_background() is None:
            filtered = filtered - np.median(filtered) - 40
            filtered[filtered < 0] = 0
        elif self.clip.background_is_preview:
            avg_change = int(
                round(np.average(thermal) - self.background_calculator.mean_value())
            )
            filtered[filtered < self.background_calculator.get_temp_thresh()] = 0
            np.clip(
                filtered - self.background_calculator.get_background() - avg_change,
                0,
                None,
                out=filtered,
            )

        else:
            filtered = filtered - self.background_calculator.get_background()
            filtered = filtered - np.median(filtered)
            filtered[filtered < 0] = 0
        return filtered

    def _process_frame(self, clip, thermal, ffc_affected=False):
        """
        Tracks objects through frame
        :param thermal: A numpy array of shape (height, width) and type uint16
            If specified background subtraction algorithm will be used.
        """
        self.tracking = True
        print(thermal)
        filtered = self._get_filtered_frame(thermal)
        frame_height, frame_width = filtered.shape
        mask = np.zeros(filtered.shape)
        edge = self.config.edge_pixels

        # remove the edges of the frame as we know these pixels can be spurious value
        edgeless_filtered = clip.crop_rectangle.subimage(filtered)
        thresh, mass = tools.blur_and_return_as_mask(
            edgeless_filtered, threshold=clip.threshold
        )
        thresh = np.uint8(thresh)
        dilated = thresh

        # Dilation groups interested pixels that are near to each other into one component(animal/track)
        if self.config.dilation_pixels > 0:
            dilated = cv2.dilate(dilated, self.dilate_kernel, iterations=1)

        labels, small_mask, stats, _ = cv2.connectedComponentsWithStats(dilated)
        mask[edge : frame_height - edge, edge : frame_width - edge] = small_mask

        prev_filtered = self.clip.frame_buffer.get_last_filtered()
        self.add_frame(thermal, filtered, mask, ffc_affected)

        if clip.from_metadata:
            for track in clip.tracks:
                if self.frame_on in track.frame_list:

                    track.add_frame_for_existing_region(
                        self.clip.frame_buffer.get_last_frame(),
                        clip.threshold,
                        prev_filtered,
                    )
        else:
            regions = self._get_regions_of_interest(
                clip, labels, stats, thresh, filtered, prev_filtered, mass
            )
            self.region_history.append(regions)
            self._apply_region_matchings(
                clip, regions, create_new_tracks=not ffc_affected
            )

    def _apply_region_matchings(self, clip, regions, create_new_tracks=True):
        """
        Work out the best matchings between tracks and regions of interest for the current frame.
        Create any new tracks required.
        """
        unmatched_regions, matched_tracks = self._match_existing_tracks(clip, regions)
        if create_new_tracks:
            new_tracks = self._create_new_tracks(clip, unmatched_regions)
        else:
            new_tracks = set()
        self._filter_inactive_tracks(clip, new_tracks, matched_tracks)

    def get_max_size_change(self, track, region):
        exiting = region.is_along_border and not track.last_bound.is_along_border
        entering = not exiting and track.last_bound.is_along_border

        min_change = 50
        if entering or exiting:
            min_change = 100
        max_size_change = np.clip(track.last_mass, min_change, 500)
        return max_size_change

    def _match_existing_tracks(self, clip, regions):

        scores = []
        used_regions = set()
        unmatched_regions = set(regions)
        for track in clip.active_tracks:
            for region in regions:
                score, size_change = track.get_track_region_score(
                    region, self.config.moving_vel_thresh
                )
                # we give larger tracks more freedom to find a match as they might move quite a bit.
                max_distance = np.clip(7 * track.last_mass, 900, 9025)
                max_size_change = self.get_max_size_change(track, region)

                if score > max_distance:
                    self.print_if_verbose(
                        "track {} distance score {} bigger than max score {}".format(
                            track.get_id(), score, max_distance
                        )
                    )

                    continue
                if size_change > max_size_change:
                    self.print_if_verbose(
                        "track {} size_change {} bigger than max size_change {}".format(
                            track.get_id(), size_change, max_size_change
                        )
                    )
                    continue
                scores.append((score, track, region))

        # makes tracking consistent by ordering by score then by frame since target then track id
        scores.sort(
            key=lambda record: record[1].frames_since_target_seen
            + float(".{}".format(record[1]._id))
        )
        scores.sort(key=lambda record: record[0])

        matched_tracks = set()
        for (score, track, region) in scores:
            if track in matched_tracks or region in used_regions:
                continue

            track.add_region(region)
            matched_tracks.add(track)
            used_regions.add(region)
            unmatched_regions.remove(region)

        return unmatched_regions, matched_tracks

    def _create_new_tracks(self, clip, unmatched_regions):
        """ Create new tracks for any unmatched regions """
        new_tracks = set()
        for region in unmatched_regions:
            # make sure we don't overlap with existing tracks.  This can happen if a tail gets tracked as a new object
            overlaps = [
                track.last_bound.overlap_area(region) for track in clip.active_tracks
            ]
            if len(overlaps) > 0 and max(overlaps) > (region.area * 0.25):
                continue

            track = Track.from_region(clip, region)
            new_tracks.add(track)
            clip._add_active_track(track)
            self.print_if_verbose(
                "Creating a new track {} with region {} mass{} area {}".format(
                    track.get_id(), region, track.last_bound.mass, track.last_bound.area
                )
            )
        return new_tracks

    def _filter_inactive_tracks(self, clip, new_tracks, matched_tracks):
        """ Filters tracks which are or have become inactive """

        unactive_tracks = clip.active_tracks - matched_tracks - new_tracks
        clip.active_tracks = matched_tracks | new_tracks
        for track in unactive_tracks:
            if (
                track.frames_since_target_seen + 1
                < self.config.remove_track_after_frames
            ):
                track.add_blank_frame(self.clip.frame_buffer)
                clip.active_tracks.add(track)
                self.print_if_verbose(
                    "frame {} adding a blank frame to {} ".format(
                        self.frame_on, track.get_id()
                    )
                )

    def _get_regions_of_interest(
        self, clip, labels, stats, thresh, filtered, prev_filtered, mass
    ):
        """
        Calculates pixels of interest mask from filtered image, and returns both the labeled mask and their bounding
        rectangles.
        :param filtered: The filtered frame
=        :return: regions of interest, mask frame
        """

        if prev_filtered is not None:
            delta_frame = np.abs(filtered - prev_filtered)
        else:
            delta_frame = None

        # we enlarge the rects a bit, partly because we eroded them previously, and partly because we want some context.
        padding = self.frame_padding
        edge = self.config.edge_pixels
        # find regions of interest
        regions = []
        for i in range(1, labels):

            region = Region(
                stats[i, 0],
                stats[i, 1],
                stats[i, 2],
                stats[i, 3],
                stats[i, 4],
                0,
                i,
                clip.frame_on,
            )
            # want the real mass calculated from before the dilation
            # region.mass = np.sum(region.subimage(thresh))
            region.mass = mass
            # Add padding to region and change coordinates from edgeless image -> full image
            region.x += edge - padding
            region.y += edge - padding
            region.width += padding * 2
            region.height += padding * 2

            old_region = region.copy()
            region.crop(clip.crop_rectangle)
            region.was_cropped = str(old_region) != str(region)
            region.set_is_along_border(clip.crop_rectangle)
            if self.config.cropped_regions_strategy == "cautious":
                crop_width_fraction = (
                    old_region.width - region.width
                ) / old_region.width
                crop_height_fraction = (
                    old_region.height - region.height
                ) / old_region.height
                if crop_width_fraction > 0.25 or crop_height_fraction > 0.25:
                    continue
            elif self.config.cropped_regions_strategy == "none":
                if region.was_cropped:
                    continue
            elif self.config.cropped_regions_strategy != "all":
                raise ValueError(
                    "Invalid mode for CROPPED_REGIONS_STRATEGY, expected ['all','cautious','none'] but found {}".format(
                        self.config.cropped_regions_strategy
                    )
                )

            if delta_frame is not None:
                region_difference = region.subimage(delta_frame)
                region.pixel_variance = np.var(region_difference)

            # filter out regions that are probably just noise
            if (
                region.pixel_variance < self.config.aoi_pixel_variance
                and region.mass < self.config.aoi_min_mass
            ):
                continue
            regions.append(region)
        return regions

    def add_frame(self, thermal, filtered, mask, ffc_affected=False):
        self.clip.frame_buffer.add_frame(
            thermal, filtered, mask, self.frame_on, ffc_affected
        )
        if self.calc_stats:
            self.stats.add_frame(thermal, filtered)

    def _whole_clip_stats(self, clip, frames):
        filtered = np.float32([self._get_filtered_frame(frame) for frame in frames])

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
