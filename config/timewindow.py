""" A window time frame, which can be relative to sunset and sunrise
"""
from datetime import datetime, timedelta
import logging

from astral import Location

from ml_tools import tools


class TimeWindow:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.location = None
        self.last_sunrise_check = None

    def use_sunrise_sunset(self):
        return self.start.is_relative or self.end.is_relative

    def inside_window(self):
        if self.use_sunrise_sunset:
            self.update_sun_times()
            if self.end.time < self.start.time:
                return self.start.is_after() or self.end.is_before()
        return self.start.is_after() and self.end.is_before()

    def update_sun_times(self):
        if not self.use_sunrise_sunset:
            return

        date = datetime.now().date()
        if self.last_sunrise_check is None or date > self.last_sunrise_check:
            sun_times = self.location.sun()
            self.last_sunrise_check = date

            if self.start.is_relative:
                self.start.time = (
                    sun_times["sunset"] + timedelta(seconds=self.start.offset_s)
                ).time()
            if self.end.is_relative:
                self.end.time = (
                    sun_times["sunrise"] + timedelta(seconds=self.end.offset_s)
                ).time()
            logging.info(
                "start_rec is {} end_rec is {}".format(self.start.time, self.end.time)
            )

    def set_location(self, location_config):
        self.location = Location()
        lat, lng = location_config.get_lat_long(use_default=True)
        self.location.latitude = lat
        self.location.longitude = lng

        self.location.altitude = location_config.altitude
        self.location.timezone = tools.get_timezone_str(lat, lng)


class RelAbsTime:
    def __init__(self, time_str, default_offset=None, default_time=None):
        self.is_relative = False
        self.offset_s = 0
        self.time = None
        if time_str == "":
            self.any_time = True
            return

        try:
            self.any_time = False
            self.time = datetime.strptime(time_str, "%H:%M").time()
        except:
            if default_time:
                self.time = default_time
            else:
                self.is_relative = True
                self.offset_s = self.parse_duration(time_str, default_offset)

    def is_after(self):
        return self.any_time or datetime.now().time() > self.time

    def is_before(self):
        return self.any_time or datetime.now().time() < self.time

    def parse_duration(self, time_str, default_offset=None):

        if not time_str:
            return default_offset

        time_type = time_str[-1]
        if time_type.isalpha():
            try:
                offset = int(time_str[:-1])
            except ValueError:
                return default_offset
            if time_type == "s":
                return offset
            elif time_type == "m":
                return offset * 60
            elif time_type == "h":
                return offset * 60 * 60
            return offset

        try:
            offset = int(time_str[:-1])
            return offset * 60
        except ValueError:
            pass
        return default_offset
