#!/usr/bin/python3
from datetime import timedelta
import numpy as np
import os
import logging
import socket
import time
import absl.logging
import psutil
from cptv import Frame

from .telemetry import Telemetry
from .locationconfig import LocationConfig
from .thermalconfig import ThermalConfig
from .motiondetector import MotionDetector
from .cptvrecorder import CPTVRecorder

from ml_tools.logs import init_logging
from ml_tools.config import Config


SOCKET_NAME = "/var/run/lepton-frames"
VOSPI_DATA_SIZE = 160
TELEMETRY_PACKET_COUNT = 4


# Links to socket and continuously waits for 1 connection
def main():
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    init_logging()
    try:
        os.unlink(SOCKET_NAME)
    except OSError:
        if os.path.exists(SOCKET_NAME):
            raise
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_RCVBUF,
        400 * 400 * 2 + TELEMETRY_PACKET_COUNT * VOSPI_DATA_SIZE,
    )
    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_SNDBUF,
        400 * 400 * 2 + TELEMETRY_PACKET_COUNT * VOSPI_DATA_SIZE,
    )

    sock.bind(SOCKET_NAME)
    sock.listen(1)
    config = Config.load_from_file()
    thermal_config = ThermalConfig.load_from_file()
    location_config = LocationConfig.load_from_file()

    clip_classifier = PiClassifier(config, thermal_config, location_config)
    while True:
        logging.info("waiting for a connection")
        connection, client_address = sock.accept()
        logging.info("connection from %s", client_address)
        try:
            start = time.time()
            handle_connection(connection, clip_classifier)
            # print(psutil.cpu_percent())
            # print(psutil.virtual_memory())  # physical memory usage
        finally:
            # Clean up the connection
            connection.close()
            print("took {} seconds".format(time.time() - start))


def handle_connection(connection, clip_classifier):
    img_dtype = np.dtype("uint16")
    # big endian > little endian <
    # lepton3 is big endian while python is little endian

    thermal_frame = np.empty(
        (clip_classifier.res_y, clip_classifier.res_x), dtype=img_dtype
    )

    max_cpu = 0
    max_mem = 0
    buf = bytearray(400 * 400 * 2 + TELEMETRY_PACKET_COUNT * VOSPI_DATA_SIZE)
    view = memoryview(buf)
    while True:
        start_data = time.time()
        nbytes = connection.recv_into(
            view, 400 * 400 * 2 + TELEMETRY_PACKET_COUNT * VOSPI_DATA_SIZE
        )

        # data = connection.recv_into(400 * 400 * 2 + TELEMETRY_PACKET_COUNT * VOSPI_DATA_SIZE)

        # data = connection.recv(400 * 400 * 2 + TELEMETRY_PACKET_COUNT * VOSPI_DATA_SIZE)
        end_data = time.time()
        print("receiving data took {}".format(1000 * 1000 * (end_data - start_data)))
        if not nbytes:
            logging.info("disconnected from camera")
            clip_classifier.disconnected()
            print("max cpu {} max mem {}".format(max_cpu, max_mem))
            print("total classification {}".format(clip_classifier.total_time))

            return

        if nbytes > clip_classifier.res_y * clip_classifier.res_x * 2:
            telemetry = Telemetry.parse_telemetry(
                view[: TELEMETRY_PACKET_COUNT * VOSPI_DATA_SIZE]
            )

            thermal_frame = np.frombuffer(
                view, dtype=img_dtype, offset=TELEMETRY_PACKET_COUNT * VOSPI_DATA_SIZE
            ).reshape(clip_classifier.res_y, clip_classifier.res_x)
        else:
            telemetry = Telemetry()
            telemetry.last_ffc_time = timedelta(milliseconds=time.time())
            telemetry.time_on = timedelta(
                milliseconds=time.time(), seconds=MotionDetector.FFC_PERIOD.seconds + 1
            )
            start_data = time.time()
            thermal_frame = np.frombuffer(
                view[:nbytes], dtype=img_dtype, offset=0
            ).reshape(clip_classifier.res_y, clip_classifier.res_x)

        # swap from big to little endian
        lepton_frame = Frame(
            thermal_frame.byteswap(), telemetry.time_on, telemetry.last_ffc_time
        )
        end_data = time.time()
        print("parsing data took {}".format(1000 * 1000 * (end_data - start_data)))
        # t_max = np.amax(lepton_frame.pix)
        # t_min = np.amin(lepton_frame.pix)
        # if t_max > 10000 or t_min == 0:
        #     logging.warning(
        #         "received frame has odd values skipping thermal frame max {} thermal frame min {}".format(
        #             t_max, t_min
        #         )
        #     )
        #     # this frame has bad data probably from lack of CPU
        #     clip_classifier.skip_frame()
        #     continue

        clip_classifier.process_frame(lepton_frame)
        # max_cpu = max(max_cpu, psutil.cpu_percent())
        # max_mem = max(max_mem, psutil.virtual_memory()[3])

        # print(
        #     "cpu {} memory % used {}:".format(
        #         psutil.cpu_percent(), psutil.virtual_memory()[2]
        #     )
        # )


class PiClassifier:
    """ Classifies frames from leptond """

    PROCESS_FRAME = 3
    NUM_CONCURRENT_TRACKS = 1

    def __init__(self, config, thermal_config, location_config):
        self.res_x = config.classify.res_x
        self.res_y = config.classify.res_y
        self.total_time = 0
        self.motion_detector = MotionDetector(
            self.res_x,
            self.res_y,
            thermal_config.motion,
            location_config,
            thermal_config.recorder,
            config.tracking.dynamic_thresh,
            CPTVRecorder(location_config, thermal_config),
        )

    def disconnected(self):
        self.motion_detector.force_stop()

    def process_frame(self, lepton_frame):
        start = time.time()
        self.motion_detector.process_frame(lepton_frame)
        end = time.time()
        self.total_time += end - start
        print("Time for a frame is {}us".format(1000 * 1000 * (end - start)))
