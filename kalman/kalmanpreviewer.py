from ml_tools.previewer import Previewer
import cv2
import numpy as np


class KalmanPreviewer(Previewer):
    def __init__(self, config, preview_type):
        self.config = config
        self.colourmap = self._load_colourmap()

        # make sure all the required files are there
        self.track_descs = {}
        self.font
        self.font_title
        self.preview_type = preview_type
        self.frame_scale = 1
        self.debug = config.debug
        # super(Previewer, self).__init__(config, preview_type)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)

        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )

        self.kalman.processNoiseCov = np.eye(4, 4, dtype=np.float32) * 0.03

    def reset_kalman(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)

        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )

        self.kalman.processNoiseCov = np.eye(4, 4, dtype=np.float32) * 0.03

    def add_tracks(
        self,
        draw,
        tracks,
        frame_number,
        track_predictions=None,
        screen_bounds=None,
        colours=None,
        tracks_text=None,
        v_offset=0,
    ):
        # look for any tracks that occur on this frame
        for index, track in enumerate(tracks):
            frame_offset = frame_number - track.start_frame
            if frame_offset >= 0 and frame_offset < len(track.bounds_history) - 1:
                # draw frame
                rect = track.bounds_history[frame_offset]
                draw.rectangle(
                    self.rect_points(rect),
                    outline=self.TRACK_COLOURS[index % len(self.TRACK_COLOURS)],
                )

                pts = np.array(
                    [np.float32(rect.mid_x * 4), np.float32(rect.mid_y * 4)], np.float32
                )
                print("correct with pt", pts)
                self.kalman.correct(pts)

                prediction = self.kalman.predict()
                draw.arc(
                    (
                        prediction[0] - 4,
                        prediction[1] - 4,
                        prediction[0] + 4,
                        prediction[1] + 4,
                    ),
                    0,
                    360,
                )
                # # draw centre
                # xx = rect.mid_x * 4.0
                # yy = rect.mid_y * 4.0
                # center = track.bounds_history[frame_offset].mid_x
                # draw.arc((xx - 4, yy - 4, xx + 4, yy + 4), 0, 360)
