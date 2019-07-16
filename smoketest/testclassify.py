import json
import logging
import subprocess
from pathlib import Path

import testconfig


def convert_to_dict(obj):
    return obj.__dict__


class Match:
    def __init__(
        self,
        expected,
        got,
        start_error,
        end_error,
        length_error,
        expected_animal,
        got_animal,
        score,
        improvement,
    ):
        self.start_error = start_error
        self.end_error = end_error
        self.length_error = length_error
        self.expected_animal = expected_animal
        self.got_animal = got_animal
        self.expected = expected
        self.got = got
        self.match = got_animal == expected_animal
        self.score = score
        self.improvement = improvement

    @classmethod
    def new_match(cls, expected, got):
        expected_length = expected["end_s"] - expected["start_s"]
        got_length = got["end_s"] - got["start_s"]
        start_error_s = expected["start_s"] - got["start_s"]
        end_error_s = expected["end_s"] - got["end_s"]

        opt_start_error_s = expected["opt_start"] - got["start_s"]
        opt_end_error_s = expected["opt_end"] - got["end_s"]

        score = round(opt_start_error_s + opt_end_error_s, 2)
        expected_score = Match.calc_score(expected)

        return cls(
            expected=expected,
            got=got,
            start_error=start_error_s,
            end_error=end_error_s,
            length_error=expected_length - got_length,
            expected_animal=expected["tag"],
            got_animal=got["tag"],
            score=score,
            improvement=score - expected_score,
        )

    @staticmethod
    def calc_score(track):
        score = track["opt_start"] - track["start_s"]
        score += track["opt_end"] - track["end_s"]
        return round(score, 2)


class TestClassify:
    def write_results(self, config, results):
        classify_config = config.load_classifier_config()
        result = {"Results": results, "Config": classify_config}
        with open("smoketest-results.json", "w") as outfile:
            json.dump(result, outfile, indent=2, default=convert_to_dict)

    def TestSet(self):
        config = testconfig.TestConfig().load_config()
        tests = self.load_test(config)
        classify_results = []
        output_results = []
        for test in tests:
            result = self.run_classify_on_test(test, config)
            classify_results.append(result)
            output_results.append(self.compare_output(test, result))
        self.write_results(config, output_results)
        self.print_summary(output_results)

    def print_summary(self, results):
        print("===== SUMMARY =====")
        for result in results:
            improved = len(
                [res for res in result.get("matches", []) if res.improvement >= 0]
            )
            worsened = len(
                [res for res in result.get("matches", []) if res.improvement < 0]
            )
            print(
                "{} {}-matches {}-mismatches {}-unmatched {}-improved  {}-matches worsened".format(
                    result["filename"],
                    len(result.get("matches", [])),
                    len(result.get("mismatches", [])),
                    len(result.get("unmatched", [])),
                    improved,
                    worsened,
                )
            )

    def compare_output(self, expected, result):
        file = Path(expected["filename"])
        result = easy_result(result)
        got_tracks = result["Tracks"]
        expected_tracks = sorted(expected["Tracks"], key=lambda x: x["start_s"])
        expected_tracks = [
            track for track in expected_tracks if track.get("expected", True)
        ]

        for i, track in enumerate(got_tracks):
            if len(expected_tracks) > i:
                expected = expected_tracks[i]
                match = Match.new_match(expected, track)
                if match.match:
                    result.setdefault("matches", []).append(match)
                else:
                    result.setdefault("mismatches", []).append(match)
                    print(
                        "mismatches {} start {} end {} \nexpected {} start {} end {}".format(
                            track["tag"],
                            track["start_s"],
                            track["end_s"],
                            expected["tag"],
                            expected["start_s"],
                            expected["end_s"],
                        )
                    )
            else:
                result.setdefault("unmatched", []).append(track)
                print(
                    "Unmatched track tag {} start {} end {}".format(
                        track["tag"], track["start_s"], track["end_s"]
                    )
                )
        result["expected"] = expected_tracks
        result["got"] = got_tracks
        del result["Tracks"]
        return result

    def run_classify_on_test(self, recording, conf):
        rec_file = Path(recording["filename"])
        print("processing %s", rec_file)

        working_dir = rec_file.parent
        command = conf.classify_cmd.format(
            folder=str(working_dir), source=rec_file.resolve()
        )
        output = subprocess.check_output(
            command,
            cwd=conf.classify_dir,
            shell=True,
            encoding="ascii",
            stderr=subprocess.DEVNULL,
        )
        try:
            classify_info = json.loads(output)
        except json.decoder.JSONDecodeError as err:
            raise ValueError(
                "failed to JSON decode classifier output:\n{}".format(output)
            ) from err
        classify_info["id"] = recording["id"]
        return classify_info

    def load_test(self, confifg):
        with open(confifg.test_json) as file:
            meta = json.load(file)
        return meta["Recordings"]


def easy_results(results):
    recordings = []
    for result in results:
        recordings.append(easy_result(result))


def easy_result(result, ai_tag=True, tag_precedence=None):
    recording = {}
    meta = result["cptv_meta"]
    recording["id"] = meta["id"]
    recording["filename"] = result["source"]
    recording["DeviceId"] = meta["DeviceId"]
    recording["GroupId"] = meta["GroupId"]
    recording["Group"] = meta["Group"]
    recording["Tracks"] = []
    for track in meta["Tracks"]:
        new_track = {}
        new_track["opt_start"] = track["data"]["start_s"]
        new_track["opt_end"] = track["data"]["end_s"]
        new_track["start_s"] = track["data"]["start_s"]
        new_track["end_s"] = track["data"]["end_s"]
        if ai_tag:
            new_track["tag"] = track["data"].get("tag", "")
        else:
            tag = get_best_tag(track, tag_precedence=tag_precedence)
            if tag:
                new_track["tag"] = tag["what"]
        recording["Tracks"].append(new_track)

    recording["Tracks"] = sorted(recording["Tracks"], key=lambda x: x["start_s"])
    return recording


def get_best_tag(track_meta, tag_precedence, min_confidence=-1):
    """ returns highest precidence non AI tag from the metadata """

    track_tags = track_meta.get("TrackTags", [])
    track_tags = [tag for tag in track_tags if tag.get("confidence") > min_confidence]

    if not track_tags:
        return None

    tag = None
    default_prec = tag_precedence.get("default", 100)
    best = None
    for track_tag in track_tags:
        ranking = tag_ranking(track_tag, default_prec, tag_precedence)

        # if 2 track_tags have same confidence ignore both
        if ranking == best:
            tag = None
        elif best is None or ranking < best:
            best = ranking
            tag = track_tag
    return tag


def tag_ranking(track_tag, default_prec, tag_precedence):
    """ returns a ranking of tags based of what they are and confidence """

    what = track_tag.get("what")
    confidence = 1 - track_tag.get("confidence", 0)
    automatic = track_tag["automatic"]
    prec = tag_precedence.get(what, default_prec) * 10
    if automatic:
        prec = prec + 5
    return prec + confidence


def main():
    test = TestClassify()
    test.TestSet()


if __name__ == "__main__":
    main()
