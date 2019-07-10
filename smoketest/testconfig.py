import attr
from pathlib import Path
import json
import yaml
CONFIG_FILE = "testconfig.json"


@attr.s
class TestConfig:
    classify_cmd = attr.ib(
        default="python3 /home/zaza/Cacophony/classifier-pipeline/classify.py --processor-folder {folder} {source}"
    )
    classify_dir = attr.ib(default="./classify")
    classify_config = attr.ib(
        default="/home/zaza/Cacophony/classifier-pipeline/classifier.yaml"
    )
    test_json = attr.ib(default="./classify-test.json")

    def load_config(self):
        if not Path(CONFIG_FILE).is_file():
            print(
                "No config file '{}'.  Running with default config.".format(CONFIG_FILE)
            )
            return self

        print("Attempting to load config from file '{}'...".format(CONFIG_FILE))
        with open(CONFIG_FILE) as f:
            return TestConfig(**json.load(f))

    def load_classifier_config(self):
        with open(self.classify_config) as f:
            return yaml.safe_load(f)
