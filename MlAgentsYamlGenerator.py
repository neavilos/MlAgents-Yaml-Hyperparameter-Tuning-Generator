import argparse
import json
import os
import random
import time

import yaml

from FieldSpecs import InputIntSpec, InputBehaviorSpec, InputBoolSpec, InputStrSpec, InputFloatSpec
from YamlBehaviorGenerator import MlAgentsYamlBehaviorGenerator

_VERSION = 1.1
_SLEEP_SECONDS_WHEN_FILE_EXISTS = 2
_DEFAULT_BEHAVIOR_CONFIG_NAME = "default"


class MlAgentsYamlGenerator(object):
    __yaml_name = "behaviors"

    def __create_agent_name(self, behavior_name, config, valid_behavior_name_spec_names):
        if len(valid_behavior_name_spec_names) <= 0:
            return behavior_name

        unique_model_name = ','.join(
            "{0}:{1}".format(x[0], x[1]) for x in config if x[0] in valid_behavior_name_spec_names)
        if self.__hash_name:
            return "{0}_{1}".format(behavior_name, abs(hash(unique_model_name)))
        return "{0}_({1})".format(behavior_name, unique_model_name)

    @staticmethod
    def __product(sampled_states, depth=0):
        if len(sampled_states) <= depth + 1:
            for element in sampled_states[depth]:
                yield [element]
            return
        for right_element in MlAgentsYamlGenerator.__product(sampled_states, depth + 1):
            for left_element in sampled_states[depth]:
                yield [left_element] + right_element

    def __generate_yaml(self):
        behaviors = {}

        if not self.__behavior_specs:
            return {
                MlAgentsYamlGenerator.__yaml_name: {
                    _DEFAULT_BEHAVIOR_CONFIG_NAME: MlAgentsYamlBehaviorGenerator(
                        _DEFAULT_BEHAVIOR_CONFIG_NAME).generate_behavior()
                }
            }

        for behavior_spec in self.__behavior_specs:
            valid_behavior_name_spec_names = [x.spec_name for x in behavior_spec.field_configs if x.num_states() > 1]
            for chosen_config in MlAgentsYamlGenerator.__product(
                    [config.yield_named_states() for config in behavior_spec.field_configs]):
                agent_name = self.__create_agent_name(behavior_spec.behavior_name, chosen_config,
                                                      valid_behavior_name_spec_names)
                yaml_generator = MlAgentsYamlBehaviorGenerator(agent_name, {key: val for (key, val) in chosen_config})
                behaviors[agent_name] = yaml_generator.generate_behavior()
        return {MlAgentsYamlGenerator.__yaml_name: behaviors}

    def __read_behavior_specs(self, input_behavior_spec_path):
        with open(input_behavior_spec_path, 'r') as behavior_spec_file:
            spec_input = behavior_spec_file.read()
            behavior_spec_json = json.loads(spec_input)

        for behavior_spec_name in behavior_spec_json:
            field_specs = []
            for spec in behavior_spec_json[behavior_spec_name]:
                match spec["type"]:
                    case "int":
                        field_specs.append(InputIntSpec.create_spec(spec))
                        continue
                    case "float":
                        field_specs.append(InputFloatSpec.create_spec(spec))
                        continue
                    case "str":
                        field_specs.append(InputStrSpec.create_spec(spec))
                        continue
                    case "bool":
                        field_specs.append(InputBoolSpec.create_spec(spec))
                        continue
                raise Exception("Invalid type {0} given for spec {1}!".format(spec["type"], spec["name"]))
            self.__behavior_specs.append(InputBehaviorSpec(behavior_spec_name, field_specs))

    def __init__(self, input_behavior_spec_path, output_yaml_file_name="", hash_name=False, print_to_console=False):
        self.__hash_name = hash_name
        self.__behavior_specs = []

        if input_behavior_spec_path:
            self.__read_behavior_specs(input_behavior_spec_path)

        if print_to_console or not output_yaml_file_name:
            print(yaml.dump(self.__generate_yaml()))
        else:
            with open(output_yaml_file_name, "w") as output_yaml_file:
                yaml.dump(self.__generate_yaml(), output_yaml_file)


def validate_or_create_output_path(output_yaml_path):
    if os.path.exists(output_yaml_path):
        print("File at {0} already exists! Printing to console.".format(output_yaml_path))
        time.sleep(_SLEEP_SECONDS_WHEN_FILE_EXISTS)
        return True
    return False


def create_and_parse_arguments():
    parser = argparse.ArgumentParser(
        prog="MlAgentsYamlGenerator",
        usage="%(prog)s --input_json --output_yaml [options]"
    )

    parser.add_argument("--input_json",
                        default="",
                        help="Path to behavior config json. Program will create a default yaml if this flag is unset.")
    parser.add_argument("--output_yaml",
                        default="",
                        help="Output path to write yaml file. Program will print yaml to console if this flag is unset.")
    parser.add_argument("--seed",
                        required=False,
                        help="Set random seed to make consistent picking values from ranges.")
    parser.add_argument("--hash_name",
                        required=False,
                        action="store_true",
                        help="Whether to hash the behavior parameters name or not. Set this flag to use hashed names.")
    parser.add_argument("-v", "--version", action="version", version="%(prog)s {0}".format(_VERSION))
    return parser.parse_args()


if __name__ == "__main__":
    args = create_and_parse_arguments()

    print_to_console = validate_or_create_output_path(args.output_yaml)

    if args.seed:
        random.seed(args.seed)
    MlAgentsYamlGenerator(args.input_json, args.output_yaml, hash_name=args.hash_name,
                          print_to_console=print_to_console)
