from enum import Enum

import YamlFields


class MlAgentsYamlBehaviorGenerator(object):
    class NetworkSettingsTypes(Enum):
        GENERAL = 0
        CURIOSITY = 1
        GAIL = 2
        RND = 3

    def __init__(self, agent_name: str, changed_params=None):
        if changed_params is None:
            changed_params = {}
        self.__agent_name: str = agent_name
        self.__changed_params = changed_params

    def __lookup_name(self, field_name):
        for field_template in YamlFields.AgentBehaviorFieldTemplates:
            if field_name in field_template:
                return field_template[field_name].yaml_name
        raise Exception("Can't find field {0}.".format(field_name))

    def __maybe_provide_field_if_forced_or_changed(self, field_spec):
        if field_spec.field_name in self.__changed_params:
            changed_value = self.__changed_params[field_spec.field_name]
            if field_spec.is_valid_value(changed_value):
                return changed_value

        return field_spec.default_value if field_spec.always_include_in_config else None

    def __provide_trainer_specific_hparams(self, subfields):
        trainer_type = YamlFields._DefaultAgentsBehaviorFields.trainer_type.field_name
        if trainer_type in self.__changed_params and self.__changed_params[trainer_type] == "sac":
            return {
                self.__lookup_name(field_name): self.__lookup_value(field_name) for field_name in
                YamlFields.AgentsBehaviorContainerFieldsTemplate[
                    DefaultAgentsBehaviorContainerFields.sac_hyperparameters.field_name].subfields
            }
        return {
            self.__lookup_name(field_name): self.__lookup_value(field_name) for field_name in subfields
        }

    def __lookup_value(self, field_name):
        for field_template in YamlFields.AgentBehaviorFieldTemplates:
            if field_name not in field_template:
                continue
            if field_name in YamlFields.AgentsBehaviorFieldTemplate:
                return self.__maybe_provide_field_if_forced_or_changed(
                    YamlFields.AgentsBehaviorFieldTemplate[field_name])
            if field_name in YamlFields.AgentsBehaviorContainerFieldsTemplate:
                return self.__provide_trainer_specific_hparams(
                    YamlFields.AgentsBehaviorContainerFieldsTemplate[field_name].subfields) if \
                    field_name != YamlFields._DefaultAgentsBehaviorContainerFields.hyperparameters.field_name else {
                    self.__lookup_name(subfield_name): self.__lookup_value(subfield_name) for subfield_name in
                    YamlFields.AgentsBehaviorContainerFieldsTemplate[field_name].subfields
                }

            return {
                self.__lookup_name(subfield_name): self.__lookup_value(subfield_name) for subfield_name in
                field_template[field_name].subfields
            }
        raise Exception("Can't find field {0}.".format(field_name))

    def __remove_empty_fields_and_containers(self, config):
        for field_name in list(config):
            field = config[field_name]
            if isinstance(field, dict):
                self.__remove_empty_fields_and_containers(field)
            if field == {} or field is None:
                del (config[field_name])
        if config == {}:
            del (config)

    def generate_behavior(self) -> dict:
        behavior_config_template = YamlFields._DefaultAgentsBehavior.behavior

        behavior_config = {
            self.__lookup_name(field_name): self.__lookup_value(field_name) for field_name in
            behavior_config_template.subfields
        }

        self.__remove_empty_fields_and_containers(behavior_config)

        return behavior_config
