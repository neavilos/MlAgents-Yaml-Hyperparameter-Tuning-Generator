import re


def _make_schedule_field_spec(field_name):
    return FieldSpec(field_name, field_name, "linear", ["linear", "constant"])


class ContainerFieldSpec(object):
    def __init__(self, field_name, subfields=[], yaml_name=""):
        self.field_name = field_name
        self.subfields = subfields
        self.yaml_name = yaml_name if yaml_name else field_name


class FieldSpec(object):
    def __init__(self, field_name, yaml_name="", default_value=None, value_range=[], always_include_in_config=True):
        self.field_name = field_name
        self.always_include_in_config = always_include_in_config
        self.default_value = default_value
        self.__value_range = value_range if value_range and isinstance(default_value, str) else []
        self.yaml_name = yaml_name if yaml_name else field_name

    def is_valid_value(self, new_value):
        return new_value in self.__value_range if self.__value_range else True


class _DefaultAgentsBehaviorFields(object):
    # Common Trainer Fields
    trainer_type = FieldSpec("trainer_type", yaml_name="trainer_type", default_value="ppo",
                             value_range=["ppo", "sac", "poca"])
    summary_freq = FieldSpec("summary_freq", yaml_name="summary_freq", default_value=5e4, )
    time_horizon = FieldSpec("time_horizon", yaml_name="time_horizon", default_value=64)
    max_steps = FieldSpec("max_steps", yaml_name="max_steps", default_value=500000)
    keep_checkpoints = FieldSpec("keep_checkpoints", yaml_name="keep_checkpoints", default_value=5)
    even_checkpoints = FieldSpec("even_checkpoints", yaml_name="even_checkpoints", default_value=False,
                                 always_include_in_config=False)
    checkpoint_interval = FieldSpec("checkpoint_interval", yaml_name="checkpoint_interval", default_value=50000)
    init_path = FieldSpec("init_path", yaml_name="init_path", default_value=None, always_include_in_config=False)
    threaded = FieldSpec("threaded", yaml_name="threaded", default_value=False)

    # Common Hyperparameter Fields
    batch_size = FieldSpec("batch_size", yaml_name="batch_size", default_value=512)
    buffer_size = FieldSpec("buffer_size", yaml_name="buffer_size", default_value=10240)
    learning_rate = FieldSpec("learning_rate", yaml_name="learning_rate", default_value=3e-4)
    learning_rate_schedule = _make_schedule_field_spec("learning_rate_schedule")

    # Common Network Setting Fields
    hidden_units = FieldSpec("hidden_units", yaml_name="hidden_units", default_value=128)
    num_layers = FieldSpec("num_layers", yaml_name="num_layers", default_value=2)
    normalize = FieldSpec("normalize", yaml_name="normalize", default_value=False)
    vis_encode_type = FieldSpec("vis_encode_type", yaml_name="vis_encode_type", default_value="simple")
    conditioning_type = FieldSpec("conditioning_type", yaml_name="conditioning_type", default_value="hyper",
                                  value_range=["hyper", "none"])

    # PPO-specific Fields
    beta = FieldSpec("beta", yaml_name="beta", default_value=5e-3)
    epsilon = FieldSpec("epsilon", yaml_name="epsilon", default_value=0.2)
    beta_schedule = _make_schedule_field_spec("beta_schedule")
    epsilon_schedule = _make_schedule_field_spec("epsilon_schedule")
    lambd = FieldSpec("lambd", yaml_name="lambd", default_value=0.95)
    num_epoch = FieldSpec("num_epoch", yaml_name="num_epoch", default_value=3)
    shared_critic = FieldSpec("shared_critic", yaml_name="shared_critic", default_value=False)

    # SAC-specific fields
    buffer_init_steps = FieldSpec("buffer_init_steps", yaml_name="buffer_init_steps", default_value=0)
    init_ent_coef = FieldSpec("init_ent_coef", yaml_name="init_ent_coef", default_value=1.0)
    save_replay_buffer = FieldSpec("save_replay_buffer", yaml_name="save_replay_buffer", default_value=False)
    tau = FieldSpec("tau", yaml_name="tau", default_value=5e-3)
    steps_per_update = FieldSpec("steps_per_update", yaml_name="steps_per_update", default_value=1)
    reward_signal_num_update = FieldSpec("reward_signal_num_update", yaml_name="reward_signal_num_update",
                                         default_value=1)

    # Extrinsic Reward Fields
    extrinsic_strength = FieldSpec("extrinsic_strength", yaml_name="strength", default_value=1.0)
    extrinsic_gamma = FieldSpec("extrinsic_gamma", yaml_name="gamma", default_value=0.99)

    # Curiosity Intrinsic Reward Fields
    curiosity_strength = FieldSpec("curiosity_strength", yaml_name="strength", default_value=1.0,
                                   always_include_in_config=False)
    curiosity_gamma = FieldSpec("curiosity_gamma", yaml_name="gamma", default_value=0.99,
                                always_include_in_config=False)
    curiosity_learning_rate = FieldSpec("curiosity_learning_rate", yaml_name="learning_rate", default_value=3e-4,
                                        always_include_in_config=False)
    curiosity_hidden_units = FieldSpec("curiosity_hidden_units", yaml_name="hidden_units", default_value=128,
                                       always_include_in_config=False)
    curiosity_num_layers = FieldSpec("curiosity_num_layers", yaml_name="num_layers", default_value=2,
                                     always_include_in_config=False)
    curiosity_normalize = FieldSpec("curiosity_normalize", yaml_name="normalize", default_value=False,
                                    always_include_in_config=False)
    curiosity_vis_encode_type = FieldSpec("curiosity_vis_encode_type", yaml_name="vis_encode_type",
                                          default_value="simple", always_include_in_config=False)
    curiosity_conditioning_type = FieldSpec("curiosity_conditioning_type", yaml_name="conditioning_type",
                                            default_value="hyper", value_range=["hyper", "none"],
                                            always_include_in_config=False)

    # GAIL Intrinsic Reward Fields
    gail_strength = FieldSpec("gail_strength", yaml_name="strength", default_value=1.0, always_include_in_config=False)
    gail_gamma = FieldSpec("gail_gamma", yaml_name="gamma", default_value=0.99, always_include_in_config=False)
    gail_learning_rate = FieldSpec("gail_learning_rate", yaml_name="learning_rate", default_value=3e-4,
                                   always_include_in_config=False)
    gail_hidden_units = FieldSpec("gail_hidden_units", yaml_name="hidden_units", default_value=128,
                                  always_include_in_config=False)
    gail_num_layers = FieldSpec("gail_num_layers", yaml_name="num_layers", default_value=2,
                                always_include_in_config=False)
    gail_normalize = FieldSpec("gail_normalize", yaml_name="normalize", default_value=False,
                               always_include_in_config=False)
    gail_vis_encode_type = FieldSpec("gail_vis_encode_type", yaml_name="vis_encode_type", default_value="simple",
                                     always_include_in_config=False)
    gail_conditioning_type = FieldSpec("gail_conditioning_type", yaml_name="conditioning_type", default_value="hyper",
                                       value_range=["hyper", "none"], always_include_in_config=False)
    gail_use_actions = FieldSpec("gail_use_actions", yaml_name="use_actions", default_value=False,
                                 always_include_in_config=False)
    gail_use_vail = FieldSpec("gail_use_vail", yaml_name="use_vail", default_value=False,
                              always_include_in_config=False)
    gail_demo_path = FieldSpec("gail_demo_path", yaml_name="demo_path", default_value="None",
                               always_include_in_config=False)

    # RND Intrinsic Reward Fields
    rnd_strength = FieldSpec("rnd_strength", yaml_name="strength", default_value=1.0, always_include_in_config=False)
    rnd_gamma = FieldSpec("rnd_gamma", yaml_name="gamma", default_value=0.99, always_include_in_config=False)
    rnd_learning_rate = FieldSpec("rnd_learning_rate", yaml_name="learning_rate", default_value=3e-4,
                                  always_include_in_config=False)
    rnd_hidden_units = FieldSpec("rnd_hidden_units", yaml_name="hidden_units", default_value=128,
                                 always_include_in_config=False)
    rnd_num_layers = FieldSpec("rnd_num_layers", yaml_name="num_layers", default_value=2,
                               always_include_in_config=False)
    rnd_normalize = FieldSpec("rnd_normalize", yaml_name="normalize", default_value=False,
                              always_include_in_config=False)
    rnd_vis_encode_type = FieldSpec("rnd_vis_encode_type", yaml_name="vis_encode_type", default_value="simple",
                                    always_include_in_config=False)
    rnd_conditioning_type = FieldSpec("rnd_conditioning_type", yaml_name="conditioning_type", default_value="hyper",
                                      value_range=["hyper", "none"], always_include_in_config=False)

    # Behavioral Cloning Fields
    behavior_cloning_demo_path = FieldSpec("demo_path", yaml_name="demo_path", default_value="None",
                                           always_include_in_config=False)
    behavior_cloning_strength = FieldSpec("behavior_cloning_strength", yaml_name="strength", default_value=1.0,
                                          always_include_in_config=False)
    behavior_cloning_steps = FieldSpec("behavior_cloning_steps", yaml_name="steps", default_value=0,
                                       always_include_in_config=False)
    behavior_cloning_batch_size = FieldSpec("behavior_cloning_batch_size", yaml_name="batch_size", default_value=512,
                                            always_include_in_config=False)
    behavior_cloning_num_epoch = FieldSpec("behavior_cloning_num_epoch", yaml_name="num_epoch", default_value=3,
                                           always_include_in_config=False)
    behavior_cloning_samples_per_update = FieldSpec("behavior_cloning_samples_per_update",
                                                    yaml_name="samples_per_update", default_value=0,
                                                    always_include_in_config=False)

    # Memory-enhanced RNN Fields
    memory_size = FieldSpec("memory_size", yaml_name="memory_size", default_value=128, always_include_in_config=False)
    sequence_length = FieldSpec("sequence_length", yaml_name="sequence_length", default_value=64,
                                always_include_in_config=False)

    # Self-Play Fields
    save_steps = FieldSpec("save_steps", yaml_name="save_steps", default_value=20000, always_include_in_config=False)
    team_change = FieldSpec("team_change", yaml_name="team_change", default_value=100000,
                            always_include_in_config=False)
    swap_steps = FieldSpec("swap_steps", yaml_name="swap_steps", default_value=10000, always_include_in_config=False)
    play_against_latest_model_ratio = FieldSpec("play_against_latest_model_ratio",
                                                yaml_name="play_against_latest_model_ratio", default_value=0.5,
                                                always_include_in_config=False)
    window = FieldSpec("window", yaml_name="window", default_value=10, always_include_in_config=False)
    initial_elo = FieldSpec("initial_elo", yaml_name="initial_elo", default_value=1200.0,
                            always_include_in_config=False)


class _DefaultAgentsBehaviorMemoryRNNFields(object):
    memory_rnn_fields = ContainerFieldSpec("memory", [
        _DefaultAgentsBehaviorFields.memory_size.field_name,
        _DefaultAgentsBehaviorFields.sequence_length.field_name
    ])


class _DefaultAgentsBehaviorContainerFields(object):
    sac_hyperparameters = ContainerFieldSpec("sac_hyperparameters", [
        _DefaultAgentsBehaviorFields.buffer_init_steps.field_name,
        _DefaultAgentsBehaviorFields.init_ent_coef.field_name,
        _DefaultAgentsBehaviorFields.save_replay_buffer.field_name,
        _DefaultAgentsBehaviorFields.tau.field_name,
        _DefaultAgentsBehaviorFields.steps_per_update.field_name,
        _DefaultAgentsBehaviorFields.reward_signal_num_update.field_name, ])
    hyperparameters = ContainerFieldSpec("hyperparameters", [
        _DefaultAgentsBehaviorFields.batch_size.field_name,
        _DefaultAgentsBehaviorFields.buffer_size.field_name,
        _DefaultAgentsBehaviorFields.learning_rate.field_name,
        _DefaultAgentsBehaviorFields.beta.field_name,
        _DefaultAgentsBehaviorFields.epsilon.field_name,
        _DefaultAgentsBehaviorFields.lambd.field_name,
        _DefaultAgentsBehaviorFields.num_epoch.field_name,
        _DefaultAgentsBehaviorFields.learning_rate_schedule.field_name,
        _DefaultAgentsBehaviorFields.beta_schedule.field_name,
        _DefaultAgentsBehaviorFields.epsilon_schedule.field_name, ])
    network_settings = ContainerFieldSpec("network_settings", [
        _DefaultAgentsBehaviorFields.hidden_units.field_name,
        _DefaultAgentsBehaviorFields.num_layers.field_name,
        _DefaultAgentsBehaviorFields.normalize.field_name,
        _DefaultAgentsBehaviorFields.vis_encode_type.field_name,
        _DefaultAgentsBehaviorFields.conditioning_type.field_name,
        _DefaultAgentsBehaviorMemoryRNNFields.memory_rnn_fields.field_name])
    curiosity_network_settings = ContainerFieldSpec("curiosity_network_settings", [
        _DefaultAgentsBehaviorFields.curiosity_hidden_units.field_name,
        _DefaultAgentsBehaviorFields.curiosity_num_layers.field_name,
        _DefaultAgentsBehaviorFields.curiosity_normalize.field_name,
        _DefaultAgentsBehaviorFields.curiosity_vis_encode_type.field_name,
        _DefaultAgentsBehaviorFields.curiosity_conditioning_type.field_name,
        _DefaultAgentsBehaviorMemoryRNNFields.memory_rnn_fields.field_name], "network_settings")
    gail_network_settings = ContainerFieldSpec("gail_network_settings", [
        _DefaultAgentsBehaviorFields.gail_hidden_units.field_name,
        _DefaultAgentsBehaviorFields.gail_num_layers.field_name,
        _DefaultAgentsBehaviorFields.gail_normalize.field_name,
        _DefaultAgentsBehaviorFields.gail_vis_encode_type.field_name,
        _DefaultAgentsBehaviorFields.gail_conditioning_type.field_name,
        _DefaultAgentsBehaviorMemoryRNNFields.memory_rnn_fields.field_name], "network_settings")
    rnd_network_settings = ContainerFieldSpec("rnd_network_settings", [
        _DefaultAgentsBehaviorFields.rnd_hidden_units.field_name,
        _DefaultAgentsBehaviorFields.rnd_num_layers.field_name,
        _DefaultAgentsBehaviorFields.rnd_normalize.field_name,
        _DefaultAgentsBehaviorFields.rnd_vis_encode_type.field_name,
        _DefaultAgentsBehaviorFields.rnd_conditioning_type.field_name,
        _DefaultAgentsBehaviorMemoryRNNFields.memory_rnn_fields.field_name], "network_settings")
    behavior_cloning = ContainerFieldSpec("behavior_cloning", [
        _DefaultAgentsBehaviorFields.behavior_cloning_demo_path.field_name,
        _DefaultAgentsBehaviorFields.behavior_cloning_strength.field_name,
        _DefaultAgentsBehaviorFields.behavior_cloning_steps.field_name,
        _DefaultAgentsBehaviorFields.behavior_cloning_batch_size.field_name,
        _DefaultAgentsBehaviorFields.behavior_cloning_num_epoch.field_name,
        _DefaultAgentsBehaviorFields.behavior_cloning_samples_per_update.field_name,
    ])
    self_play = ContainerFieldSpec("self_play", [
        _DefaultAgentsBehaviorFields.save_steps.field_name,
        _DefaultAgentsBehaviorFields.team_change.field_name,
        _DefaultAgentsBehaviorFields.swap_steps.field_name,
        _DefaultAgentsBehaviorFields.play_against_latest_model_ratio.field_name,
        _DefaultAgentsBehaviorFields.window.field_name,
        _DefaultAgentsBehaviorFields.initial_elo.field_name,
    ])


class _DefaultAgentsBehaviorRewardSpecificContainerFields(object):
    extrinsic = ContainerFieldSpec("extrinsic", [
        _DefaultAgentsBehaviorFields.extrinsic_strength.field_name,
        _DefaultAgentsBehaviorFields.extrinsic_gamma.field_name])
    curiosity = ContainerFieldSpec("curiosity", [
        _DefaultAgentsBehaviorFields.curiosity_strength.field_name,
        _DefaultAgentsBehaviorFields.curiosity_gamma.field_name,
        _DefaultAgentsBehaviorContainerFields.curiosity_network_settings.field_name,
        _DefaultAgentsBehaviorFields.curiosity_learning_rate.field_name])
    gail = ContainerFieldSpec("gail", [
        _DefaultAgentsBehaviorFields.gail_strength.field_name,
        _DefaultAgentsBehaviorFields.gail_gamma.field_name,
        _DefaultAgentsBehaviorFields.gail_demo_path.field_name,
        _DefaultAgentsBehaviorContainerFields.gail_network_settings.field_name,
        _DefaultAgentsBehaviorFields.gail_learning_rate.field_name,
        _DefaultAgentsBehaviorFields.gail_use_actions.field_name,
        _DefaultAgentsBehaviorFields.gail_use_vail.field_name])
    rnd = ContainerFieldSpec("rnd", [
        _DefaultAgentsBehaviorFields.rnd_strength.field_name,
        _DefaultAgentsBehaviorFields.rnd_gamma.field_name,
        _DefaultAgentsBehaviorContainerFields.rnd_network_settings.field_name])


class _DefaultAgentsBehaviorRewardContainerField(object):
    reward_signals = ContainerFieldSpec("reward_signals", [
        _DefaultAgentsBehaviorRewardSpecificContainerFields.extrinsic.field_name,
        _DefaultAgentsBehaviorRewardSpecificContainerFields.curiosity.field_name,
        _DefaultAgentsBehaviorRewardSpecificContainerFields.gail.field_name,
        _DefaultAgentsBehaviorRewardSpecificContainerFields.rnd.field_name])


class _DefaultAgentsBehavior(object):
    behavior = ContainerFieldSpec("behavior", [
        _DefaultAgentsBehaviorFields.trainer_type.field_name,
        _DefaultAgentsBehaviorFields.time_horizon.field_name,
        _DefaultAgentsBehaviorFields.max_steps.field_name,
        _DefaultAgentsBehaviorFields.keep_checkpoints.field_name,
        _DefaultAgentsBehaviorFields.checkpoint_interval.field_name,
        _DefaultAgentsBehaviorFields.threaded.field_name,
        _DefaultAgentsBehaviorFields.init_path.field_name,
        _DefaultAgentsBehaviorContainerFields.hyperparameters.field_name,
        _DefaultAgentsBehaviorContainerFields.network_settings.field_name,
        _DefaultAgentsBehaviorRewardContainerField.reward_signals.field_name,
        _DefaultAgentsBehaviorContainerFields.behavior_cloning.field_name,
        _DefaultAgentsBehaviorContainerFields.self_play.field_name,
    ])


def _get_all_field_spec_names(cls):
    field_specs = vars(cls)
    return {getattr(cls, key).field_name: field_specs[key] for key in field_specs if not re.match(r"__.*__", key)}


AgentsBehaviorFieldTemplate = _get_all_field_spec_names(_DefaultAgentsBehaviorFields)
AgentsBehaviorRewardContainerFieldTemplate = _get_all_field_spec_names(_DefaultAgentsBehaviorRewardContainerField)
AgentsBehaviorRewardSpecificContainerFieldTemplate = _get_all_field_spec_names(
    _DefaultAgentsBehaviorRewardSpecificContainerFields)
AgentsBehaviorContainerFieldsTemplate = _get_all_field_spec_names(_DefaultAgentsBehaviorContainerFields)
AgentsBehaviorMemoryContainerFieldsTemplate = _get_all_field_spec_names(_DefaultAgentsBehaviorMemoryRNNFields)

AgentBehaviorFieldTemplates = [
    AgentsBehaviorFieldTemplate,
    AgentsBehaviorRewardContainerFieldTemplate,
    AgentsBehaviorRewardSpecificContainerFieldTemplate,
    AgentsBehaviorContainerFieldsTemplate,
    AgentsBehaviorMemoryContainerFieldsTemplate,
]
