from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.offline import JsonReader
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class CustomLossModel(TFModelV2):
    """Custom model that adds an imitation loss on top of the policy loss."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        self.fcnet = FullyConnectedNetwork(
            self.obs_space,
            self.action_space,
            num_outputs,
            model_config,
            name="fcnet")
        self.register_variables(self.fcnet.variables())

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Delegate to our FCNet.
        return self.fcnet(input_dict, state, seq_lens)

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        # create a new input reader per worker
        reader = JsonReader(self.model_config["custom_options"]["input_files"])
        input_ops = reader.tf_input_ops(self.model_config["custom_options"].get("expert_size",1))

        # define a secondary loss by building a graph copy with weight sharing
        obs = restore_original_dimensions(
            tf.cast(input_ops["obs"], tf.float32), self.obs_space)
        logits, _ = self.forward({"obs": obs}, [], None)

        # You can also add self-supervised losses easily by referencing tensors
        # created during _build_layers_v2(). For example, an autoencoder-style
        # loss can be added as follows:
        # ae_loss = squared_diff(
        #     loss_inputs["obs"], Decoder(self.fcnet.last_layer))
        # print("FYI: You can also use these tensors: {}, ".format(loss_inputs))

        # compute the IL loss
        self.policy_loss = policy_loss
        (action_scores, model_logits, dist) = self.get_q_value_distributions(logits)
        model_logits= tf.squeeze(model_logits)
        action_dist = Categorical(model_logits, self.model_config)

        expert_logits = tf.cast(input_ops["actions"], tf.int32)
        expert_action = tf.math.argmax(expert_logits)
        expert_action_one_hot = tf.one_hot(expert_action,self.num_outputs)
        model_action = action_dist.deterministic_sample()
        model_action_one_hot = tf.one_hot(model_action,self.num_outputs)
        model_expert = model_action_one_hot * expert_action_one_hot
        imitation_loss = 0
        loss_type = self.model_config["custom_options"].get("loss","ce")
        if loss_type == "ce":
            imitation_loss = tf.reduce_mean(-action_dist.logp(expert_logits))
        elif loss_type == "kl":
            expert_dist = Categorical(tf.one_hot(expert_logits,\
                self.num_outputs), self.model_config)
            imitation_loss = tf.reduce_mean(-action_dist.kl(expert_dist))
        elif loss_type == "dqfd":
            max_value = float("-inf")
            Q_select = model_logits #  TODO: difference in action_scores,dist and logits
            for a in range(self.num_outputs):
                max_value = tf.maximum(Q_select[a] + 0.8 * tf.cast(model_expert[a],tf.float32),max_value)
            imitation_loss =tf.reduce_mean(1 * (max_value - Q_select[tf.cast(expert_action, tf.int32)]))

        self.imitation_loss = imitation_loss
        total_loss = self.model_config["custom_options"]["lambda1"]*policy_loss\
                     + self.model_config["custom_options"]["lambda2"]\
            * self.imitation_loss
        return total_loss

    def custom_stats(self):
        return {
            "policy_loss": self.policy_loss,
            "imitation_loss": self.imitation_loss,
        }
