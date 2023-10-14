import tensorflow as tf
import sonnet as snt
import numpy as np


class ObsRewardEncoder(snt.Module):
    def __init__(self, n_agents, n_actions, obs_dim, mixing_embed_dim=32,state_latent_dim=64, action_latent_dim=64):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_dim = obs_dim


        self.mixing_embed_dim = mixing_embed_dim
        self.action_latent_dim = action_latent_dim
        self.state_latent_dim = state_latent_dim

        self.obs_encoder_avg = snt.Sequential(
            [
                snt.Linear(self.state_latent_dim * 2),
                tf.nn.relu,
                snt.Linear(self.state_latent_dim)
            ]
        )
        self.obs_decoder_avg = snt.Sequential(
            [
                snt.Linear(self.state_latent_dim),
                tf.nn.relu,
                snt.Linear(self.obs_dim)
            ]
        )

        self.action_encoder = snt.Sequential(
            [
                snt.Linear(self.state_latent_dim * 2),
                tf.nn.relu,
                snt.Linear(self.action_latent_dim)
            ]
        )

        self.reward_decoder_avg = snt.Sequential(
            [
                snt.Linear(self.state_latent_dim),
                tf.nn.relu,
                snt.Linear(1)
            ]
        )
    
    def predict(self, obs, actions):
        other_actions = self.other_actions(actions)
        obs_reshaped = tf.reshape(obs, [-1, self.obs_dim])
        inputs = tf.concat([obs_reshaped, other_actions], axis=-1)

        # average
        obs_latent_avg = self.obs_encoder_avg(inputs)
        actions = tf.reshape(actions, [-1, self.n_actions])
        action_latent_avg = self.action_encoder(actions)

        pred_avg_input = tf.concat([obs_latent_avg, action_latent_avg], axis=-1)
        no_pred_avg = self.obs_decoder_avg(pred_avg_input)
        r_pred_avg = self.reward_decoder_avg(pred_avg_input)

        no_pred_avg = tf.reshape(no_pred_avg, [-1, self.n_agents, self.obs_dim])
        r_pred_avg = tf.reshape(r_pred_avg, [-1, self.n_agents, 1])

        return no_pred_avg, r_pred_avg

    def other_actions(self, actions):
        # actions: [bs, n_agents, n_actions]
        assert actions.shape[1] == self.n_agents

        other_actions = []
        for i in range(self.n_agents):
            _other_actions  = []
            for j in range(self.n_agents):
                if i != j:
                    _other_actions.append(actions[:, j])
            _other_actions = tf.concat(_other_actions, axis=-1)
            other_actions.append(_other_actions)

        other_actions = tf.reshape(tf.transpose(tf.stack(other_actions)), [-1, (self.n_agents - 1) * self.n_actions])

        return other_actions

    def __call__(self):
        actions = tf.constant(np.eye(self.n_actions), dtype=tf.float32)
        actions_latent_avg = self.action_encoder(actions)
        return actions_latent_avg







        