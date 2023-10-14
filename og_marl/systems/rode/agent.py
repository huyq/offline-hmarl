import copy
import tensorflow as tf
import trfl
import sonnet as snt
from og_marl.systems.rode.role_selector import DotSelector
from og_marl.systems.rode.roles import DotRole
from og_marl.systems.rode.action_encoder import ObsRewardEncoder


class RODEAgent:
    def __init__(self, args, n_agents, n_actions, role_interval, action_encoder, q_network, role_network):
        self.args = args
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.role_interval = role_interval

        self.agent = q_network
        self.role_agent = role_network
        self.n_roles = 3
        self.roles = [DotRole(args) for _ in range(self.n_roles)]

        self.action_selector = None
        self.role_selector = DotSelector(args)
        self.action_encoder = action_encoder

        self.hidden_states = None
        self.role_hidden_states = None
        self.selected_roles = None
        self.n_clusters = args.n_role_clusters
        self.role_action_spaces = tf.ones([self.n_roles, self.n_actions])

        self.role_latent = tf.ones([self.n_roles, self.args.action_latent_dim])
        self.action_repr = tf.ones([self.n_actions, self.args.action_latent_dim])
    
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, role_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, t_env=t_env)
        # the function forward returns q values of each agent, the roles are indicated by self.selected_roles

        # filter out actions infeasible for selected roles; self.selected_roles [bs*n_agents]
        # self.role_action_spaces [n_roles, n_actions]
        role_action_spaces = tf.tile(tf.expand_dims(self.role_action_spaces, axis=0), [self.n_agents, 1, 1])
        indices = self.selected_roles[:, :, tf.newaxis, tf.newaxis] * tf.ones([1, 1, self.n_actions], dtype=tf.int64)
        indices = tf.cast(indices, dtype=tf.int32)
        role_avail_actions = tf.squeeze(tf.gather(role_action_spaces, indices, axis=1))
        role_avail_actions = tf.cast(role_avail_actions, dtype=tf.int32)
        role_avail_actions = tf.reshape(role_avail_actions, [ep_batch.batch_size, self.n_agents, -1])

        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs],
                                                            role_avail_actions[bs], t_env, test_mode=test_mode)
        
        return chosen_actions, self.selected_roles, role_avail_actions

    def forward(self, observations, t, test_mode=False, t_env=None):
        # select roles
        self.role_hidden_states = self.role_agent(observations, self.role_hidden_states)
        role_outputs = None
        if t % self.role_interval == 0:
            role_outputs = self.role_selector(self.role_hidden_states, self.role_latent)
            self.selected_roles = tf.squeeze(self.role_selector.select_role(role_outputs, test_mode=test_mode, t_env=t_env))
            # [bs * n_agents]

        # compute individual q-values
        self.hidden_states = self.agent(observations, self.hidden_states)
        roles_q = []
        for role_i in range(self.n_roles):
            role_q = self.roles[role_i](self.hidden_states, self.action_repr)  # [bs * n_agents, n_actions]
            roles_q.append(role_q)
        roles_q = tf.stack(roles_q, axis=1)
        agent_outs = tf.gather(roles_q, self.selected_roles[:, :, tf.newaxis, tf.newaxis] \
                               * tf.ones([1, 1, self.n_actions], dtype=tf.int64), axis=1)
        # [bs * n_agents, 1, n_actions]

        agent_outs = tf.reshape(agent_outs, [observations.shape[0], self.n_agents, -1])
        role_outputs = tf.reshape(role_outputs, [observations.shape[0], self.n_agents, -1])

        return agent_outs, role_outputs
