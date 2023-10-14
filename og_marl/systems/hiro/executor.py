import tree
import numpy as np
import tensorflow as tf
from acme.tf import utils as tf2_utils
import tensorflow_probability as tfp

from og_marl.systems import ExecutorBase
from og_marl.utils.executor_utils import (
    epsilon_greedy_action_selection, 
    concat_agent_id_to_obs
)

class HIROExecutor(ExecutorBase):
    def __init__(
        self,
        agents,
        variable_client,
        hi_policy_network,
        lo_q_network,
        adder=None,
        add_agent_id_to_obs=False,
        gaussian_noise_network=None,
        exploration_timesteps=0,
        checkpoint_subpath="",
        must_checkpoint=False,
    ):
        super().__init__(
            agents=agents,
            variable_client=variable_client,
            add_agent_id_to_obs=add_agent_id_to_obs,
            checkpoint_subpath=checkpoint_subpath,
            must_checkpoint=must_checkpoint,
        )

        # Store optional adder
        self._adder = adder

        # Store networks
        self._hi_policy_network = hi_policy_network
        self._gaussian_noise_network = gaussian_noise_network

        self._lo_q_network = lo_q_network

        # Recurrent core states for policy network, per agent
        self._core_states_hi = {agent: None for agent in agents}
        self._core_states_lo = {agent: None for agent in agents}

        # subgoal settings
        self.sg = {agent: None for agent in agents}
        self._hi_interval = 5

        # Exploration
        self._exploration_timesteps = exploration_timesteps

        # Epsilon-greedy exploration
        self._eps = 1.0
        self._eps_dec = 1e-5
        self._eps_min = 0.05

        # Counter
        self._timestep = 0

        # Checkpointing
        self._variables_to_checkpoint.update(
            {"hi_policy_network": self._hi_policy_network.variables,
             "lo_q_network": self._lo_q_network.variables}
        )
        if self._must_checkpoint:
            self.restore_checkpoint()
    
    def observe_first(self, timestep, extras={}):
        # Re-initialize the recurrent core states for Q-network
        for agent in self._agents:
            self._core_states_hi[agent] = self._hi_policy_network.initial_state(1)
            self._core_states_lo[agent] = self._lo_q_network.initial_state(1)

        if self._adder is not None:

            # Adder first timestep to adder
            extras.update({"zero_padding_mask": np.array(1)})

            self._adder.add_first(timestep, extras)
    
    def observe(self, actions, next_timestep, next_extras={}):

        if self._adder is not None:

            # Add core states to extras
            next_extras.update({"zero_padding_mask": np.array(1)})

            # Add timestep to adder
            self._adder.add(actions, next_timestep, next_extras)
    
    def select_actions(self, observations):
        # Get agent actions
        epsilon = self._decay_epsilon()
        epsilon = tf.convert_to_tensor(epsilon, dtype="float32")
        actions, next_core_states = self._select_actions(
            observations,  epsilon
        )

        # Update core states
        for agent in self._core_states_lo.keys():
            self._core_states_lo[agent] = next_core_states[agent]

        # TODO: either do this or _select_action, not both
        

        self._timestep += 1

        # Convert actions to numpy
        actions = tree.map_structure(tf2_utils.to_numpy_squeeze, actions)

        return actions
    
    @tf.function
    def _select_actions(self, observations, epsilon):
        lo_actions = {}
        next_core_states_lo = {}
        for agent in observations.keys():
            if self._timestep % self._hi_interval == 0:
                hi_action, next_core_states_hi = self._select_action_hi(
                    agent,
                    observations[agent].observation,
                    observations[agent].legal_actions,
                    self._core_states_hi[agent],
                )

                if self._gaussian_noise_network is not None:
                    hi_action = self._gaussian_noise_network(hi_action)

                self.sg[agent] = hi_action
                self._core_states_hi[agent] = next_core_states_hi
                
            else:
                # subgoal transition
                self.sg[agent] += (self.last_obs[agent] - observations[agent].observation)
            
            lo_actions[agent], next_core_states_lo[agent] = self._select_action_lo(
                agent,
                tf.concat([observations[agent].observation, tf.squeeze(self.sg[agent])], axis=-1),
                observations[agent].legal_actions,
                self._core_states_lo[agent],
                epsilon,
            )
        

        return lo_actions, next_core_states_lo
    
    def _select_action_hi(self, agent, observation, legal_actions, core_state):
        # Add agent ID to embed
        if self._add_agent_id_to_obs:
            agent_id = self._agents.index(agent)
            observation = concat_agent_id_to_obs(
                observation, agent_id, len(self._agents)
            )

        # Add a dummy batch dimension
        observation = tf.expand_dims(observation, axis=0)
        legal_actions = tf.expand_dims(legal_actions, axis=0)

        # Pass observation embedding through policy network
        hi_action, next_core_state_hi = self._hi_policy_network(observation, core_state)

        return hi_action, next_core_state_hi

    def _select_action_lo(self, agent, observation, legal_actions, core_state, eps):
        # Add agent ID to embed
        if self._add_agent_id_to_obs:
            agent_id = self._agents.index(agent)
            observation = concat_agent_id_to_obs(
                observation, agent_id, len(self._agents)
            )

        # Add a dummy batch dimension
        observation = tf.expand_dims(observation, axis=0)
        legal_actions = tf.expand_dims(legal_actions, axis=0)

        # Pass observation through Q-network
        action_values, next_core_state = self._lo_q_network(observation, core_state)

        # Pass action values through action selector
        action, _ = epsilon_greedy_action_selection(
            action_values=action_values, legal_actions=legal_actions, epsilon=eps
        )

        return action, next_core_state
    
    def _decay_epsilon(self):
        if self._eps_dec != 0:
            self._eps = self._eps - self._eps_dec
        self._eps = max(self._eps, self._eps_min)
        return self._eps

    def get_stats(self):
        """Return extra stats to log."""
        return {"Epsilon": self._eps}