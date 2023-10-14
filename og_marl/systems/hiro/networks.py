"""Independent and Centralised Critic Implementations for TD3 based systems."""
import tensorflow as tf
import sonnet as snt

from og_marl.utils.trainer_utils import batch_concat_agent_id_to_obs

class ObservationAndActionCritic(snt.Module):

    def __init__(self, num_agents, num_actions):
        self.N = num_agents
        self.A = num_actions

        self._critic_network = snt.Sequential(
            [
                snt.Linear(128),
                tf.keras.layers.ReLU(),
                snt.Linear(128),
                tf.keras.layers.ReLU(),
                snt.Linear(1)
            ]
        )

        super().__init__()

    def initialise(self, observation, action):
        """ A method to initialise the parameters in the critic network.

        observation: a dummy observation with no batch dimension. We assume
            all agent's observations have the same shape.
        state: a dummy environment state with no batch dimension.
        action: a dummy action with no batch dimension. We assume all agents have 
            the same action shape.
        """
        observation = tf.stack([observation]*self.N, axis=0) # observation for each agent
        observation = tf.reshape(observation, (1,1) + observation.shape) # add time and batch dim
        actions = tf.stack([action]*self.N, axis=0) # action for each agent
        actions = tf.reshape(actions, (1,1) + actions.shape) # add time and batch dim


        self(observation, actions) # __call__ with dummy inputs

    def __call__(self, observations, agent_actions):
        """Forward pass of critic network.
        
        observations [T,B,N,O]
        states [T,B,S]
        agent_actions [T,B,N,A]: the actions the agent took.
        other_actions [T,B,N,A]: the actions the other agents took.
        """
        # Concat states and joint actions
        critic_input = tf.concat([observations, agent_actions], axis=-1)

        # Concat agent IDs to critic input
        # critic_input = batch_concat_agent_id_to_obs(critic_input)

        q_values = self._critic_network(critic_input)
        return q_values