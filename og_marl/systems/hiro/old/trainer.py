import copy
import tensorflow as tf
import trfl
import sonnet as snt
import numpy as np

from og_marl.systems import TrainerBase
from og_marl.utils.trainer_utils import (
    sample_batch_agents,
    gather,
    batch_concat_agent_id_to_obs,
    switch_two_leading_dims,
    merge_batch_and_agent_dim_of_time_major_sequence,
    expand_batch_and_agent_dim_of_time_major_sequence,
)

from .controller import HigherController, LowerController
from .utils import Subgoal

class HIROTrainer(TrainerBase):
    def __init__(
        self,
        agents,
        dataset,
        logger,
        state_dim,
        action_dim,
        subgoal_dim,
        scale_low,
        reward_scaling,
        subgoal_interval,
        discount=0.99,
        max_gradient_norm=20.0,
        add_agent_id_to_obs=False,
        target_update_rate=0.01,
        max_trainer_steps=1e5,
    ):
        super().__init__(
            agents=agents,
            dataset=dataset,
            logger=logger,
            discount=discount,
            max_gradient_norm=max_gradient_norm,
            add_agent_id_to_obs=add_agent_id_to_obs,
            max_trainer_steps=max_trainer_steps,
        )

        self.subgoal = Subgoal(subgoal_dim)
        scale_high = self.subgoal.action_space.high * np.ones(subgoal_dim)
        self.subgoal_interval = subgoal_interval

        self.high_con = HigherController(
            state_dim=state_dim,
            goal_dim=0,
            action_dim=subgoal_dim,
            scale=scale_high,
        )

        self.low_con = LowerController(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            scale=scale_low,
        )

        self.reward_scaling = reward_scaling
        self.episode_subreward = 0
        self.sr = 0

        self.sg = self.subgoal.action_space.sample()
    
    @tf.function
    def _train(self, sample, trainer_step):
        batch = self._batch_agents(self._agents, sample)

        # Get the relevant quantities
        observations = batch["observations"]
        actions = batch["actions"]
        legal_actions = batch["legals"]
        states = batch["states"]
        rewards = batch["rewards"]
        env_discounts = tf.cast(batch["discounts"], "float32")
        mask = tf.cast(batch["mask"], "float32")  # shape=(B,T)

        # Get dims
        B, T, N, A = legal_actions.shape

        # Maybe add agent ids to observation
        if self._add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)
        
        observations = switch_two_leading_dims(observations)
        observations = merge_batch_and_agent_dim_of_time_major_sequence(observations)

        actions = switch_two_leading_dims(actions)
        actions = merge_batch_and_agent_dim_of_time_major_sequence(actions)
        

        subgoals = []
        for t in range(T):
            if t%self.subgoal_interval==0:
                sg = self.high_con.forward(observations[t])
            else:
                sg = self.subgoal_transition(observations[t-1], sgoals[-1], observations[t])
            subgoals.append(sg)
        subgoals = np.concatenate(subgoals, 1)
        sgoals = subgoals[:-1]
        next_sgoals = subgoals[1:]


        obs = observations[:-1]
        next_obs = observations[1:]
        loss, td_error = self.low_con.train(obs, sgoals, actions, rewards, next_obs, next_sgoals, mask)



        obs_arr = observations.reshape(T//self.subgoal_interval, self.subgoal_interval, *observations.shape[1:])
        act_arr = actions.reshape(T//self.subgoal_interval, self.subgoal_interval, *actions.shape[1:])
        not_done = mask.reshape(T//self.subgoal_interval, self.subgoal_interval, *actions.shape[1:])
        not_done = not_done[:,0]
        high_obs = obs_arr[:,0]
        obs_arr = np.delete(obs_arr,0,1)
        act_arr = np.delete(act_arr,0,1)
        high_rewards = rewards.reshape(T//self.subgoal_interval, self.subgoal_interval, *rewards.shape[1:])
        high_rewards = self.reward_scaling*np.sum(high_rewards, 1)

        high_actions = subgoals[np.arange(0,T,self.subgoal_interval)]

        loss, td_error = self.high_con.train(high_obs[:-1], high_actions[:-1], high_obs[1:], high_rewards[:-1], \
                                             not_done[1:], obs_arr[:-1], act_arr[:-1], self.low_con)


        
        

        






