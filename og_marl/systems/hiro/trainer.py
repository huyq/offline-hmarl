# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt
import numpy as np
import trfl

from og_marl.systems import TrainerBase
from og_marl.utils.trainer_utils import (
    gather,
    sample_batch_agents,
    switch_two_leading_dims,
    merge_batch_and_agent_dim_of_time_major_sequence,
    expand_batch_and_agent_dim_of_time_major_sequence,
    batch_concat_agent_id_to_obs
)

class HIROTrainer(TrainerBase):
    def __init__(
        self,
        agents,
        dataset,
        logger,
        hi_policy_network,
        hi_critic_network,
        lo_q_network,
        hi_policy_optimizer,
        hi_critic_optimizer,
        lo_q_optimizer,
        discount=0.99,
        gaussian_noise_network=None,
        target_update_period=200,
        target_update_rate=0.01,
        max_gradient_norm=20.0,
        add_agent_id_to_obs=False,
        max_trainer_steps=1e6,
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

        self.hi_interval = 5
        self.reward_scaling = 0.2
        self._lambda = 0.6

        self._hi_policy_optimizer = hi_policy_optimizer
        self._hi_critic_optimizer_1 = hi_critic_optimizer
        self._hi_critic_optimizer_2 = snt.optimizers.Adam(5e-4)

        self._hi_policy_network = hi_policy_network
        self._hi_critic_network_1 = hi_critic_network
        self._hi_critic_network_2 = copy.deepcopy(hi_critic_network)

        self._lo_q_network = lo_q_network
        self._lo_q_optimizer = lo_q_optimizer
        self._target_q_network = copy.deepcopy(lo_q_network)
        self._target_update_period = target_update_period
        self._target_update_rate = target_update_rate

        # Change critic 2s variables 
        critic_1_variables = (
            *self._hi_critic_network_1.variables,
        )
        critic_2_variables = (
            *self._hi_critic_network_2.variables,
        )   
        for src, dest in zip(critic_1_variables, critic_2_variables):
            dest.assign(-1.0 * src)


        self._system_variables.update(
            {
                "policy_network": self._hi_policy_network.variables,
                "q_network": self._lo_q_network.variables,
            }
        )

        # Target networks
        self._target_critic_network_1 = copy.deepcopy(hi_critic_network)
        self._target_critic_network_2 = copy.deepcopy(hi_critic_network)
        self._target_policy_network = copy.deepcopy(self._hi_policy_network)

        # Target update
        self._target_update_rate = target_update_rate

        # Gaussian noise network for target actions
        self._gaussian_noise_network = gaussian_noise_network

        # For logging
        self._policy_loss = tf.Variable(0.0, trainable=False, dtype="float32")

    @tf.function()
    def _train(self, sample, trainer_step):
        batch = sample_batch_agents(self._agents, sample, independent=True)

        # Get the relevant quantities
        observations = batch["observations"]
        replay_actions = batch["actions"]
        legal_actions = batch["legals"]
        states = batch["states"]
        rewards = batch["rewards"]
        env_discounts = tf.cast(batch["discounts"], "float32")
        mask = tf.cast(batch["mask"], "float32")  # shape=(B,T)

        # Get dims
        B, T, N, A = legal_actions.shape

        # Maybe add agent ids to observation
        # if self._add_agent_id_to_obs:
            # observations = batch_concat_agent_id_to_obs(observations)
        
        # Make time-major
        observations = switch_two_leading_dims(observations)
        # replay_actions = switch_two_leading_dims(replay_actions)
        

        if states is not None:
            states = switch_two_leading_dims(states)

        # train low level q net
        observations = merge_batch_and_agent_dim_of_time_major_sequence(observations)
        hi_observations = observations[slice(0,T,self.hi_interval)]
        hi_actions, _ = snt.static_unroll(
            self._hi_policy_network,
            hi_observations,
            self._hi_policy_network.initial_state(B*N)
        )

        # compute subgoals
        subgoals = []
        for t in range(T):
            if t%self.hi_interval==0:
                sg = hi_actions[t//self.hi_interval]
            else:
                sg = observations[t-1]+subgoals[-1]-observations[t]
            subgoals.append(sg) 
        subgoals = tf.stack(subgoals)

        
        # compute low-level reward
        lo_rewards= [self.low_reward(observations[t], sg, observations[t+1]) for t in range(T-1)]
        lo_rewards.append(tf.zeros_like(lo_rewards[0]))
        lo_rewards = tf.stack(lo_rewards)
        lo_rewards = tf.reshape(lo_rewards, [T,B,N])
        lo_rewards = switch_two_leading_dims(lo_rewards)

        # concat obs and subgoals as low-level input
        lo_inputs = tf.concat([observations, subgoals], axis=-1)
        target_qs_out_lo, _ = snt.static_unroll(
            self._target_q_network, 
            lo_inputs, 
            self._target_q_network.initial_state(B*N)
        )

        target_qs_out_lo = expand_batch_and_agent_dim_of_time_major_sequence(target_qs_out_lo, B, N)
        target_qs_out_lo = switch_two_leading_dims(target_qs_out_lo)

        with tf.GradientTape() as tape:
            # Unroll online network
            qs_out_lo, _ = snt.static_unroll(
                self._lo_q_network, 
                lo_inputs, 
                self._lo_q_network.initial_state(B*N)
            )

            # Expand batch and agent_dim
            qs_out_lo = expand_batch_and_agent_dim_of_time_major_sequence(qs_out_lo, B, N)

            # Make batch-major again
            qs_out_lo = switch_two_leading_dims(qs_out_lo)

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qs_lo = gather(qs_out_lo, replay_actions, axis=3, keepdims=False)

            # Max over target Q-Values/ Double q learning
            target_max_qs_lo = self._get_target_max_qs(
                qs_out_lo, target_qs_out_lo, legal_actions
            )

            # Maybe do mixing (Noop in IQL)
            # chosen_action_qs_lo, target_max_qs_lo = self._mixing(
                # chosen_action_qs_lo, target_max_qs_lo, states
            # )

            # Compute targets
            targets_lo = self._compute_targets(
                lo_rewards, env_discounts, target_max_qs_lo
            )  # shape=(B,T-1)

            # Chop off last time step
            chosen_action_qs_lo = chosen_action_qs_lo[:, :-1]  # shape=(B,T-1)

            # TD-Error Loss
            loss_lo = 0.5 * tf.square(targets_lo - chosen_action_qs_lo)

            # Mask out zero-padded timesteps
            loss_lo = self._apply_mask(loss_lo, mask)
        
        # Get trainable variables
        variables_lo = (*self._lo_q_network.trainable_variables,)

        # Compute gradients.
        gradients = tape.gradient(loss_lo, variables_lo)

        # Maybe clip gradients.
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

        # Apply gradients.
        self._lo_q_optimizer.apply(gradients, variables_lo)

        # Get online and target variables
        online_variables_lo = (
            *self._lo_q_network.variables,
        )
        target_variables_lo = (
            *self._target_q_network.variables,
        )

        # Maybe update target network
        self._update_target_network(online_variables_lo, target_variables_lo)


        rewards = switch_two_leading_dims(rewards)
        replay_actions = switch_two_leading_dims(replay_actions)
        env_discounts = switch_two_leading_dims(env_discounts)
        mask = switch_two_leading_dims(mask)

    
        hi_observations = observations[slice(0,T,self.hi_interval)]
        lo_states = tf.reshape(observations, [T//self.hi_interval, self.hi_interval, *observations.shape[1:]])
        lo_actions = tf.reshape(replay_actions, [T//self.hi_interval, self.hi_interval, *replay_actions.shape[1:]])
        hi_actions = self.offline_corrections(hi_actions, lo_states, lo_actions)
  
        hi_rewards = tf.reshape(rewards, [T//self.hi_interval, self.hi_interval, *rewards.shape[1:]])
        hi_rewards = self.reward_scaling * tf.reduce_sum(hi_rewards, 1)
        hi_rewards = merge_batch_and_agent_dim_of_time_major_sequence(hi_rewards)
        hi_env_discounts = env_discounts[slice(0,T,self.hi_interval)]
        hi_env_discounts = merge_batch_and_agent_dim_of_time_major_sequence(hi_env_discounts)
        mask = mask[slice(0,T,self.hi_interval)]


        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:
            # Unroll hi-level target policy
            target_actions_hi, _ = snt.static_unroll(
                self._target_policy_network,
                hi_observations,
                self._target_policy_network.initial_state(B*N)
            )
            target_actions_hi = expand_batch_and_agent_dim_of_time_major_sequence(target_actions_hi, B, N)
            if self._gaussian_noise_network:
                noisy_target_actions_hi = self._gaussian_noise_network(target_actions_hi)

            noisy_target_actions_hi = merge_batch_and_agent_dim_of_time_major_sequence(noisy_target_actions_hi)
            # Target critics
            target_qs_1_hi = self._target_critic_network_1(hi_observations, noisy_target_actions_hi)
            target_qs_2_hi = self._target_critic_network_2(hi_observations, noisy_target_actions_hi)

            target_qs_hi = tf.squeeze(tf.minimum(target_qs_1_hi, target_qs_2_hi))
            # Compute Bellman targets
            targets_hi = tf.stop_gradient(
                hi_rewards[:-1]
                + self._discount * hi_env_discounts[:-1] * target_qs_hi[1:]
            )
            # Online critics
            qs_1 = tf.squeeze(self._hi_critic_network_1(hi_observations, hi_actions))
            qs_2 = tf.squeeze(self._hi_critic_network_2(hi_observations, hi_actions))



            # Squared TD-Error
            critic_loss_1 = 0.5 * (targets_hi - qs_1[:-1]) ** 2
            critic_loss_2 = 0.5 * (targets_hi - qs_2[:-1]) ** 2

            # Masked mean
            critic_mask = tf.squeeze(tf.stack([mask[:-1]]*N, axis=2))
            critic_mask = merge_batch_and_agent_dim_of_time_major_sequence(critic_mask)
            critic_loss_1 = tf.reduce_sum(critic_loss_1 * critic_mask) / tf.reduce_sum(critic_mask)
            critic_loss_2 = tf.reduce_sum(critic_loss_2 * critic_mask) / tf.reduce_sum(critic_mask)

        # Train critic 1
        variables = (
            *self._hi_critic_network_1.trainable_variables,
        )
        gradients = tape.gradient(critic_loss_1, variables)
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]
        self._hi_critic_optimizer_1.apply(gradients, variables)

        # # Train critic 2
        variables = (
            *self._hi_critic_network_2.trainable_variables,
        )
        gradients = tape.gradient(critic_loss_2, variables)
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]
        self._hi_critic_optimizer_2.apply(gradients, variables)


        # Update target networks
        online_variables = (
            *self._hi_critic_network_1.variables,
            *self._hi_critic_network_2.variables,
        )
        target_variables = (
            *self._target_critic_network_1.variables,
            *self._target_critic_network_2.variables,
        )   
        self._update_target_network(
            online_variables,
            target_variables,
        )

        del tape # clear the gradient tape

        if trainer_step % 2 == 0:  # TD3 style delayed policy update

            # Compute policy loss
            with tf.GradientTape(persistent=True) as tape:
                
                # Unroll online policy
                online_actions, _ = snt.static_unroll(
                    self._hi_policy_network,
                    hi_observations,
                    self._hi_policy_network.initial_state(B*N)
                )
                # online_actions = expand_batch_and_agent_dim_of_time_major_sequence(online_actions, B, N)

                qs_1 = self._hi_critic_network_1(hi_observations, online_actions)
                qs_2 = self._hi_critic_network_2(hi_observations, online_actions)
                qs = tf.reduce_mean((qs_1, qs_2), axis=0)
                
                policy_loss = - tf.squeeze(qs)

                # Masked mean
                policy_mask = tf.squeeze(tf.stack([mask] * N, axis=2))
                policy_mask = merge_batch_and_agent_dim_of_time_major_sequence(policy_mask)
                policy_loss = tf.reduce_sum(policy_loss * policy_mask) / tf.reduce_sum(policy_mask)

            # Train policy
            variables = (*self._hi_policy_network.trainable_variables,)
            gradients = tape.gradient(policy_loss, variables)
            gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]
            self._hi_policy_optimizer.apply(gradients, variables)

            del tape # clear gradient tape

            # Update target policy
            online_variables = (*self._hi_policy_network.variables,)
            target_variables = (*self._target_policy_network.variables,)
            self._update_target_network(
                online_variables,
                target_variables,
            )

            # For logging 
            self._policy_loss.assign(policy_loss)

        logs = {
            "Mean Q-values": tf.reduce_mean((qs_1 + qs_2) / 2),
            "Trainer Steps": trainer_step,
            "Critic Loss": (critic_loss_1 + critic_loss_2) / 2,
            "Policy Loss": self._policy_loss,
        }


        return logs

    def _update_target_network(
        self, online_variables, target_variables
    ):
        """Update the target networks."""

        tau = self._target_update_rate
        for src, dest in zip(online_variables, target_variables):
            dest.assign(dest * (1.0 - tau) + src * tau)
    
    def _get_target_max_qs(self, qs_out, target_qs_out, legal_actions):
        qs_out_selector = tf.where(
            legal_actions, qs_out, -9999999
        )  # legal action masking
        cur_max_actions = tf.argmax(qs_out_selector, axis=3)
        target_max_qs = gather(target_qs_out, cur_max_actions, axis=-1)
        return target_max_qs
    
    def _compute_targets(self, rewards, env_discounts, target_max_qs):
        if self._lambda is not None:
            # Get time and batch dim
            B, T = rewards.shape[:2]
 
            # Duplicate rewards and discount for all agents
            rewards = tf.broadcast_to(rewards, target_max_qs.shape)
            env_discounts = tf.broadcast_to(env_discounts, target_max_qs.shape)

            # Make time major for trfl
            rewards = tf.transpose(rewards, perm=[1, 0, 2])
            env_discounts = tf.transpose(env_discounts, perm=[1, 0, 2])
            target_max_qs = tf.transpose(target_max_qs, perm=[1, 0, 2])

            # Flatten agent dim into batch-dim
            rewards = tf.reshape(rewards, shape=(T, -1))
            env_discounts = tf.reshape(env_discounts, shape=(T, -1))
            target_max_qs = tf.reshape(target_max_qs, shape=(T, -1))

            # Q(lambda)
            targets = trfl.multistep_forward_view(
                rewards[:-1],
                self._discount * env_discounts[:-1],
                target_max_qs[1:],
                lambda_=self._lambda,
                back_prop=False,
            )
            # Unpack agent dim again
            targets = tf.reshape(targets, shape=(T - 1, B, -1))

            # Make batch major again
            targets = tf.transpose(targets, perm=[1, 0, 2])
        else:
            targets = (
                rewards[:, :-1]
                + self._discount * env_discounts[:, :-1] * target_max_qs[:, 1:]
            )
        return tf.stop_gradient(targets)
    
    def _apply_mask(self, loss, mask):
        mask = tf.broadcast_to(mask[:, :-1], loss.shape)
        loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
        return loss

    def _add_extras_to_critic_loss(self, critic_loss_1, critic_loss_2, qs_1, qs_2, observations, states, mask):
        return critic_loss_1, critic_loss_2, {} # extras

    def _critic_apply_extras(self, tape, critic_extras):
        return

    def _add_extras_to_policy_loss(self, policy_loss, online_actions, replay_actions, mean_qs):
        return policy_loss

    def after_train_step(self):
        info = {}
        return info
    
    def offline_corrections(self, original_goals, lo_states, lo_actions, k=8):
        T_hi, T_lo, B, A = lo_actions.shape
        diff_goal = lo_states[:,-1] - lo_states[:,0]                # T*1*A

        random_goals = tf.random.normal(shape=(k, *original_goals.shape), mean=diff_goal, stddev=0.5*0.02)
        random_goals = tf.clip_by_value(random_goals, 0, 1)      

        candidates = tf.concat(
            [
              tf.expand_dims(original_goals,0),
              tf.expand_dims(diff_goal, 0),
              random_goals
            ], axis=0)   ## Tx10*A
        # candidates = switch_two_leading_dims(candidates)
        flat_lo_states = tf.stack([lo_states for _ in range(k+2)], axis=0)
        input_candidates = tf.stack([(candidates+flat_lo_states[:,:,0]) for _ in 
                                     range(flat_lo_states.shape[2])], axis=2) - flat_lo_states  ## Txkx
        input_candidates = tf.reshape(input_candidates, [input_candidates.shape[0],-1, *input_candidates.shape[3:]])
        # input_candidates = switch_two_leading_dims(input_candidates)
        

        lo_actions = tf.stop_gradient(lo_actions)
        lo_states = tf.reshape(lo_states, [-1, *lo_states.shape[2:]])

        # pred_lo_actions = self._lo_q_network(tf.concat([lo_states, input_candidates], axis=-1)) ## 
        pred_lo_actions = []
        for i in range(k+2):
            pred_qs, _ = snt.static_unroll(
                self._target_q_network, 
                tf.concat([lo_states, input_candidates[i]], axis=-1),
                self._target_q_network.initial_state(lo_states.shape[1])
            )
            pred_lo_actions.append(pred_qs)
        pred_lo_actions = tf.stack(pred_lo_actions)
        pred_logits = tf.math.softmax(pred_lo_actions)
        pred_logits = tf.reshape(pred_logits, [k+2, T_hi, T_lo, *pred_logits.shape[2:]])


        lo_actions = tf.reshape(lo_actions, [T_hi, T_lo, B*A])
        flat_lo_actions = tf.stack([lo_actions for _ in range(k+2)])
        flat_lo_actions_onehot = tf.one_hot(flat_lo_actions, depth=pred_logits.shape[-1])
        log_probs = tf.reduce_sum(pred_logits * flat_lo_actions_onehot, -1) 
        log_probs = tf.reduce_sum(log_probs, axis=2)    

        best_actions  = tf.argmax(log_probs, axis=0, output_type=tf.int32) 
        best_actions = tf.reshape(best_actions, [best_actions.shape[0]*best_actions.shape[1]])
        
 
        # actions = tf.gather_nd(
            # candidates,
            # tf.stack([best_actions, tf.range(original_goals.shape[0], dtype=tf.int64)], -1))
        
        candidates = tf.reshape(candidates, [k+2, T_hi*B*A, -1])
        candidates = switch_two_leading_dims(candidates)

        actions = tf.gather_nd(candidates, tf.stack([tf.range(candidates.shape[0]), best_actions], axis=1))
        actions = tf.reshape(actions, [*original_goals.shape])
        
        return actions


    def subgoal_transition(self, s, sg, n_s):
        return s[:sg.shape[0]] + sg - n_s[:sg.shape[0]]

    def low_reward(self, s, sg, n_s):
        squared_diffs = tf.reduce_sum((s+sg-n_s)**2, axis=-1)
        low_reward = -tf.sqrt(squared_diffs)
        return low_reward
    




    




