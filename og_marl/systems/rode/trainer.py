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
import sonnet as snt

from og_marl.systems.iql import IQLTrainer 
from og_marl.utils.trainer_utils import (
    gather,
    batch_concat_agent_id_to_obs,
    switch_two_leading_dims,
    merge_batch_and_agent_dim_of_time_major_sequence,
    expand_batch_and_agent_dim_of_time_major_sequence,
    sample_batch_agents
)

class RODETrainer(IQLTrainer):
    def __init__(
        self,
        agents,
        dataset,
        logger,
        optimizer,
        q_network,
        mixer,
        action_encoder,
        discount=0.99,
        lambda_=0.6,
        max_gradient_norm=20.0,
        add_agent_id_to_obs=False,
        target_update_rate=0.001,
        max_trainer_steps=1e6,
    ):
        super().__init__(
            agents=agents,
            dataset=dataset,
            logger=logger,
            optimizer=optimizer,
            q_network=q_network,
            discount=discount,
            lambda_=lambda_,
            max_gradient_norm=max_gradient_norm,
            add_agent_id_to_obs=add_agent_id_to_obs,
            target_update_rate=target_update_rate,
            max_trainer_steps=max_trainer_steps,
        )

        self._mixer = mixer
        self._target_mixer = copy.deepcopy(mixer)

        self._action_encoder = action_encoder
        self._optimizer = optimizer

    def _batch_agents(self, agents, sample):
        return sample_batch_agents(agents, sample, independent=False)

    def _mixing(self, chosen_action_qs, target_max_qs, states):
        """QMIX"""
        chosen_action_qs = self._mixer(chosen_action_qs, states)
        target_max_qs = self._target_mixer(target_max_qs, states)
        return chosen_action_qs, target_max_qs

    def _get_trainable_variables(self):
        variables = (
            *self._q_network.trainable_variables,
            *self._mixer.trainable_variables,
        )
        return variables

    def _get_variables_to_update(self):
        # Online variables
        online_variables = (
            *self._q_network.variables,
            *self._mixer.variables,
        )

        # Get target variables
        target_variables = (
            *self._target_q_network.variables,
            *self._target_mixer.variables,
        )

        return online_variables, target_variables
    
    @tf.function
    def _train_action_encoder(self, sample, trainer_step):
        batch = self._batch_agents(self._agents, sample)
        observations = batch["observations"]
        actions = batch["actions"]
        legal_actions = batch["legals"]
        # states = batch["states"]
        # rewards = batch["rewards"]
        # env_discounts = tf.cast(batch["discounts"], "float32")
        # mask = tf.cast(batch["mask"], "float32")  # shape=(B,T)

        B, T, N, A = legal_actions.shape
        actions_onehot = tf.one_hot(actions, A)
        actions_onehot = tf.reshape(actions_onehot, [-1, N, A])

        # observations = switch_two_leading_dims(observations)
        # observations = merge_batch_and_agent_dim_of_time_major_sequence(observations)
        trailing_dims = observations.shape[3:]
        _observations = tf.reshape(observations, [B*T,N,*trailing_dims])
        
        pred_obs_loss = None
        pred_r_loss = None

        with tf.GradientTape() as tape: 
            no_pred, r_pred = self._action_encoder.predict(_observations, actions_onehot)

            no_pred = tf.reshape(no_pred, [B,T,N,-1])[:,:-1]
            no = observations[:, 1:]
            repeated_rewards = tf.repeat(batch["rewards"][:, :-1], repeats=N)

            pred_obs_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum((no_pred-no)**2)))
            pred_r_loss = tf.reduce_mean((r_pred - repeated_rewards)**2)

            pred_loss = pred_obs_loss + 10 * pred_r_loss
        
        variables = (
            *self._action_encoder.trainable_variables,
        )

        gradients = tape.gradient(pred_loss, variables)
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

        self._optimizer.apply(gradients, variables)

        return {
            "Loss": pred_loss,
            "Trainer Steps": trainer_step,
        }


    @tf.function
    def _train(self, sample, trainer_step):
        batch = self._batch_agents(self._agents, sample)
        observations = batch["observations"]
        actions = batch["actions"]
        legal_actions = batch["legals"]
        states = batch["states"]
        rewards = batch["rewards"]
        env_discounts = tf.cast(batch["discounts"], "float32")
        mask = tf.cast(batch["mask"], "float32")  # shape=(B,T)


        B, T, N, A = legal_actions.shape

  
        with tf.GradientTape(persistent=True) as tape:
            for t in range(T):
                action_outs, role_outs = self.mac.forward(observations, t=t)

            chosen_action_qs = tf.gather(action_outs[:, :-1], actions, batch_dims=3, axis=3)  # Remove the last dim
            chosen_action_qs = tf.squeeze(chosen_action_qs, axis=3)

            chosen_role_qs = tf.gather(role_outs, roles, batch_dims=3, axis=3)
            chosen_role_qs = tf.squeeze(chosen_role_qs, axis=3)


            target_action_outs, target_role_outs = self.mac.forward(observations)
            target_role_outs.append(tf.zeors([B,N,self.mac.n_roles]))

            target_action_outs = tf.stack(target_action_outs[1:], axis=1)  # Concat across time
            target_role_outs = tf.stack(target_role_outs[1:], axis=1)

            target_action_outs[legal_actions[:, 1:] == 0] = -9999999

            target_action_max_qs = self._get_target_max_qs(
                action_outs, target_action_outs, legal_actions
            )

            target_role_max_qs = self._get_target_max_qs(
                role_outs, target_role_outs, legal_actions
            )

            if self.mixer is not None:
                chosen_action_qs, target_action_max_qs = self._mixing(
                    chosen_action_qs, target_action_max_qs, states
                )
            if self.role_mixer is not None:
                chosen_role_qs, target_role_max_qs = self._mixing(
                    chosen_role_qs, target_role_max_qs, states
                )

            # Calculate 1-step Q-Learning targets
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
            rewards_shape = list(rewards.shape)
            rewards_shape[1] = role_t
            role_rewards = th.zeros(rewards_shape).to(self.device)
            role_rewards[:, :rewards.shape[1]] = rewards.detach().clone()
            role_rewards = role_rewards.view(batch.batch_size, role_at,
                                             self.role_interval).sum(dim=-1, keepdim=True)
            # role_terminated
            terminated_shape_o = terminated.shape
            terminated_shape = list(terminated_shape_o)
            terminated_shape[1] = role_t
            role_terminated = th.zeros(terminated_shape).to(self.device)
            role_terminated[:, :terminated_shape_o[1]] = terminated.detach().clone()
            role_terminated = role_terminated.view(batch.batch_size, role_at, self.role_interval).sum(dim=-1, keepdim=True)
            # role_terminated
            role_targets = role_rewards + self.args.gamma * (1 - role_terminated) * target_role_max_qvals

            # Td-error
            td_error = (chosen_action_qvals - targets.detach())
            role_td_error = (chosen_role_qvals - role_targets.detach())

            mask = mask.expand_as(td_error)
            mask_shape = list(mask.shape)
            mask_shape[1] = role_t
            role_mask = th.zeros(mask_shape).to(self.device)
            role_mask[:, :mask.shape[1]] = mask.detach().clone()
            role_mask = role_mask.view(batch.batch_size, role_at, self.role_interval, -1)[:, :, 0]

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask
            masked_role_td_error = role_td_error * role_mask

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask.sum()
            role_loss = (masked_role_td_error ** 2).sum() / role_mask.sum()
            loss += role_loss

        # Get trainable variables
        variables = self._get_trainable_variables()

        # Compute gradients.
        gradients = tape.gradient(loss, variables)

        # Maybe clip gradients.
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

        # Apply gradients.
        self._optimizer.apply(gradients, variables)

        # Get online and target variables
        online_variables, target_variables = self._get_variables_to_update()

        # Maybe update target network
        self._update_target_network(online_variables, target_variables, trainer_step)

        return {
            "Loss": loss,
            "Mean Q-values": tf.reduce_mean(qs_out),
            "Mean Chosen Q-values": tf.reduce_mean(chosen_action_qs),
            "Trainer Steps": trainer_step,
        }





            

        
        















