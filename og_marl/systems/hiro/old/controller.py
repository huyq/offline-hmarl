import tensorflow as tf 
import numpy as np
import sonnet as snt

class TD3Actor(snt.Module):
    def __init__(self, state_dim, goal_dim, action_dim, scale=None):
        super().__init__()
        if scale is None:
            scale = tf.ones(state_dim)
        else:
            scale = tf.Tensor(scale)
        self.scale = tf.Variable(scale.numpy(), trainable=False, dtype=tf.float32)

        self.l1 = snt.Linear(300)
        self.l2 = snt.Linear(300)
        self.l3 = snt.Linear(action_dim)
    
    def __call__(self, state, goal=None):
        if goal!=None:
            inputs = tf.concat([state, goal], 1)
        else:
            inputs = state
        x = tf.nn.relu(self.l1(inputs))
        x = tf.nn.relu(self.l2(x))
        x = self.scale * tf.nn.tanh(self.l3(x))
        return x

class TD3Critic(snt.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()
        # Q1
        self.Q1 = snt.Sequential(
            [
                snt.Linear(300),
                tf.nn.relu,
                snt.Linear(300),
                tf.nn.relu,
                snt.Linear(1),
            ]
        )
        # Q2
        self.Q2 = snt.Sequential(
            [
                snt.Linear(300),
                tf.nn.relu,
                snt.Linear(300),
                tf.nn.relu,
                snt.Linear(1),
            ]
        )

    def forward(self, state, action, goal=None):
        if goal!= None:
            sa = tf.concat([state, goal, action], 1)
        else:
            sa = tf.concat([state, action], 1)
        q = self.Q1(sa)
        return q



class TD3Controller(object):
    def __init__(
            self,
            state_dim,
            goal_dim,
            action_dim, 
            scale,
            model_path=None,
            actor_lr=0.0001,
            critic_lr=0.001,
            expl_noise=0.1,
            policy_noise=0.2,
            noise_clip=0.5,
            gamma=0.99,
            policy_freq=2,
            tau=0.005
        ):
        self.scale = scale
        self.model_path = model_path

        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_freq = policy_freq
        self.tau = tau

        self.actor = TD3Actor(state_dim, goal_dim, action_dim, scale=scale)
        self.actor_target = TD3Actor(state_dim, goal_dim, action_dim, scale=scale)
        self.actor_optimizer = snt.optimizers.Adam(actor_lr)

        self.critic1 = TD3Critic(state_dim, goal_dim, action_dim)
        self.critic2 = TD3Critic(state_dim, goal_dim, action_dim)
        self.critic1_target = TD3Critic(state_dim, goal_dim, action_dim)
        self.critic2_target = TD3Critic(state_dim, goal_dim, action_dim)
        self.critic1_optimizer = snt.optimizers.Adam(critic_lr)
        self.critic2_optimizer = snt.optimizers.Adam(critic_lr)

        self._initialize_target_networks()
        self._initialized = False
        self.total_it = 0

        self.name = 'td3'
    
    def _initialize_target_networks(self):
        self._update_target_network(self.critic1_target, self.critic1, 1.0)
        self._update_target_network(self.critic2_target, self.critic2, 1.0)
        self._update_target_network(self.actor_target, self.actor, 1.0)
        self._initialized = True

    def _update_target_network(self, target, origin, tau):
        for target_param, origin_param in zip(target.parameters(), origin.parameters()):
            target_param.data.copy_(tau * origin_param.data + (1.0 - tau) * target_param.data)

    def forward(self, state, goal):
        action = self.actor(state, goal)
        return tf.squeeze(action)

    def _train(self, states, goals, actions, rewards, n_states, n_goals, not_done):
        self.total_it += 1
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.actor.trainable_variables + self.critic1.trainable_variables + self.critic2.trainable_variables)
            noise = tf.clip_by_value(tf.random.normal(tf.shape(actions)) * self.policy_noise, -self.noise_clip, self.noise_clip)

            n_actions = self.actor_target([n_states, n_goals]) + noise
            n_actions = tf.clip_by_value(n_actions, -self.actor.scale, self.actor.scale)

            target_Q1 = self.critic1_target([n_states, n_goals, n_actions])
            target_Q2 = self.critic2_target([n_states, n_goals, n_actions])
            target_Q = tf.math.minimum(target_Q1, target_Q2)
            target_Q_detached = (rewards + not_done * self.gamma * target_Q).numpy()

            current_Q1 = self.critic1([states, goals, actions])
            current_Q2 = self.critic2([states, goals, actions])

            critic1_loss = tf.reduce_mean(tf.losses.huber(current_Q1, target_Q_detached))
            critic2_loss = tf.reduce_mean(tf.losses.huber(current_Q2, target_Q_detached))
            critic_loss = critic1_loss + critic2_loss

            td_error = tf.reduce_mean(target_Q_detached - current_Q1).numpy()

            actor_loss = None
            if self.total_it % self.policy_freq == 0:
                a = self.actor([states, goals])
                Q1 = self.critic1([states, goals, a])
                actor_loss = -tf.reduce_mean(Q1)

        grads = tape.gradient(critic_loss, self.critic1.trainable_variables + self.critic2.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(grads[:len(self.critic1.trainable_variables)], self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(grads[len(self.critic1.trainable_variables):], self.critic2.trainable_variables))

        if actor_loss is not None:
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            self._update_target_network(self.critic1_target, self.critic1, self.tau)
            self._update_target_network(self.critic2_target, self.critic2, self.tau)
            self._update_target_network(self.actor_target, self.actor, self.tau)

            return {'actor_loss_'+self.name: actor_loss, 'critic_loss_'+self.name: critic_loss}, \
                    {'td_error_'+self.name: td_error}
    
        return {'critic_loss_'+self.name: critic_loss}, \
                    {'td_error_'+self.name: td_error}

def HigherController(TD3Controller):
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        scale,
        model_path,
        actor_lr=0.0001,
        critic_lr=0.001,
        expl_noise=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005):
        super(HigherController, self).__init__(
            state_dim, goal_dim, action_dim, scale, model_path,
            actor_lr, critic_lr, expl_noise, policy_noise,
            noise_clip, gamma, policy_freq, tau
        )

        self.name = 'high'
        self.action_dim = action_dim
    
    def off_policy_corrections(self, low_con, batch_size, sgoals, states, actions, candidate_goals=8):
        first_s = [s[0] for s in states] # First x
        last_s = [s[-1] for s in states] # Last x

        # Shape: (batch_size, 1, subgoal_dim)
        # diff = 1
        diff_goal = (np.array(last_s) -
                     np.array(first_s))[:, np.newaxis, :self.action_dim]

        # Shape: (batch_size, 1, subgoal_dim)
        # original = 1
        # random = candidate_goals
        original_goal = np.array(sgoals)[:, np.newaxis, :]
        random_goals = np.random.normal(loc=diff_goal, scale=.5*self.scale[None, None, :],
                                        size=(batch_size, candidate_goals, original_goal.shape[-1]))
        random_goals = random_goals.clip(-self.scale, self.scale)

        # Shape: (batch_size, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)
        #states = np.array(states)[:, :-1, :]
        actions = np.array(actions)
        seq_len = len(states[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = actions[0][0].shape
        obs_dim = states[0][0].shape
        ncands = candidates.shape[1]

        true_actions = actions.reshape((new_batch_sz,) + action_dim)
        observations = states.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            subgoal = candidates[:,c]
            candidate = (subgoal + states[:, 0, :self.action_dim])[:, None] - states[:, :, :self.action_dim]
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = low_con.policy(observations, candidate)

        difference = (policy_actions - true_actions)
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).transpose(1, 0, 2, 3)

        logprob = -0.5*np.sum(np.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)

        return candidates[np.arange(batch_size), max_indices]
    
    def train(self, states, actions, n_states, rewards, not_done, states_arr, actions_arr, low_con):
        if not self._initialized:
            self._initialize_target_networks()


        actions = self.off_policy_corrections(
            low_con,
            states.shape[0],
            actions.numpy(),
            states_arr.numpy(),
            actions_arr.numpy()
        )
        return self._train(states, actions, rewards, n_states, not_done)
    
    def _train(self, states, actions, rewards, n_states, not_done):
        self.total_it += 1
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.actor.trainable_variables + self.critic1.trainable_variables + self.critic2.trainable_variables)
            noise = tf.clip_by_value(tf.random.normal(tf.shape(actions)) * self.policy_noise, -self.noise_clip, self.noise_clip)

            n_actions = self.actor_target([n_states]) + noise
            n_actions = tf.clip_by_value(n_actions, -self.actor.scale, self.actor.scale)

            target_Q1 = self.critic1_target([n_states, n_actions])
            target_Q2 = self.critic2_target([n_states, n_actions])
            target_Q = tf.math.minimum(target_Q1, target_Q2)
            target_Q_detached = (rewards + not_done * self.gamma * target_Q).numpy()

            current_Q1 = self.critic1([states, actions])
            current_Q2 = self.critic2([states, actions])

            critic1_loss = tf.reduce_mean(tf.losses.huber(current_Q1, target_Q_detached))
            critic2_loss = tf.reduce_mean(tf.losses.huber(current_Q2, target_Q_detached))
            critic_loss = critic1_loss + critic2_loss

            td_error = tf.reduce_mean(target_Q_detached - current_Q1).numpy()

            actor_loss = None
            if self.total_it % self.policy_freq == 0:
                a = self.actor([states])
                Q1 = self.critic1([states, a])
                actor_loss = -tf.reduce_mean(Q1)

        grads = tape.gradient(critic_loss, self.critic1.trainable_variables + self.critic2.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(grads[:len(self.critic1.trainable_variables)], self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(grads[len(self.critic1.trainable_variables):], self.critic2.trainable_variables))

        if actor_loss is not None:
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            self._update_target_network(self.critic1_target, self.critic1, self.tau)
            self._update_target_network(self.critic2_target, self.critic2, self.tau)
            self._update_target_network(self.actor_target, self.actor, self.tau)

            return {'actor_loss_'+self.name: actor_loss, 'critic_loss_'+self.name: critic_loss}, \
                    {'td_error_'+self.name: td_error}
    
        return {'critic_loss_'+self.name: critic_loss}, \
                    {'td_error_'+self.name: td_error}
    
class LowerController(TD3Controller):
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        scale,
        model_path,
        actor_lr=0.0001,
        critic_lr=0.001,
        expl_noise=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005):
        super(LowerController, self).__init__(
            state_dim, goal_dim, action_dim, scale, model_path,
            actor_lr, critic_lr, expl_noise, policy_noise,
            noise_clip, gamma, policy_freq, tau
        )
        self.name = 'low'

    def train(self, states, sgoals, actions, n_states, n_sgoals, rewards, not_done):
        if not self._initialized:
            self._initialize_target_networks()

        return self._train(states, sgoals, actions, rewards, n_states, n_sgoals, not_done)
   


    






