import tensorflow as tf
import sonnet as snt

class DotSelector(snt.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.epsilon_start = self.args.epsilon_start
        self.epsilon_finish = self.args.role_epsilon_finish
        self.epsilon_anneal_time = self.args.epsilon_anneal_time
        self.epsilon_anneal_time_exp = self.args.epsilon_anneal_time_exp
        self.delta = (self.epsilon_start - self.epsilon_finish) / self.epsilon_anneal_time
        self.role_action_spaces_update_start = self.args.role_action_spaces_update_start
        self.epsilon_start_t = 0
        self.epsilon_reset = True

        self.fc1 = snt.Linear(2 * args.rnn_hidden_dim)
        self.fc2 = snt.Linear(args.action_latent_dim)

        self.epsilon = 0.05

    def __call__(self, inputs, role_latent):
        x = self.fc2(tf.nn.relu(self.fc1(inputs)))  # [bs, action_dim] [n_roles, action_dim] (bs may be bs*n_agents)
        x = tf.expand_dims(x, -1)
        role_latent_reshaped = tf.tile(tf.expand_dims(role_latent, 0), [x.shape[0], 1, 1])

        role_q = tf.squeeze(tf.matmul(role_latent_reshaped, x), -1)

        return role_q

    def select_role(self, role_qs, test_mode=False, t_env=None):
        self.epsilon = self.epsilon_schedule(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = tf.identity(role_qs).numpy()

        random_numbers = tf.random.uniform_like(role_qs[:, 0], 0, 1)
        pick_random = tf.cast(random_numbers < self.epsilon, tf.int64)
        random_roles = tf.random.categorical(tf.ones(role_qs.shape), 1)[:, 0]

        picked_roles = tf.where(pick_random, random_roles, tf.argmax(masked_q_values, axis=1))

        # [bs, 1]
        return picked_roles

    def epsilon_schedule(self, t_env):
        if t_env is None:
            return 0.05

        if t_env > self.role_action_spaces_update_start and self.epsilon_reset:
            self.epsilon_reset = False
            self.epsilon_start_t = t_env
            self.epsilon_anneal_time = self.epsilon_anneal_time_exp
            self.delta = (self.epsilon_start - self.epsilon_finish) / self.epsilon_anneal_time

        if t_env - self.epsilon_start_t > self.epsilon_anneal_time:
            epsilon = self.epsilon_finish
        else:
            epsilon = self.epsilon_start - (t_env - self.epsilon_start_t) * self.delta

        return epsilon