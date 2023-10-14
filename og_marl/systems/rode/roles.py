import tensorflow as tf
import sonnet as snt

class DotRole(snt.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_actions = args.n_actions

        self.q_fc = snt.Linear(args.action_latent_dim)
        self.action_space = tf.ones(args.n_actions, dtype=tf.float32)

    def __call__(self, h, action_latent):
        role_key = self.q_fc(h)  # [bs, action_latent] [n_actions, action_latent]
        role_key = tf.expand_dims(role_key, axis=-1)

        action_latent_reshaped = tf.tile(tf.expand_dims(action_latent, axis=0), [role_key.shape[0], 1, 1])
        q = tf.squeeze(tf.matmul(action_latent_reshaped, role_key), axis=-1)

        return q

    def update_action_space(self, new_action_space):
        self.action_space = tf.constant(new_action_space, dtype=tf.float32)


class QRole(snt.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_actions = args.n_actions

        self.q_fc = snt.Linear(args.action_latent_dim)
        self.action_space = tf.ones(args.n_actions, dtype=tf.float32)

    def __call__(self, h, action_latent):
        q = self.q_fc(h)
        return q

    def update_action_space(self, new_action_space):
        self.action_space = tf.constant(new_action_space, dtype=tf.float32)