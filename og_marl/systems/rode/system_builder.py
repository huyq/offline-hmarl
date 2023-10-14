import copy
import tensorflow as tf
import sonnet as snt
from mava.utils.loggers import logger_utils
import functools
import launchpad as lp

from og_marl.systems.rode.trainer import RODETrainer
from og_marl.offline_tools.offline_dataset import MAOfflineDataset
from og_marl.systems.rode.action_encoder import ObsRewardEncoder
from og_marl.systems.iql import IQLSystemBuilder
from og_marl.utils.executor_utils import concat_agent_id_to_obs

class RODESystemBuilder(IQLSystemBuilder):
    def __init__(
        self,
        environment_factory,
        logger_factory,
        q_network,
        mixer,
        batch_size=64,
        min_replay_size=64,
        max_replay_size=5000,  # num episodes in buffer
        sequence_length=20,
        period=10,
        samples_per_insert=None,
        eps_start=1.0,
        eps_min=0.05,
        eps_dec=1e-5,
        variable_update_period=3,  # Update varibles every 3 episodes
        max_gradient_norm=20.0,
        discount=0.99,
        lambda_=0.6,
        target_update_rate=0.01,
        optimizer=snt.optimizers.Adam(learning_rate=1e-4),
        offline_environment_logging=False,
        trajectories_per_file=100,
        add_agent_id_to_obs=False,
        offline_env_log_dir=None,
        record_evaluator_every=None,
        record_executor_every=None,
        evaluation_period=100,  # ~ every 100 trainer steps
        evaluation_episodes=32,
        max_trainer_steps=1e6,
        checkpoint_subpath="",
        must_checkpoint=False,
    ):
        super().__init__(
            environment_factory,
            logger_factory,
            q_network,
            optimizer=optimizer,
            max_gradient_norm=max_gradient_norm,
            discount=discount,
            variable_update_period=variable_update_period,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            max_replay_size=max_replay_size,
            sequence_length=sequence_length,
            period=period,
            samples_per_insert=samples_per_insert,
            eps_start=eps_start,
            eps_min=eps_min,
            eps_dec=eps_dec,
            lambda_=lambda_,
            offline_environment_logging=offline_environment_logging,
            trajectories_per_file=trajectories_per_file,
            add_agent_id_to_obs=add_agent_id_to_obs,
            offline_env_log_dir=offline_env_log_dir,
            record_evaluator_every=record_evaluator_every,
            record_executor_every=record_executor_every,
            evaluation_period=evaluation_period,
            evaluation_episodes=evaluation_episodes,
            target_update_rate=target_update_rate,
            max_trainer_steps=max_trainer_steps,
            must_checkpoint=must_checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )


        self._action_encoder_steps = 1e4
        self._mixer = mixer
        self._trainer_fn = RODETrainer

        
    def _build_trainer(self, dataset, logger):

        # Initialise networks
        networks = self._initialise_networks()
        q_network = networks["q_network"]
        action_encoder = networks["action_encoder"]

        trainer = self._trainer_fn(
            agents=self._agents,
            q_network=q_network,
            mixer=self._mixer,
            action_encoder=action_encoder,
            optimizer=self._optimizer,
            discount=self._discount,
            target_update_rate=self._target_update_rate,
            lambda_=self._lambda,
            dataset=dataset,
            max_gradient_norm=self._max_gradient_norm,
            logger=logger,
            max_trainer_steps=self._max_trainer_steps,
            add_agent_id_to_obs=self._add_agent_id_to_obs,
        )

        return trainer


    def _initialise_networks(self):
        q_network = copy.deepcopy(self._q_network)

        spec = list(self._environment_spec.get_agent_specs().values())[0]
        dummy_observation = tf.zeros_like(spec.observations.observation)

        if self._add_agent_id_to_obs:
            dummy_observation = concat_agent_id_to_obs(
                dummy_observation, 1, len(self._agents)
            )

        dummy_observation = tf.expand_dims(
            dummy_observation, axis=0
        )  # add dummy batch dim

        # Initialise q-network
        dummy_core_state = q_network.initial_state(1)  # Dummy recurent core state
        q_network(dummy_observation, dummy_core_state)
        obs_dim = spec.observations.observation.shape[-1]

        action_encoder = ObsRewardEncoder(
            n_agents=len(self._agents),
            n_actions=spec.actions.num_values,
            obs_dim=spec.observations.observation.shape[-1]
        )

        return {"q_network": q_network, "action_encoder":action_encoder}


    def run_offline(
        self,
        dataset_dir,
        shuffle_buffer_size=5000
    ):
        # Create logger
        logger = self._logger_factory("trainer")

        # Create environment for the offline dataset
        environment = self._environment_factory()

        # Build offline dataset
        dataset = MAOfflineDataset(
            environment=environment,
            logdir=dataset_dir,
            batch_size=self._batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
        )

        trainer = self._build_trainer(dataset, logger)

        evaluator = self.evaluator(trainer)

        trainer_steps = 0
        ## train action encoder
        while trainer_steps < self._action_encoder_steps:
            if self._trainer_step_counter >= self._action_encoder_steps:
                lp.stop()

            # Increment trainer step counter
            self._trainer_step_counter += 1
            
            # Sample dataset
            sample = next(self._dataset_iter)

            # Pass sample to _train method
            logs = self._train_action_encoder(
                sample, trainer_step=tf.convert_to_tensor(self._trainer_step_counter)
            )

            # Write logs
            self._logger.write(logs)

            trainer_steps += 1


        ## train policy network
        while trainer_steps-self._action_encoder_steps < self._max_trainer_steps:

            trainer_logs = trainer.step()  # logging done in trainer

            if trainer_steps % self._evaluation_period == 0:
                evaluator_logs = evaluator.run_evaluation(
                    trainer_steps, self._evaluation_episodes
                )  # logging done in evaluator

            trainer_steps += 1

        # Final evaluation
        evaluator_logs = evaluator.run_evaluation(
            trainer_steps,
            10 * self._evaluation_episodes,
            use_best_checkpoint=True,
        )
