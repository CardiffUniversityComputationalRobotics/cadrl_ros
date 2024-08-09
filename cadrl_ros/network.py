import os
import re
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import layers


class Actions:
    # Define 11 choices of actions to be:
    # [v_pref,      [-pi/6, -pi/12, 0, pi/12, pi/6]]
    # [0.5*v_pref,  [-pi/6, 0, pi/6]]
    # [0,           [-pi/6, 0, pi/6]]
    def __init__(self):
        self.actions = (
            np.mgrid[1.0:1.1:0.5, -np.pi / 6 : np.pi / 6 + 0.01 : np.pi / 12]
            .reshape(2, -1)
            .T
        )
        self.actions = np.vstack(
            [
                self.actions,
                np.mgrid[0.5:0.6:0.5, -np.pi / 6 : np.pi / 6 + 0.01 : np.pi / 6]
                .reshape(2, -1)
                .T,
            ]
        )
        self.actions = np.vstack(
            [
                self.actions,
                np.mgrid[0.0:0.1:0.5, -np.pi / 6 : np.pi / 6 + 0.01 : np.pi / 6]
                .reshape(2, -1)
                .T,
            ]
        )
        self.num_actions = len(self.actions)


class NetworkVPCore(tf.Module):
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions
        self.build_model()

    def build_model(self):
        self.x = tf.keras.Input(
            shape=(Config.NN_INPUT_SIZE,), dtype=tf.float32, name="X"
        )

        # Define layers
        self.fc1 = layers.Dense(256, activation="relu", name="fullyconnected1")(self.x)

        self.logits_p = layers.Dense(
            self.num_actions, activation=None, name="logits_p"
        )(self.fc1)
        softmax_layer = layers.Softmax()
        self.softmax_p = (softmax_layer(self.logits_p) + Config.MIN_POLICY) / (
            1.0 + Config.MIN_POLICY * self.num_actions
        )

        self.logits_v = layers.Dense(1, activation=None, name="logits_v")(self.fc1)

        self.logits_v = layers.Lambda(lambda x: tf.squeeze(x, axis=1))(self.logits_v)

        # Create the model
        self.model = tf.keras.Model(
            inputs=self.x, outputs=[self.softmax_p, self.logits_v]
        )

    def predict_p(self, x):
        # Forward pass through the model to get predictions
        return self.model(x)[0]

    def predict_v(self, x):
        return self.model(x)[1]

    def simple_load(self, checkpoint_dir):
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()


class NetworkVP_rnn(NetworkVPCore):
    def __init__(self, device, model_name, num_actions):
        super().__init__(device, model_name, num_actions)

    def build_model(self):
        super().build_model()

        if Config.USE_REGULARIZATION:
            regularizer = tf.keras.regularizers.l2(0.0)
        else:
            regularizer = None

        if Config.NORMALIZE_INPUT:
            self.avg_vec = tf.constant(Config.NN_INPUT_AVG_VECTOR, dtype=tf.float32)
            self.std_vec = tf.constant(Config.NN_INPUT_STD_VECTOR, dtype=tf.float32)
            self.x_normalized = (self.x - self.avg_vec) / self.std_vec
        else:
            self.x_normalized = self.x

        if Config.MULTI_AGENT_ARCH == "RNN":
            num_hidden = 64
            max_length = Config.MAX_NUM_OTHER_AGENTS_OBSERVED
            self.num_other_agents = layers.Lambda(lambda x: x[:, 0])(self.x)
            self.host_agent_vec = layers.Lambda(
                lambda x: x[
                    :,
                    Config.FIRST_STATE_INDEX : Config.HOST_AGENT_STATE_SIZE
                    + Config.FIRST_STATE_INDEX,
                ]
            )(self.x_normalized)
            self.other_agent_vec = layers.Lambda(
                lambda x: x[
                    :, Config.HOST_AGENT_STATE_SIZE + Config.FIRST_STATE_INDEX :
                ]
            )(self.x_normalized)
            self.other_agent_seq = layers.Lambda(
                lambda x: tf.reshape(
                    x, [-1, max_length, Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH]
                )
            )(self.other_agent_vec)

            # Use a Lambda layer to apply sequence masking within Keras
            mask = layers.Lambda(lambda x: tf.sequence_mask(x, maxlen=max_length))(
                self.num_other_agents
            )

            rnn_layer = tf.keras.layers.LSTM(num_hidden, return_state=True)
            self.rnn_state = rnn_layer(self.other_agent_seq, mask=mask)
            self.rnn_output = self.rnn_state[0]  # `h` state of LSTM
            # Replace the `tf.concat` with a Keras Lambda layer
            self.layer1_input = layers.Lambda(lambda x: tf.concat(x, axis=1))(
                [self.host_agent_vec, self.rnn_output]
            )
            self.layer1 = tf.keras.layers.Dense(
                256,
                activation=tf.nn.relu,
                kernel_regularizer=regularizer,
                name="layer1",
            )(self.layer1_input)

        self.layer2 = tf.keras.layers.Dense(256, activation=tf.nn.relu, name="layer2")(
            self.layer1
        )
        self.final_flat = tf.keras.layers.Flatten()(self.layer2)

        super().build_model()


class Config:
    #########################################################################
    # GENERAL PARAMETERS
    NORMALIZE_INPUT = True
    USE_DROPOUT = False
    USE_REGULARIZATION = True
    ROBOT_MODE = True
    EVALUATE_MODE = True

    SENSING_HORIZON = 8.0

    MIN_POLICY = 1e-4

    MAX_NUM_AGENTS_IN_ENVIRONMENT = 20
    MULTI_AGENT_ARCH = "RNN"

    DEVICE = "/cpu:0"  # Device

    HOST_AGENT_OBSERVATION_LENGTH = (
        4  # dist to goal, heading to goal, pref speed, radius
    )
    OTHER_AGENT_OBSERVATION_LENGTH = 7  # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_LENGTH = 1  # num other agents
    AGENT_ID_LENGTH = 1  # id
    IS_ON_LENGTH = 1  # 0/1 binary flag

    HOST_AGENT_AVG_VECTOR = np.array(
        [0.0, 0.0, 1.0, 0.5]
    )  # dist to goal, heading to goal, pref speed, radius
    HOST_AGENT_STD_VECTOR = np.array(
        [5.0, 3.14, 1.0, 1.0]
    )  # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_AVG_VECTOR = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0]
    )  # other px, other py, other vx, other vy, other radius, combined radius, distance between
    OTHER_AGENT_STD_VECTOR = np.array(
        [5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0]
    )  # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_AVG_VECTOR = np.array([0.0])
    RNN_HELPER_STD_VECTOR = np.array([1.0])
    IS_ON_AVG_VECTOR = np.array([0.0])
    IS_ON_STD_VECTOR = np.array([1.0])

    if MAX_NUM_AGENTS_IN_ENVIRONMENT > 2:
        if MULTI_AGENT_ARCH == "RNN":
            # NN input:
            # [num other agents, dist to goal, heading to goal, pref speed, radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius]
            MAX_NUM_OTHER_AGENTS_OBSERVED = 10
            OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH
            HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH
            FULL_STATE_LENGTH = (
                RNN_HELPER_LENGTH
                + HOST_AGENT_OBSERVATION_LENGTH
                + MAX_NUM_OTHER_AGENTS_OBSERVED * OTHER_AGENT_FULL_OBSERVATION_LENGTH
            )
            FIRST_STATE_INDEX = 1

            NN_INPUT_AVG_VECTOR = np.hstack(
                [
                    RNN_HELPER_AVG_VECTOR,
                    HOST_AGENT_AVG_VECTOR,
                    np.tile(OTHER_AGENT_AVG_VECTOR, MAX_NUM_OTHER_AGENTS_OBSERVED),
                ]
            )
            NN_INPUT_STD_VECTOR = np.hstack(
                [
                    RNN_HELPER_STD_VECTOR,
                    HOST_AGENT_STD_VECTOR,
                    np.tile(OTHER_AGENT_STD_VECTOR, MAX_NUM_OTHER_AGENTS_OBSERVED),
                ]
            )

    FULL_LABELED_STATE_LENGTH = FULL_STATE_LENGTH + AGENT_ID_LENGTH
    NN_INPUT_SIZE = FULL_STATE_LENGTH


if __name__ == "__main__":
    actions = Actions().actions
    num_actions = Actions().num_actions
    nn = NetworkVP_rnn(Config.DEVICE, "network", num_actions)
    nn.simple_load("path_to_checkpoints")

    obs = np.zeros((Config.FULL_STATE_LENGTH))
    obs = np.expand_dims(obs, axis=0)

    num_queries = 10000
    t_start = time.time()
    for i in range(num_queries):
        obs[0, 0] = 10  # num other agents
        obs[0, 1] = np.random.uniform(0.5, 10.0)  # dist to goal
        obs[0, 2] = np.random.uniform(-np.pi, np.pi)  # heading to goal
        obs[0, 3] = np.random.uniform(0.2, 2.0)  # pref speed
        obs[0, 4] = np.random.uniform(0.2, 1.5)  # radius
        predictions = nn.predict_p(obs)[0]
    t_end = time.time()
    print("avg query time:", (t_end - t_start) / num_queries)
    print("total time:", t_end - t_start)
    # action = actions[np.argmax(predictions)]
    # print "action:", action