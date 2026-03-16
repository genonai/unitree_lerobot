from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize  # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_  # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
from enum import IntEnum
import threading
import time
from multiprocessing import Process, Array

import logging_mp

logger_mp = logging_mp.get_logger(__name__)

Inspire_Num_Motors = 6
kTopicInspireCommand = "rt/inspire/cmd"
kTopicInspireState = "rt/inspire/state"


class Inspire_Controller:
    def __init__(
        self,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=100.0,
        Unit_Test=False,
        simulation_mode=False,
    ):
        logger_mp.info("Initialize Inspire_Controller...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        # initialize handcmd publisher and handstate subscriber
        self.HandCmb_publisher = ChannelPublisher(kTopicInspireCommand, MotorCmds_)
        self.HandCmb_publisher.Init()

        self.HandState_subscriber = ChannelSubscriber(kTopicInspireState, MotorStates_)
        self.HandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array = Array("d", Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", Inspire_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.right_hand_state_array):  # any(self.left_hand_state_array) and
                break
            time.sleep(0.01)
            logger_mp.warning("[Inspire_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Inspire_Controller] Subscribe dds ok.")

        hand_control_process = Process(
            target=self.control_process,
            args=(
                left_hand_array,
                right_hand_array,
                self.left_hand_state_array,
                self.right_hand_state_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
            ),
        )
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Inspire_Controller OK!\n")

    def _subscribe_hand_state(self):
        while True:
            hand_msg = self.HandState_subscriber.Read()
            if hand_msg is not None:
                for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
                    self.left_hand_state_array[idx] = hand_msg.states[id].q
                for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
                    self.right_hand_state_array[idx] = hand_msg.states[id].q
            time.sleep(0.002)

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor state target q
        """
        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[id].q = left_q_target[idx]
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[id].q = right_q_target[idx]

        self.HandCmb_publisher.Write(self.hand_msg)
        # logger_mp.debug("hand ctrl publish ok.")

    def control_process(
        self,
        left_hand_array,
        right_hand_array,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
    ):
        self.running = True

        left_q_target = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        # initialize inspire hand's cmd msg
        self.hand_msg = MotorCmds_()
        self.hand_msg.cmds = [
            unitree_go_msg_dds__MotorCmd_()
            for _ in range(len(Inspire_Right_Hand_JointIndex) + len(Inspire_Left_Hand_JointIndex))
        ]

        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0

        try:
            while self.running:
                start_time = time.time()

                # get dual hand state
                with left_hand_array.get_lock():
                    left_hand_mat = np.array(left_hand_array[:]).copy()
                with right_hand_array.get_lock():
                    right_hand_mat = np.array(right_hand_array[:]).copy()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                action_data = np.concatenate((left_hand_mat, right_hand_mat))
                if dual_hand_data_lock is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        left_q_target = left_hand_mat
                        right_q_target = right_hand_mat

                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller has been closed.")


# Update hand state, according to the official documentation, https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand
# the state sequence is as shown in the table below
# ┌──────┬───────┬──────┬────────┬────────┬────────────┬────────────────┬───────┬──────┬────────┬────────┬────────────┬────────────────┐
# │ Id   │   0   │  1   │   2    │   3    │     4      │       5        │   6   │  7   │   8    │   9    │    10      │       11       │
# ├──────┼───────┼──────┼────────┼────────┼────────────┼────────────────┼───────┼──────┼────────┼────────┼────────────┼────────────────┤
# │      │                    Right Hand                                │                   Left Hand                                  │
# │Joint │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │
# └──────┴───────┴──────┴────────┴────────┴────────────┴────────────────┴───────┴──────┴────────┴────────┴────────────┴────────────────┘
class Inspire_Right_Hand_JointIndex(IntEnum):
    kRightHandPinky = 0
    kRightHandRing = 1
    kRightHandMiddle = 2
    kRightHandIndex = 3
    kRightHandThumbBend = 4
    kRightHandThumbRotation = 5


class Inspire_Left_Hand_JointIndex(IntEnum):
    kLeftHandPinky = 6
    kLeftHandRing = 7
    kLeftHandMiddle = 8
    kLeftHandIndex = 9
    kLeftHandThumbBend = 10
    kLeftHandThumbRotation = 11


class Inspire_1DOF_Controller:
    """
    Eval-side 1DOF grip controller for Inspire RH56DFQ hand.
    Takes normalized [0,1] grip value from policy output, expands to 6 DOF.
    DFX protocol only (matching existing eval-side Inspire_Controller).
    """

    def __init__(self, left_gripper_value_in, right_gripper_value_in,
                 dual_gripper_data_lock=None, dual_gripper_state_out=None,
                 dual_gripper_action_out=None, simulation_mode=False,
                 fps=100.0):
        logger_mp.info("Initialize Inspire_1DOF_Controller (eval)...")
        self.fps = fps
        self.simulation_mode = simulation_mode
        self.sub_ready = False

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        self.HandCmd_publisher = ChannelPublisher(kTopicInspireCommand, MotorCmds_)
        self.HandCmd_publisher.Init()
        self.HandState_subscriber = ChannelSubscriber(kTopicInspireState, MotorStates_)
        self.HandState_subscriber.Init()

        self.left_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)

        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while not self.sub_ready:
            time.sleep(0.01)
            logger_mp.warning("[Inspire_1DOF_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Inspire_1DOF_Controller] Subscribe dds ok.")

        self.control_thread = threading.Thread(
            target=self._control_loop,
            args=(left_gripper_value_in, right_gripper_value_in,
                  self.left_hand_state_array, self.right_hand_state_array,
                  dual_gripper_data_lock, dual_gripper_state_out, dual_gripper_action_out))
        self.control_thread.daemon = True
        self.control_thread.start()

        logger_mp.info("Initialize Inspire_1DOF_Controller (eval) OK!")

    def _subscribe_hand_state(self):
        while True:
            hand_msg = self.HandState_subscriber.Read()
            if hand_msg is not None:
                self.sub_ready = True
                for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
                    self.left_hand_state_array[idx] = hand_msg.states[id].q
                for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
                    self.right_hand_state_array[idx] = hand_msg.states[id].q
            time.sleep(0.002)

    def _control_loop(self, left_gripper_value_in, right_gripper_value_in,
                      left_hand_state_array, right_hand_state_array,
                      dual_gripper_data_lock=None, dual_gripper_state_out=None,
                      dual_gripper_action_out=None):
        self.running = True

        self.hand_msg = MotorCmds_()
        self.hand_msg.cmds = [
            unitree_go_msg_dds__MotorCmd_()
            for _ in range(len(Inspire_Right_Hand_JointIndex) + len(Inspire_Left_Hand_JointIndex))
        ]
        # Initialize open
        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0

        try:
            while self.running:
                start_time = time.time()

                # Read normalized [0,1] grip from policy (no trigger normalization needed)
                left_grip = np.clip(left_gripper_value_in.value, 0.0, 1.0)
                right_grip = np.clip(right_gripper_value_in.value, 0.0, 1.0)

                # Expand to 6 DOF
                for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
                    self.hand_msg.cmds[id].q = left_grip
                for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
                    self.hand_msg.cmds[id].q = right_grip
                self.HandCmd_publisher.Write(self.hand_msg)

                # State: average 6D → 1D
                left_state_avg = np.mean(np.array(left_hand_state_array[:]))
                right_state_avg = np.mean(np.array(right_hand_state_array[:]))

                if dual_gripper_data_lock is not None and dual_gripper_state_out is not None and dual_gripper_action_out is not None:
                    with dual_gripper_data_lock:
                        dual_gripper_state_out[0] = left_state_avg
                        dual_gripper_state_out[1] = right_state_avg
                        dual_gripper_action_out[0] = left_grip
                        dual_gripper_action_out[1] = right_grip

                elapsed = time.time() - start_time
                time.sleep(max(0, (1 / self.fps) - elapsed))
        finally:
            logger_mp.info("Inspire_1DOF_Controller (eval) has been closed.")
