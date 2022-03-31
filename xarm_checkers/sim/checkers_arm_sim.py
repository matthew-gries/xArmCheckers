from typing import Tuple, List, Optional, Dict
import time
import numpy as np
import pybullet as pb
import pybullet_data

from xarm_checkers.sim.checkers_board import add_checkerboard
from checkers.game import Game

class RobotArm:
    GRIPPER_CLOSED = 0.
    GRIPPER_OPENED = 1.
    def __init__(self):
        '''Robot Arm simulated in Pybullet, with support for performing top-down
        grasps within a specified workspace
        '''
        # placing robot higher above ground improves top-down grasping ability
        self._id = pb.loadURDF("assets/urdf/xarm.urdf",
                               basePosition=(0, 0, 0.05),
                               flags=pb.URDF_USE_SELF_COLLISION)

        # these are hard coded based on how urdf is written
        self.arm_joint_ids = [1,2,3,4,5]
        self.gripper_joint_ids = [6,7]
        self.dummy_joint_ids = [8]
        self.finger_joint_ids = [9,10]
        self.end_effector_link_index = 11

        self.arm_joint_limits = np.array(((-2, -1.58, -2, -1.8, -2),
                                          ( 2,  1.58,  2,  2.0,  2)))
        # self.gripper_joint_limits = np.array(((0.05,0.05),
        #                                       (1.38, 1.38)))
        # Don't open the gripper as much
        self.gripper_joint_limits = np.array(((0.075,0.075),
                                              (0.25, 0.25)))

        # chosen to move arm out of view of camera
        self.home_arm_jpos = [0., -1.1, 1.4, 1.3, 0.]

        # joint constraints are needed for four-bar linkage in xarm fingers
        for i in [0,1]:
            constraint = pb.createConstraint(self._id,
                                             self.gripper_joint_ids[i],
                                             self._id,
                                             self.finger_joint_ids[i],
                                             pb.JOINT_POINT2POINT,
                                             (0,0,0),
                                             (0,0,0.03),
                                             (0,0,0))
            pb.changeConstraint(constraint, maxForce=1000000)

        # reset joints in hand so that constraints are satisfied
        hand_joint_ids = self.gripper_joint_ids + self.dummy_joint_ids + self.finger_joint_ids
        hand_rest_states = [0.05, 0.05, 0.055, 0.0155, 0.031]
        [pb.resetJointState(self._id, j_id, jpos)
                 for j_id,jpos in zip(hand_joint_ids, hand_rest_states)]

        # allow finger and linkages to move freely
        pb.setJointMotorControlArray(self._id,
                                     self.dummy_joint_ids+self.finger_joint_ids,
                                     pb.POSITION_CONTROL,
                                     forces=[0,0,0])

    def move_gripper_to(self, position: List[float], theta: float):
        '''Commands motors to move end effector to desired position, oriented
        downwards with a rotation of theta about z-axis

        Parameters
        ----------
        position
            xyz position that end effector should move toward
        theta
            rotation (in radians) of the gripper about the z-axis.

        Returns
        -------
        bool
            True if movement is successful, False otherwise.
        '''
        quat = pb.getQuaternionFromEuler((0,-np.pi,theta))
        arm_jpos, _ = self.solve_ik(position, quat)

        return self.move_arm_to_jpos(arm_jpos)

    def solve_ik(self,
                 pos: List[float],
                 quat: Optional[List[float]]=None,
                ) -> Tuple[List[float], Dict[str, float]]:
        '''Calculates inverse kinematics solution for a desired end effector
        position and (optionally) orientation, and returns residuals

        Hint
        ----
        To calculate residuals, you can get the pose of the end effector link using
        `pybullet.getLinkState` (but you need to set the arm joint positions first)

        Parameters
        ----------
        pos
            target xyz position of end effector
        quat
            target orientation of end effector as unit quaternion if specified.
            otherwise, ik solution ignores final orientation

        Returns
        -------
        list
            joint positions of arm that would result in desired end effector
            position and orientation. in order from base to wrist
        dict
            position and orientation residuals:
                {'position' : || pos - achieved_pos ||,
                 'orientation' : 1 - |<quat, achieved_quat>|}
        '''
        n_joints = pb.getNumJoints(self._id)
        all_jpos = pb.calculateInverseKinematics(self._id,
                                                 self.end_effector_link_index,
                                                 pos,
                                                 quat,
                                                 maxNumIterations=20,
                                                 jointDamping=n_joints*[0.005])
        arm_jpos = all_jpos[:len(self.arm_joint_ids)]

        # teleport arm to check acheived pos and orientation
        old_arm_jpos = list(zip(*pb.getJointStates(self._id, self.arm_joint_ids)))[0]
        [pb.resetJointState(self._id, i, jp) for i,jp in zip(self.arm_joint_ids, arm_jpos)]
        achieved_pos, achieved_quat = pb.getLinkState(self._id, self.end_effector_link_index)[:2]
        [pb.resetJointState(self._id, i, jp) for i,jp in zip(self.arm_joint_ids, old_arm_jpos)]

        residuals = {'position' : np.linalg.norm(np.subtract(pos, achieved_pos)),
                     'orientation' : 1 - np.abs(np.dot(quat, achieved_quat))}

        return arm_jpos, residuals

    def move_arm_to_jpos(self, arm_jpos: List[float]) -> bool:
        '''Commands motors to move arm to desired joint positions

        Parameters
        ----------
        arm_jpos
            joint positions (radians) of arm joints, ordered from base to wrist

        Returns
        -------
        bool
            True if movement is successful, False otherwise.
        '''
        # cannot use setJointMotorControlArray because API does not expose
        # maxVelocity argument, which is needed for stable object manipulation
        for j_id, jpos in zip(self.arm_joint_ids, arm_jpos):
            pb.setJointMotorControl2(self._id,
                                     j_id,
                                     pb.POSITION_CONTROL,
                                     jpos,
                                     positionGain=0.2,
                                     maxVelocity=0.8)

        return self.monitor_movement(arm_jpos, self.arm_joint_ids)

    def set_gripper_state(self, gripper_state: float) -> bool:
        '''Commands motors to move gripper to given state

        Parameters
        ----------
        gripper_state
            gripper state is a continuous number from 0. (fully closed)
            to 1. (fully open)

        Returns
        -------
        bool
            True if movement is successful, False otherwise.

        Raises
        ------
        AssertionError
            If `gripper_state` is outside the range [0,1]
        '''
        assert 0 <= gripper_state <= 1, 'Gripper state must be in range [0,1]'

        gripper_jpos = (1-gripper_state)*self.gripper_joint_limits[0] \
                       + gripper_state*self.gripper_joint_limits[1]

        pb.setJointMotorControlArray(self._id,
                                     self.gripper_joint_ids,
                                     pb.POSITION_CONTROL,
                                     gripper_jpos,
                                     positionGains=[0.2, 0.2])

        success = self.monitor_movement(gripper_jpos, self.gripper_joint_ids)
        return success

    def monitor_movement(self,
                         target_jpos: List[float],
                         joint_ids: List[int],
                        ) -> bool:
        '''Monitors movement of motors to detect early stoppage or success.

        Note
        ----
        Current implementation calls `pybullet.stepSimulation`, without which the
        simulator will not move the motors.  You can avoid this by setting
        `pybullet.setRealTimeSimulation(True)` but this is usually not advised.

        Parameters
        ----------
        target_jpos
            final joint positions that motors are moving toward
        joint_ids
            the joint ids associated with each `target_jpos`, used to read out
            the joint state during movement

        Returns
        -------
        bool
            True if movement is successful, False otherwise.
        '''
        old_jpos = list(zip(*pb.getJointStates(self._id, joint_ids)))[0]
        while True:
            [pb.stepSimulation() for _ in range(10)]

            time.sleep(0.01)

            achieved_jpos = list(zip(*pb.getJointStates(self._id, joint_ids)))[0]
            if np.allclose(target_jpos, achieved_jpos, atol=1e-3):
                # success
                return True

            if np.allclose(achieved_jpos, old_jpos, atol=1e-3):
                # movement stopped
                return False
            old_jpos = achieved_jpos


class Camera:
    def __init__(self, workspace: np.ndarray) -> None:
        '''Camera that is mounted to view workspace from above

        Hint
        ----
        For this camera setup, it may be easiest if you use the functions
        `pybullet.computeViewMatrix` and `pybullet.computeProjectionMatrixFOV`.
        cameraUpVector should be (0,1,0)

        Parameters
        ----------
        workspace
            2d array describing extents of robot workspace that is to be viewed,
            in the format: ((min_x,min_y), (max_x, max_y))

        Attributes
        ----------
        img_width : int
            width of rendered image
        img_height : int
            height of rendered image
        view_mtx : List[float]
            view matrix that is positioned to view center of workspace from above
        proj_mtx : List[float]
            proj matrix that set up to fully view workspace
        '''
        self.img_width = 100
        self.img_height = 100

        cx, cy = np.mean(workspace, axis=0)
        eye_pos = (cx, cy, 0.25)
        target_pos = (cx, cy, 0)
        self.view_mtx = pb.computeViewMatrix(cameraEyePosition=eye_pos,
                                             cameraTargetPosition=target_pos,
                                            cameraUpVector=(0,1,0))
        self.proj_mtx = pb.computeProjectionMatrixFOV(fov=25,
                                                      aspect=1,
                                                      nearVal=0.01,
                                                      farVal=1)

    def get_rgb_image(self) -> np.ndarray:
        '''Takes rgb image

        Returns
        -------
        np.ndarray
            shape (H,W,3) with dtype=np.uint8
        '''
        rgba = pb.getCameraImage(width=self.img_width,
                                 height=self.img_height,
                                 viewMatrix=self.view_mtx,
                                 projectionMatrix=self.proj_mtx,
                                 renderer=pb.ER_TINY_RENDERER)[2]

        return rgba[...,:3]


class TopDownGraspingEnv:
    def __init__(self, render: bool=True) -> None:
        '''Pybullet simulator with robot that performs top down grasps of a
        single object.  A camera is positioned to take images of workspace
        from above.
        '''
        self.client = pb.connect(pb.GUI if render else pb.DIRECT)
        pb.setPhysicsEngineParameter(numSubSteps=0,
                                     numSolverIterations=100,
                                     solverResidualThreshold=1e-7,
                                     constraintSolverType=pb.CONSTRAINT_SOLVER_LCP_SI)
        pb.setGravity(0,0,-10)

        # create ground plane
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        # offset plane y-dim to place white tile under workspace
        self.plane_id = pb.loadURDF('plane.urdf', (0,-0.5,0))

        # makes collisions with plane more stable
        pb.changeDynamics(self.plane_id, -1,
                          linearDamping=0.04,
                          angularDamping=0.04,
                          restitution=0,
                          contactStiffness=3000,
                          contactDamping=100)

        # add robot
        self.robot = RobotArm()

        # add board
        self.board_id = add_checkerboard()
        self.board_height = 0.01
        pb.changeDynamics(self.board_id, -1,
                          lateralFriction=1,
                          spinningFriction=0.005,
                          rollingFriction=0.005)


        self.workspace = np.array(((0.10, -0.05), # ((min_x, min_y)
                                   (0.20, 0.05))) #  (max_x, max_y))
        self.board_dim = self.workspace[1][0] - self.workspace[0][0]

        # maps row, col to x, y positions on board
        self.board_positions = np.zeros((8, 8, 2))
        for i in range(8):
            for j in range(8):
                self.board_positions[i][j][0] = self.workspace[0][0] + (2*i + 1)*(self.board_dim/16)
                self.board_positions[i][j][1] = self.workspace[0][1] + (2*j + 1)*(self.board_dim/16)

        self.white_pieces_ids = []
        self.yellow_pieces_ids = []
        self.piece_height = 0.01
        self.grasp_height = self.board_height + self.piece_height + 0.005

        for _ in range(12):
            white_id = pb.loadURDF("assets/urdf/white_piece.urdf")
            pb.changeDynamics(white_id, -1,
                lateralFriction=1,
                spinningFriction=0.005,
                rollingFriction=0.005)
            yellow_id = pb.loadURDF("assets/urdf/yellow_piece.urdf")
            pb.changeDynamics(yellow_id, -1,
                lateralFriction=1,
                spinningFriction=0.005,
                rollingFriction=0.005)

            self.white_pieces_ids.append(white_id)
            self.yellow_pieces_ids.append(yellow_id)

        # Keeps track of the positions on the checkers board as defined by the `Checkers` class implementation
        self.yellow_checkers_board_spots = set()
        self.white_checkers_board_spots = set()
        # mapping between positions on the checkers board as defined by the `Checkers` class implementation
        # and (row, column) board positions
        # 
        # For example, player 1's (yellow) first piece is at position 1, which maps to (0, 1) in (row, col)
        # The call to `set_pieces` will initialize these
        self.board_spot_to_row_col = {}

        if render:
            self.draw_workspace()

        # add camera
        self.camera = Camera(self.workspace)

        # checkers implementation
        self.checkers = Game()

    def draw_workspace(self) -> None:
        '''This is just for visualization purposes, to help you with the object
        resetting.  Must be in GUI mode, otherwise error occurs

        Note
        ----
        Pybullet debug lines only show up in GUI mode so they won't help you
        with camera placement.
        '''
        corner_ids = ((0,0), (0,1), (1,1), (1,0), (0,0))
        for i in range(4):
            start = (*self.workspace[corner_ids[i],[0,1]], 0.)
            end = (*self.workspace[corner_ids[i+1],[0,1]], 0.)
            pb.addUserDebugLine(start, end, (0,0,0), 3)

    def perform_grasp(self, x, y, theta) -> bool:
        '''Perform top down grasp in the workspace.  All grasps will occur
        at a height of the center of mass of the object (i.e. object_width/2)

        Parameters
        ----------
        x
            x position of the grasp in world frame
        y
            y position of the grasp in world frame
        theta
            target rotation about z-axis of gripper during grasp

        Returns
        -------
        bool
            True if object was successfully grasped, False otherwise. It is up
            to you to decide how to determine success
        '''
        self.robot.move_arm_to_jpos(self.robot.home_arm_jpos)
        self.robot.set_gripper_state(self.robot.GRIPPER_OPENED)
        
        # add a little bit of offset
        grasp_height_offset = self.grasp_height + 0.05

        # manually move the gripper above the spot so we can move it directly down
        arm_pos, _ = self.robot.solve_ik(
            pos=[x, y, grasp_height_offset],
            quat=pb.getQuaternionFromEuler((0,-np.pi,theta))
        )
        self.robot.move_arm_to_jpos(arm_jpos=arm_pos)

        pos = np.array((x, y, self.grasp_height))
        self.robot.move_gripper_to(pos, theta)
        self.robot.set_gripper_state(self.robot.GRIPPER_CLOSED)

        # manually move the gripper up without moving it to the sides
        arm_pos, _ = self.robot.solve_ik(
            pos=[x, y, grasp_height_offset],
            quat=pb.getQuaternionFromEuler((0,-np.pi,theta))
        )
        self.robot.move_arm_to_jpos(arm_jpos=arm_pos)
        self.robot.move_arm_to_jpos(self.robot.home_arm_jpos)

        # TODO figure out how to check for success
        # check if object is above plane
        # min_object_height = 0.05
        # obj_height = pb.getBasePositionAndOrientation(self.object_id)[0][2]
        # success = obj_height > min_object_height

        return True

    def place_piece(self, x, y, theta) -> bool:

        # add a little bit of offset
        grasp_height_offset = self.grasp_height + 0.05

        self.robot.move_arm_to_jpos(self.robot.home_arm_jpos)

        # manually move the gripper above the spot so we can move it directly down
        arm_pos, _ = self.robot.solve_ik(
            pos=[x, y, grasp_height_offset],
            quat=pb.getQuaternionFromEuler((0,-np.pi,theta))
        )
        self.robot.move_arm_to_jpos(arm_jpos=arm_pos)

        self.robot.move_gripper_to([x, y, self.grasp_height], theta)
        self.robot.set_gripper_state(self.robot.GRIPPER_OPENED)

        # manually move the gripper up without moving it to the sides
        arm_pos, _ = self.robot.solve_ik(
            pos=[x, y, grasp_height_offset],
            quat=pb.getQuaternionFromEuler((0,-np.pi,theta))
        )
        self.robot.move_arm_to_jpos(arm_jpos=arm_pos)

        self.robot.move_arm_to_jpos(self.robot.home_arm_jpos)

        return True # how to check this if we even need to

    def move_piece(self, from_spot: int, to_spot: int):
        """
        Move a piece from the given spot on the board to the given spot. A `spot` in this
        context refers to how the spots are numbered given.

        This doesn't affect anything about the state of the game, just physically moves pieces
        """

        from_row, from_col = self.board_spot_to_row_col[from_spot]
        to_row, to_col = self.board_spot_to_row_col[to_spot]

        from_x, from_y = self.board_positions[from_row][from_col]
        to_x, to_y = self.board_positions[to_row][to_col]

        # TODO what theta works, 0 might be ok
        self.perform_grasp(from_x, from_y, 0)
        self.place_piece(to_x, to_y, 0)

    
    def set_board_position(self) -> None:
        '''Center the board in the workspace
        '''
        x = (self.workspace[0][0] + self.workspace[1][0]) / 2
        y = (self.workspace[0][1] + self.workspace[1][1]) / 2
        theta = 0
        pos = np.array((x,y,0.01))
        quat = pb.getQuaternionFromEuler((0,0, theta))
        pb.resetBasePositionAndOrientation(self.board_id, pos, quat)

    def set_pieces(self) -> None:
        '''Place the initial pieces on the board
        '''
        self.checkers = Game()
        initial_yellow = [
            (0, 1), (0, 3), (0, 5), (0, 7),
            (1, 0), (1, 2), (1, 4), (1, 6),
            (2, 1), (2, 3), (2, 5), (2, 7)
        ]

        initial_white = [
            (5, 0), (5, 2), (5, 4), (5, 6),
            (6, 1), (6, 3), (6, 5), (6, 7),
            (7, 0), (7, 2), (7, 4), (7, 6)
        ]

        initial_empty = [
            (3, 0), (3, 2), (3, 4), (3, 6),
            (4, 1), (4, 3), (4, 5), (4, 7)
        ]

        for piece_id, (i, j), checkers_position in zip(self.yellow_pieces_ids, initial_yellow, range(1, 13)):
            position = self.board_positions[i][j]
            pos = np.array([position[0], position[1], self.board_height+(self.piece_height/2) + 0.2])
            quat = pb.getQuaternionFromEuler((0,0,0))
            pb.resetBasePositionAndOrientation(piece_id, pos, quat)
            self.yellow_checkers_board_spots.add((i, j))
            self.board_spot_to_row_col[checkers_position] = (i, j)

        for piece_id, (i, j), checkers_position in zip(self.white_pieces_ids, initial_white, range(21, 33)):
            position = self.board_positions[i][j]
            pos = np.array([position[0], position[1], self.board_height+(self.piece_height/2) + 0.2])
            quat = pb.getQuaternionFromEuler((0,0,0))
            pb.resetBasePositionAndOrientation(piece_id, pos, quat)
            self.white_checkers_board_spots.add((i, j))
            self.board_spot_to_row_col[checkers_position] = (i, j)

        for (i, j), checkers_position in zip(initial_empty, range(13, 21)):
            self.board_spot_to_row_col[checkers_position] = (i, j)


    def take_picture(self) -> np.ndarray:
        '''Takes picture using camera

        Returns
        -------
        np.ndarray
            rgb image of shape (H,W,3) and dtype of np.uint8
        '''
        return self.camera.get_rgb_image()


def check_workspace_reachability():
    '''Use this to test your solve_ik implementation.  If it is working properly
    then you should see a green semi circle that extends up to the edge of the
    workspace.  Outside of this range, the robot is not able to perform a top
    down grasp.
    '''
    env = TopDownGraspingEnv(True)

    top_down_quat = pb.getQuaternionFromEuler((0,-np.pi,0))

    # perform scan over x,y positions
    for x in np.linspace(0.1, 0.3, num=20):
        for y in np.linspace(-0.15, 0.15, num=20):
            pos = np.array((x, y, env.piece_height))
            _, residuals = env.robot.solve_ik(pos, top_down_quat)

            if residuals['position'] < 1e-2 and residuals['orientation'] < 1e-3:
                color = (0,1,0) # green means its feasible
            else:
                color = (1,0,0) # red means not feasible

            pb.addUserDebugLine(pos-0.001, pos+0.001, color, 10)

    env.set_board_position()
    env.set_board_texture()
    # env.set_pieces()

    while 1:
        env.take_picture()
        # time.sleep(0.5)


def test_move_gripper_to():
    '''Use this to test your implementation of `RobotArm.move_gripper_to`.
    If it is working correctly, then the gripper should move to a position, then slowly
    rotate its gripper, before returning to the home position.
    '''
    env = TopDownGraspingEnv(True)

    while 1:
        env.robot.move_arm_to_jpos(env.robot.home_arm_jpos)

        position = (0.16, 0., 0.01)
        for theta in np.linspace(-np.pi/4, np.pi/4, num=6, endpoint=True):
            env.robot.move_gripper_to(position, theta)
            time.sleep(0.1)

        time.sleep(0.5)


def test_checkers_board_object():
    '''Function to check that the camera lines up with the checkers board
    '''
    env = TopDownGraspingEnv(True)
    env.set_board_position()
    env.set_pieces()

    while 1:
        env.take_picture()

def test_move_piece():
    env = TopDownGraspingEnv(True)
    env.set_board_position()
    env.set_pieces()
    env.move_piece(9, 13)

    while 1:
        env.take_picture()

if __name__ == "__main__":
    test_move_piece()
    # test_checkers_board_object()
    # check_workspace_reachability()