from omni.isaac.core.utils.nucleus import get_assets_root_path

from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.wheeled_robots.robots import WheeledRobot

from omni.isaac.core.utils.types import ArticulationAction
# This extension includes several generic controllers that could be used with multiple robots

from omni.isaac.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController

# Robot specific controller

from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.sensor import Camera
import carb 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import Keypoint_slam
#this will allow me to get ground truth data
import omni.replicator.core as rep
import omni.isaac.core.utils.prims as prim_utils
import controller
# from omni.replicator.core import  AnnotatorRegistry
class GenericRobot:



    def __init__(self, world):
        #must be done before physics are initalized
        self.world = world
        assets_root_path = get_assets_root_path()
        self.img_counter = 0
        if assets_root_path is None:

            # Use carb to log warnings, errors and infos in your application (shown on terminal)

            carb.log_error("Could not find nucleus server with /Isaac folder")

        asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd" # "/Isaac/Robots/Clearpath/Jackal/jackal.usd" #
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/Fancy_Robot")
        # jetbot_robot = self.world.scene.add(Robot(prim_path="/World/Fancy_Robot", name="fancy_robot"))
        self.jetbot = world.scene.add(

            WheeledRobot(

                prim_path="/World/Fancy_Robot",

                name="fancy_robot",

                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],

                create_robot=True,

                usd_path=asset_path,

            )

        )
        self.scaler = 1
        # prim_utils.get_prim_at_path("/World/Fancy_Robot").set_local_scale(self.scaler*np.array([1.0, 1.0, 1.0]))
        jetbot_view = ArticulationView(prim_paths_expr="/World/Fancy_Robot", name="jetbot_view")
        self.world.scene.add(jetbot_view)
        print("Num of degrees of freedom before first reset: " + str(self.jetbot.num_dof))
        # self.jetbot.set_world_pose(position=np.array([0, 0, 0.1]))
        self.kp = Keypoint_slam.KeypointSlam()
        self.kp.setUp()

    def init_after_reset(self):
        self._jetbot = self.world.scene.get_object("fancy_robot")
        # self.world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)

        # Initialize our controller after load and the first reset

        # self._my_controller = WheelBasePoseController(name="cool_controller",

        #                                                 open_loop_wheel_controller=

        #                                                     DifferentialController(name="simple_control",

        #                                                                             wheel_radius=self.scaler*0.03, wheel_base=self.scaler*0.1125,max_linear_speed=1e+21,max_angular_speed=1e+21),

        #                                             is_holonomic=False)
        self._my_controller = controller.RobustPoseController(wheel_radius=0.03, wheel_base=0.1125, scaler=self.scaler)
        print("Num of degrees of freedom after first reset: " + str(self._jetbot.num_dof)) # prints 2
        print("Joint Positions after first reset: " + str(self._jetbot.get_joint_positions()))

    def initalize_camera(self):
    
        # self.camera = Camera(prim_path="/World/Jetbot/chassis/rgb_camera/jetbot_camera", resolution=(256, 256))
        self.camera_prim_path = "/World/Fancy_Robot/chassis/rgb_camera/jetbot_camera"
        self.camera = Camera(prim_path="/World/Fancy_Robot/chassis/rgb_camera/jetbot_camera", resolution=(1500, 1500))
        self.camera.initialize()
        # self.camera.set_focal_length(1.8)
      
        self.render_product = rep.create.render_product("/World/Fancy_Robot/chassis/rgb_camera/jetbot_camera", [1500, 1500])
        # self.writer = rep.WriterRegistry.get("BasicWriter")
        self.skeleton_anno = rep.annotators.get("skeleton_data")
        self.dist_anno = rep.annotators.get("distance_to_image_plane")
        self.image_anno = rep.annotators.get("rgb")
   
        self.skeleton_anno.attach(self.render_product)
        self.dist_anno.attach(self.render_product)
        self.image_anno.attach(self.render_product)
        rep.orchestrator.set_capture_on_play(False) #throws an odd seg fault without this....
        rep.orchestrator.run()
        
        return
    
    # def update_image(self):
    #     rep.orchestrator.step(1)
    def get_image(self):

        # self.camera.update(render=True)
     
        self.img_counter += 1
        # rep.orchestrator.step(2)
        data_rgb = self.image_anno.get_data()
        try:
            data = self.skeleton_anno.get_data()
        except:
            print("skeleton instance seg changed")
            return
        
        if data_rgb is None or data is None:
            print("RGB image or skeleton data not ready yet")
            return
        
        data_dist = self.dist_anno.get_data()
    
        self.kp.process(data,data_dist,data_rgb,self.camera_prim_path, self.camera)
        if self.img_counter ==50:
            self.img_counter = 0
            self.kp.optimize()

        return
        

    def get_position(self):
        position,orientation = self._jetbot.get_world_pose()
        
        return position
    
    def send_robot_actions(self, gp):
        
        position, orientation = self._jetbot.get_world_pose()
        # print(position)
        # print("GP: ",gp)
        action = self._my_controller.compute_control(position,orientation, gp)
        self._jetbot.apply_action(action)
        # self._jetbot.apply_action(self._my_controller.forward(start_position=position,

        #                                                     start_orientation=orientation,

        #                                                     goal_position=gp,heading_tol=0.1,position_tol=0.15))
        # # print("AA: ")
        # print(self._my_controller.forward(start_position=position,

        #                                                     start_orientation=orientation,

        #                                                     goal_position=gp,lateral_velocity=1000,yaw_velocity=2,heading_tol=0.2,position_tol=4))

        return