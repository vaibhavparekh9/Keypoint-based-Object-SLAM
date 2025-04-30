import omni
from omni.isaac.nucleus import is_file
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import is_stage_loading
# import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core import World

from omni.isaac.core import PhysicsContext
from omni.isaac.core.utils.semantics import add_update_semantics, remove_all_semantics
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import carb 
import sys

# from shreyash_thesis.planning import planning as c 
# from isaac_extensions import theia
# from isaac_extensions import agent

#code for using built in isaac robots 
from omni.isaac.core.utils.nucleus import get_assets_root_path

from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import ArticulationView
import omni.graph.core as og
import omni.replicator.core as rep
# import omni.timeline
#code for using built in isaac robotss

import generic_robot

class WassmannHandler:
    """
   

    Attributes:
        kit: The simulation kit.
        world: The simulation world.
        physx: The physics context.
    """

    def __init__(self, kit) -> None:
        self.kit = kit
        # self.draw = _debug_draw.acquire_debug_draw_interface()
        # self.graph = c.TraversalGraph('/src/src/planning/graphs/test_world_1_graph.json')
        self.load_stage()
        self.world = World(physics_dt=1/80,rendering_dt=1/40)
        self.physx = self.world.get_physics_context()
        self.physx.set_gravity(-9.8)

        omni.timeline.get_timeline_interface().play()
        
        # self.forklift=agent.Agent(self.graph, "/src/config/forklift.json",self.world)
        # self.forklift2=agent.Agent(self.graph, "/src/config/forklift2.json",self.world)
        
        #loading in the isaac sim jetbot
        self.jetbot = generic_robot.GenericRobot(self.world)
        #end code to load in isaac sim jetbot
        self.world.initialize_physics()
        # #self.theia = theia.Theia(self.graph, "/src/config/theia_robot.json", self.world)
        self.iteration=0
        self.new_frame_ready = False
        self.kit.update()
        # self.timeline = omni.timeline.get_timeline_interface()
        # self.kit.update()
        # self.kit.update()




  

    def spin(self) -> None:
        """
        Start the simulation spin loop.
        """
        # self.theia.initialize()
        # self.kit.update()
        # self._jetbot = self.world.scene.get_object("fancy_robot")
        # print("Num of degrees of freedom after first reset: " + str(self._jetbot.num_dof)) # prints 2
        # print("Joint Positions after first reset: " + str(self._jetbot.get_joint_positions()))
        self.jetbot.init_after_reset()
        self.kit.update()
        
        self.jetbot.initalize_camera()
        self.kit.update()
       
        # self.timeline.play()
        self.iter = 0
        self.camcounter=0
        self.gp = np.array([[0,-10],[25,-10],[25,7.5],[0.01,7.5]]) 
        self.world.add_physics_callback("sending_actions", callback_fn=self.robot_physics_step)
        self.world.add_render_callback("render_callback", self.render_callback)
        self.jetbot.send_robot_actions(self.gp[self.iter])
        while self.kit.is_running():
            self.spin_once()

        omni.timeline.get_timeline_interface().stop()
        self.kit.close()

    def load_stage(self):
        """
        Load the simulation stage.
        """
        # usd_path = "omniverse://cerlabnucleus.lan.local.cmu.edu/Users/gmetts/theia_isaac_qual/world/test_world_1.usd"
        # usd_path = "omniverse://cerlabnucleus.lan.local.cmu.edu/Library/ANSYS/test.usd"
        # usd_path = "omniverse://cerlabnucleus.lan.local.cmu.edu/Users/weihuanw/sim_environments/large_modular_warehouse_base.usd"
        usd_path = "omniverse://cerlabnucleus.lan.local.cmu.edu/Users/ewassman/Large_warehouse.usd"
        try:
            result = is_file(usd_path)
        except:
            result = False

        if result:
            omni.usd.get_context().open_stage(usd_path)
        else:
            carb.log_error(
                f"the usd path {usd_path} could not be opened"
            )
            self.kit.close()
            sys.exit()

        print("Loading stage...")
        while is_stage_loading(): 
            self.kit.update()

        # remove_all_semantics(get_prim_at_path("/Environment"), recursive=True)
        # add_update_semantics(get_prim_at_path("/Environment"))
        print("Loading Complete")        

    def robot_physics_step(self,step_size):
        if np.linalg.norm(self.jetbot.get_position()[0:2] - self.gp[self.iter]) < 0.3:
            print("Iteration: " + str(self.iter))
            if(self.iter == 3):
                self.iter = 0
            else:
                self.iter += 1
        
        self.jetbot.send_robot_actions(self.gp[self.iter])
        self.camcounter+=1
        # self.jetbot.update_image() # Safe here
        self.new_frame_ready = True  # Tell render callback to read next frame
        return
        # self.jetbot.get_image()

    def render_callback(self,step_size):
        if self.camcounter>=10:
            if self.new_frame_ready:
                self.jetbot.get_image()  # Only called when we know new data is ready
                self.new_frame_ready = False
            self.camcounter=0
        return
        


    def spin_once(self):
        """
        Perform one iteration of the simulation spin loop.
        """
        
        # self.draw_update()
        # self.forklift.spin_once()
        # self.forklift2.spin_once()
        # print(self.jetbot.get_position())
        
  
        # self.camcounter+=1
        # # if(self.camcounter == 1000):
        # self.jetbot.get_image()
        # self.camcounter = 0
        # self.jetbot.send_robot_actions(self.gp[self.iter])
        self.kit.update()
        
        
        # self.kit.update()

    #TODO: Add reset function