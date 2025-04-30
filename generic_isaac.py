from omni.isaac.kit import SimulationApp

# CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}
CONFIG = {"sync_loads": True, "headless": False, "multi_gpu": False}

#TODO: Create config json
kit = SimulationApp(launch_config=CONFIG)

import simulation_handler


if __name__ == "__main__":
    
    sim = simulation_handler.WassmannHandler(kit)
    sim.spin()
