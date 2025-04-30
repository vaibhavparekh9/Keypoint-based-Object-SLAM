import numpy as np
import matplotlib.pyplot as plt
import ast
import omni.isaac.core.utils.prims as prim_utils
from mpl_toolkits.mplot3d import Axes3D
import omni.usd
# import omni.syntheticdata as sd
# from omni.isaac.core.utils.physics import raycast_closest
# from omni.isaac.core.utils.viewports import get_viewport_interface
import omni.physx
from omni.kit.viewport.utility import get_active_viewport
from omni.physx import get_physx_scene_query_interface
from pxr import Gf
import helpers
import cv2
# from functools import partial
import gtsam
from gtsam import (DoglegOptimizer, DoglegParams,
                   DummyPreconditionerParameters, GaussNewtonOptimizer,
                   GaussNewtonParams, GncLMOptimizer, GncLMParams, GncLossType,
                   LevenbergMarquardtOptimizer, LevenbergMarquardtParams,
                   NonlinearFactorGraph, Ordering, PCGSolverParameters, Point2,
                   PriorFactorPoint2, Values)
#export PYTHONPATH=/src/github/borglab/gtsam/build/python:$PYTHONPATH



class KeypointSlam:
    def __init__(self):
        
        self.symbol_list_x  = []
        self.symbol_list_l = []
        self.graph = gtsam.NonlinearFactorGraph()
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.05, 0.05, 0.05, np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)])
            )
        self.odometry_noise_vector = np.array([0.05, 0.05, 0.05, np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)])
        self.br_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.deg2rad(5),np.deg2rad(5), 0.5]))  # angle and range
        self.br_noise_vector = np.array([np.deg2rad(5),np.deg2rad(5), 0.5])
        self.landmark_to_symbol = dict()
        self.initial = gtsam.Values()
        self.gt_robot_poses = []
        # self.vertical_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 0.01)
        self.expected_vertical_displacement = np.array([0.0, 0.0, 25.9])
        self.gt_keypoint_positions = []

    
    @staticmethod
    def vertical_constraint_error(this, values, jacobians):
        #jacobians: Optional[List[np.ndarray]]
        # keys = this.keys()
        
        keys = this.keys()
        p1 = values.atPoint3(keys[0])
        p2 = values.atPoint3(keys[1])
        # assert values.exists(keys[0]), f"Key {keys[0]} not in values"
        # assert values.exists(keys[1]), f"Key {keys[1]} not in values"
        # print("p1: ",p1)
        expected_disp = np.array([0.0, 0.0, 26.0])
        # vec1 = np.array([p1.x(), p1.y(), p1.z()])
        # vec2 = np.array([p2.x(), p2.y(), p2.z()]) 
        # print("Error: ",np.array(p2 - p1 - expected_disp ).shape)
        if jacobians is not None:
            jacobians[0] = -np.eye(3)
            jacobians[1] = np.eye(3)
            # jacobians.append(-np.eye(3))  # ∂error/∂p1
            # jacobians.append(np.eye(3))   # ∂error/∂p2
        return np.array(p2 - p1 - expected_disp, dtype=np.float64)

    def create_vertical_constraint_factor(self, Bottom_symbol, Top_symbol):
        print("constructed")
        vertical_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 0.3) #unsure why 0.3
        factor = gtsam.CustomFactor(
            vertical_noise_model,
            [Bottom_symbol.key(), Top_symbol.key()],
            KeypointSlam.vertical_constraint_error  # now clean and static
        )
        self.graph.add(factor)



    @staticmethod
    def parallel_constraint_error(this, values, jacobians):
        #jacobians: Optional[List[np.ndarray]]
        # keys = this.keys()
        
        keys = this.keys()
        p1 = values.atPoint3(keys[0])
        p2 = values.atPoint3(keys[1])
        p3 = values.atPoint3(keys[2])
        p4 = values.atPoint3(keys[3])
        # assert values.exists(keys[0]), f"Key {keys[0]} not in values"
        # assert values.exists(keys[1]), f"Key {keys[1]} not in values"
        # print("p1: ",p1)
        # expected_disp = np.array([0.0, 0.0, 26.0])
        # vec1 = np.array([p1.x(), p1.y(), p1.z()])
        # vec2 = np.array([p2.x(), p2.y(), p2.z()]) 
        # print("Error: ",np.array(p2 - p1 - expected_disp ).shape)
        A= p2-p1
        b = p4-p3

        if jacobians is not None:
            jacobians[0] = -np.skew(b)   
            jacobians[1] =  np.skew(b)   
            jacobians[2] =  np.skew(A)   
            jacobians[3] = -np.skew(A)   

        return np.cross(A,b)

    def perpendicular_constraint_error(this: gtsam.CustomFactor,
                                    values: gtsam.Values,
                                    jacobians) -> np.ndarray:
        kA1, kA2, kB1, kB2 = this.keys()

        # Retrieve points
        A1 = values.atPoint3(kA1)
        A2 = values.atPoint3(kA2)
        B1 = values.atPoint3(kB1)
        B2 = values.atPoint3(kB2)


        a = A2 - A1
        b = B2 - B1
        #May need to make this value positive? 
        error = np.dot(a, b)  # scalar

        if jacobians is not None:
            # Each Jacobian is 1x3
            jacobians[0] = -b.reshape(1, 3)  # ∂e/∂A1
            jacobians[1] =  b.reshape(1, 3)  # ∂e/∂A2
            jacobians[2] = -a.reshape(1, 3)  # ∂e/∂B1
            jacobians[3] =  a.reshape(1, 3)  # ∂e/∂B2

        return np.array([error])  # must return a (1,) array

    def create_perpendicular_constraint_factor(self, line1_a, line1_b, line2_a, line2_b):
        #we will, just take the dot product and try to make it zero
        perpendicular_noise_model = gtsam.noiseModel.Isotropic.Sigma(1, 0.5) #try to mess with this vale
        factor = gtsam.CustomFactor(
            perpendicular_noise_model,
            [line1_a.key(), line1_b.key(),line2_a.key(),line2_b.key()],
            KeypointSlam.perpendicular_constraint_error  # now clean and static
        )
        self.graph.add(factor)


    def create_parallel_constraint_factor(self, line1_a, line1_b, line2_a, line2_b):
        #we will, just take the cross product and try to make it zero
        parallel_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 0.5) #this one too
        factor = gtsam.CustomFactor(
            parallel_noise_model,
            [line1_a.key(), line1_b.key(),line2_a.key(),line2_b.key()],
            KeypointSlam.parallel_constraint_error  # now clean and static
        )
        self.graph.add(factor)
    #probably want to move camera_prim_path and render_product_path to the constructor
    def setUp(self):
        """Set up the optimization problem and ordering"""
        # create graph
        i1 = gtsam.Symbol('x', 1)
        self.symbol_list_x.append(i1)
        
        prior_mean = gtsam.Pose3()
        print("initial pose: ",prior_mean)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])  # x, y, z, roll, pitch, yaw
        )
        self.graph.add(gtsam.PriorFactorPose3(i1.key(), prior_mean, prior_noise))
        self.create_initial_guess(i1,prior_mean)
        self.last_pose = prior_mean
    def create_symbol(self, symbol_list):
        # print(symbol_list[-1])
        if len(symbol_list)>0 and symbol_list[-1].chr() == 120: #yeah its ascii #len >0 because we start with pose 0 for robot poses
            # print("symbol list character: ",symbol_list[-1].chr())
            symbol_list.append(gtsam.Symbol('x',symbol_list[-1].index()+1))
        else:
            #there will be up to 40 landmarks, need to associate properly
            # print("symbol list character: ",symbol_list[-1].chr())
            if(len(symbol_list)>0):
                symbol_list.append(gtsam.Symbol('l',symbol_list[-1].index()+1)) 
            else:
                symbol_list.append(gtsam.Symbol('l',1))

    def create_initial_guess(self,symbol,pose):
        self.initial.insert(symbol.key(),pose)

    # def create_vertical_constraint_factor(self, Bottom_symbol, Top_symbol):
    #     #they should be 26 m apart in z direction
    #     factor = gtsam.CustomFactor(
    #         self.vertical_noise_model,
    #         [Bottom_symbol.key(), Top_symbol.key()],
    #         lambda values, keys, this: self.vertical_constraint_error(
    #         values.atPoint3(keys[0]),
    #         values.atPoint3(keys[1])
    #         )
    #     )
    #     self.graph.add(factor)



    def isaac_to_pose3(self, prim_path) -> gtsam.Pose3:
        """
        Convert a 4x4 transformation matrix (from Isaac Sim) to gtsam.Pose3.
        """
        transform = omni.usd.get_world_transform_matrix(prim_utils.get_prim_at_path(prim_path))
        R = transform.ExtractRotation()
        axis = np.array(R.GetAxis())
        axis = axis / np.linalg.norm(axis)  # Normalize the axis
        angle = R.GetAngle()
        rotvec = axis * np.deg2rad(angle)
        # print("rotation vector: ",rotvec)
        t = np.array(transform.ExtractTranslation())
        # print(t.shape)
        return gtsam.Pose3(gtsam.Rot3.Rodrigues(rotvec[0],rotvec[1],rotvec[2]), gtsam.Point3(t))

    def pose3_to_numpy(self, pose: gtsam.Pose3) -> np.ndarray:
        """Convert a gtsam.Pose3 to a 4x4 numpy array."""
        #TODO: needs to be fixed may be fixed now
        T = np.array(pose.matrix())
        # R = np.array(pose.rotation())
        # t = np.array(pose.translation())
        # T[:3, :3] = R
        # T[:3, 3] = t
        return T
    def spherical_to_unit3(self, spherical):
        """
        Convert spherical coordinates (azimuth, elevation, range) to gtsam.Unit3.
        """
        range_ = 1
        azimuth, elevation = spherical
        x = range_ * np.cos(elevation) * np.sin(azimuth)
        y = range_ * np.cos(elevation) * np.cos(azimuth)
        z = range_ * np.sin(elevation)
        
        return gtsam.Unit3(np.array([x, y, z]))
    def point_to_spherical(self, point):
        x = point[0]
        y = point[1]
        z = point[2]
        range_ = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(y, x)
        elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
        return azimuth, elevation, range_
    def spherical_to_point(self, spherical):
        """
        Convert spherical coordinates (azimuth, elevation, range) to a 3D point.
        """
        azimuth, elevation, range_ = spherical
        x = range_ * np.cos(elevation) * np.sin(azimuth)
        y = range_ * np.cos(elevation) * np.cos(azimuth)
        z = range_ * np.sin(elevation)
        return gtsam.Point3(x, y, z)
    def spherical_to_point_numpy(self, spherical):
        """
        Convert spherical coordinates (azimuth, elevation, range) to a 3D point.
        """
        
        azimuth, elevation, range_= spherical
        x = range_ * np.cos(elevation) * np.sin(azimuth)
        y = range_ * np.cos(elevation) * np.cos(azimuth)
        z = range_ * np.sin(elevation)
        return np.array([x, y, z])
    def xyzrpy_to_pose3(self, xyzrpy):
        """
        Convert a 6D vector (x, y, z, roll, pitch, yaw) to gtsam.Pose3.
        """
        x, y, z, roll, pitch, yaw = xyzrpy
        rotation = gtsam.Rot3.RzRyRx(yaw, pitch, roll) #technically it is not rpy becasue the axis is unchaning? 
        translation = gtsam.Point3(x, y, z)
        return gtsam.Pose3(rotation, translation)

    
    def relation_finder(self, landmark_name,landmark_dict):
        #parallel constraints ((back,top, right), (back, bottom, right)) ((back, top, left),(back, bottom, left))
        #within- group of two points, same except top/bottom, between group, same except right, left
        #parallel constraints ((front, top, right), back, top, right)) ((front, bottom, right), (back, bottom, right))
        #within- group of two points, same except front/back, between group, same except top/bottom
        #parallel constraints, ((front, top right),(front, bottom, right)) ((back, top, right),(back, bottom, right))
        #within group same ecvept top/bottom, betweenm groups, same except front/back
        #- all top botom groups are parallel with all other top bottom groups. all right left groups are paralled with all other right/left groups,  front/back groups are parallel with eachother
        #perpendicular constraints 
        #groups that are consistent with top/bottom should be perpendicular to all right/left and front/back groups, they are parallel to their groups and perpendicular to all others
        #given one landmark, find if its groups exist, top/bottom, front/back, and right/left
        #search for and create constraints from these three groups
        top_bottom_group = None
        front_back_group = None
        right_left_group = None

        landmark_symbol = landmark_dict[landmark_name]

        # Top-Bottom pairing
        if "top" in landmark_name:
            landmark_bottom = landmark_dict.get(landmark_name.replace("top", "bottom"))
            if landmark_bottom is not None:
                top_bottom_group = [landmark_symbol, landmark_bottom]
        elif "bottom" in landmark_name:
            landmark_top = landmark_dict.get(landmark_name.replace("bottom", "top"))
            if landmark_top is not None:
                top_bottom_group = [landmark_top, landmark_symbol]

        # Front-Back pairing
        if "front" in landmark_name:
            landmark_back = landmark_dict.get(landmark_name.replace("front", "back"))
            if landmark_back is not None:
                front_back_group = [landmark_symbol, landmark_back]
        elif "back" in landmark_name:
            landmark_front = landmark_dict.get(landmark_name.replace("back", "front"))
            if landmark_front is not None:
                front_back_group = [landmark_front, landmark_symbol]

        # Right-Left pairing
        if "right" in landmark_name:
            landmark_left = landmark_dict.get(landmark_name.replace("right", "left"))
            if landmark_left is not None:
                right_left_group = [landmark_symbol, landmark_left]
        elif "left" in landmark_name:
            landmark_right = landmark_dict.get(landmark_name.replace("left", "right"))
            if landmark_right is not None:
                right_left_group = [landmark_right, landmark_symbol]

        if top_bottom_group is not None:
            if front_back_group is not None:
                self.create_perpendicular_constraint_factor(top_bottom_group[0],top_bottom_group[1],front_back_group[0],front_back_group[1])
            if right_left_group is not None:
                self.create_perpendicular_constraint_factor(top_bottom_group[0],top_bottom_group[1],right_left_group[0],right_left_group[1])
        if front_back_group is not None and right_left_group is not None:
            self.create_perpendicular_constraint_factor(front_back_group[0],front_back_group[1],right_left_group[0],right_left_group[1])

        #Now check for parallel constraints
        all_landmarks = list(landmark_dict.keys())

        def find_parallel_pairs(current_group_name, group_prefix, replace_from, replace_to):
            # Search for other same-group landmarks
            base_pair = landmark_dict.get(current_group_name)
            alt_pair_name = current_group_name.replace(replace_from, replace_to)
            alt_pair = landmark_dict.get(alt_pair_name)
            if base_pair and alt_pair:
                return [base_pair, alt_pair]
            return None

        # Compare current top-bottom group to all other top-bottom pairs
        if top_bottom_group is not None:
            for other_name in all_landmarks:
                if other_name == landmark_name:
                    continue
                if "top" in other_name and "bottom" in landmark_dict:
                    other_pair = find_parallel_pairs(other_name, "top_bottom", "top", "bottom")
                    if other_pair is not None:
                        self.create_parallel_constraint_factor(*top_bottom_group, *other_pair)

        if front_back_group is not None:
            for other_name in all_landmarks:
                if other_name == landmark_name:
                    continue
                if "front" in other_name and "back" in landmark_dict:
                    other_pair = find_parallel_pairs(other_name, "front_back", "front", "back")
                    if other_pair is not None:
                        self.create_parallel_constraint_factor(*front_back_group, *other_pair)

        if right_left_group is not None:
            for other_name in all_landmarks:
                if other_name == landmark_name:
                    continue
                if "right" in other_name and "left" in landmark_dict:
                    other_pair = find_parallel_pairs(other_name, "right_left", "right", "left")
                    if other_pair is not None:
                        self.create_parallel_constraint_factor(*right_left_group, *other_pair)


                

    def addToGraph(self,robot_pose,landmark_bearing,landmark_range,landmark_name,landmark_location):
        #TODO noise robot pose, landmark bearings, and range measurments
        translation = omni.usd.get_world_transform_matrix(prim_utils.get_prim_at_path(robot_pose)).ExtractTranslation()
        # print("transform: ",transform)
        self.gt_robot_poses.append(translation)
        current_mean = self.isaac_to_pose3(robot_pose)
        odometry_noise = np.random.normal(0, self.odometry_noise_vector, 6)
        odometry_noise = self.xyzrpy_to_pose3(odometry_noise)
        #TODO test w/o noise
        noised_mean = current_mean.compose(odometry_noise)
        # noised_mean = current_mean

        #turn from xyz rpy to transformation matrix

        # print("real pose: ",translation)
        odometry = self.last_pose.between(noised_mean)
        rot_delta = odometry.rotation().xyz()
        # if np.linalg.norm(odometry.translation()) < 0.5 and rot_delta[0] < 0.05 and rot_delta[1] < 0.05 and rot_delta[2] < 0.05: #TODO: make efficient stop being lazy
        #     print("skipping odometry")
        #     return
        
        #TODO noise odometry #it is done


        self.create_symbol(self.symbol_list_x)
        self.graph.add(gtsam.BetweenFactorPose3(self.symbol_list_x[-2].key(),self.symbol_list_x[-1].key(),odometry,self.odometry_noise))
        self.create_initial_guess(self.symbol_list_x[-1],noised_mean) #should my initial guess be the noised mean? i think so
        self.last_pose = noised_mean #TODO: make sure this doesnt need to be the like updated and optimized version- i guess not because its batch

        if(landmark_bearing==None):
            return
        for idx, name in enumerate(landmark_name):
            landmark_symbol = self.landmark_to_symbol.get(name)
            # print("lop dict result: ",landmark_symbol)
            if landmark_symbol ==None: #TODO: make sure the initial guess is noised 
                self.create_symbol(self.symbol_list_l)
                self.gt_keypoint_positions.append([landmark_location[idx][0],landmark_location[idx][1],landmark_location[idx][2]])
                self.landmark_to_symbol[name] = self.symbol_list_l[-1]
                landmark_symbol = self.symbol_list_l[-1]
                #adding noise so that we use a noisy initialization
                # landmark_noised = gtsam.Point3(landmark_location[idx][0]+np.random.normal(0,0.5),landmark_location[idx][1]+np.random.normal(0,0.5),landmark_location[idx][2]+np.random.normal(0,0.5)) uncomment this for .5 m noise euclidian
                error_x, error_y, error_z = self.spherical_to_point_numpy(np.random.normal(0,self.br_noise_vector))
                landmark_noised = gtsam.Point3(landmark_location[idx][0]+error_x,landmark_location[idx][1]+error_y,landmark_location[idx][2]+error_z)
                relative_pose_noised = noised_mean.inverse().transformFrom(landmark_noised) 
                bearing_vec_noised = relative_pose_noised / np.linalg.norm(relative_pose_noised)
                # print(landmark_symbol)
                self.create_initial_guess(landmark_symbol,landmark_noised)

                #searches for and adds priors
                
                # self.relation_finder(name,self.landmark_to_symbol) #NEEDED

                ''' NEEDED- ADDS vertical PRIORS
                if "top" in name:
                    landmark_symbol_bottom = self.landmark_to_symbol.get(name.replace("top","bottom"))
                    if landmark_symbol_bottom != None:
                        self.create_vertical_constraint_factor(landmark_symbol_bottom,landmark_symbol) 
                else:
                    landmark_symbol_top = self.landmark_to_symbol.get(name.replace("bottom","top"))
                    if landmark_symbol_top != None:
                        self.create_vertical_constraint_factor(landmark_symbol,landmark_symbol_top)
                '''
                # related = get_related_symbols(name, self,landmark_to_symbol)
            else:
                landmark = gtsam.Point3(landmark_location[idx][0],landmark_location[idx][1],landmark_location[idx][2])
                relative_pose = noised_mean.inverse().transformFrom(landmark) 
                # relative_pose_noise = relative_pose + np.random.normal(0, np.array([0.5,0.5,0.5]), 3) uncomment this to add noise based on .5 meters in x,y,z
                # error_x, error_y, error_z = self.spherical_to_point_numpy(np.random.normal(0,self.br_noise_vector)) #this is how i am going from speherical to euclidean
                relative_pose_noise = relative_pose + self.spherical_to_point_numpy(np.random.normal(0,self.br_noise_vector))
                bearing_vec_noised = relative_pose_noise / np.linalg.norm(relative_pose)





            

            bearing = gtsam.Unit3(bearing_vec_noised)
            Br_noise = np.random.normal(0, self.br_noise_vector, 3)
            # print("bearing noise: ",Br_noise)
            
            #TODO: noise bearing and range
            
            # Bearing_point = bearing.point3()
            # Bearing_azimuth, Bearing_elevation, Br_range = self.point_to_spherical(Bearing_point) #range is always 1 cause unit3
            # bearing_noised = self.spherical_to_unit3((Bearing_azimuth+Br_noise[0], Bearing_elevation+Br_noise[1]))

            self.graph.add(gtsam.BearingRangeFactor3D(self.symbol_list_x[-1].key(),landmark_symbol.key(),bearing,landmark_range[idx]+Br_noise[2],self.br_noise))
            
        

        
    def optimize(self):
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial)
        # optimizer = gtsam.DoglegOptimizer(self.graph, self.initial)
        result = optimizer.optimize()
        self.visualize_trajectory(result)

    def visualize_trajectory(self,result):
        # Collect robot poses
        robot_positions = []
        for symb in self.symbol_list_x:
            pose = result.atPose3(symb.key())
            position = pose.translation()
            robot_positions.append([position[0], position[1], position[2]])

        # Collect landmark positions
        landmark_positions = []
        for symb in self.symbol_list_l:
            point = result.atPoint3(symb.key())
            landmark_positions.append([point[0], point[1], point[2]])

        # Convert to NumPy arrays for easier plotting
        robot_positions = np.array(robot_positions)
        gt_robot_positions = np.array(self.gt_robot_poses)
        landmark_positions = np.array(landmark_positions)

        # Plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')


        # print("robot position: ", robot_positions[-1])
        ax.plot(gt_robot_positions[:, 0], gt_robot_positions[:, 1], gt_robot_positions[:, 2],
                 label='True Robot trajectory', color='green', linewidth = 0.5)
        # Plot robot trajectory
        ax.plot(robot_positions[:, 0], robot_positions[:, 1], robot_positions[:, 2],
                 label='Robot trajectory', color='blue',linewidth=0.5)

        # print("ground truth robot position: ", gt_robot_positions[-1])
        # Plot landmarks
        ax.scatter(landmark_positions[:, 0], landmark_positions[:, 1], landmark_positions[:, 2],
                color='red', label='Landmarks', s=50, marker='^')
        # print("landmark positions: ",landmark_positions)
        # Labels and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D SLAM Result')
        ax.legend()
        ax.grid(True)

        plt.savefig("slam_results.png")
        mae_localization_error = np.sum(np.linalg.norm(gt_robot_positions-robot_positions[1:,:],axis=1))/robot_positions.shape[0]
        mae_mapping_error = np.sum(np.linalg.norm(self.gt_keypoint_positions-landmark_positions,axis=1))/landmark_positions.shape[1]
        rsme_localization_error = np.sqrt(np.sum(np.power(np.linalg.norm(gt_robot_positions-robot_positions[1:,:],axis=1),2))/robot_positions.shape[0])
        rsme_mapping_error = np.sqrt(np.sum(np.power(np.linalg.norm(self.gt_keypoint_positions-landmark_positions,axis=1),2))/landmark_positions.shape[1])
        #
        #error in localization:  0.3878149673830199 w/o priors
        #error in localization:  0.3511661709244964 w/ priors
        # improved priors - error in localization:  0.07998545622809128
        print("error in localization: ",mae_localization_error)
        print("error in mapping: ",mae_mapping_error)
        # plt.show()
        plt.close(fig)
        # if robot_positions.shape[0] >=500:
        #     raise ValueError("End test")


        

    def process(self, data,data_dist,data_rgb,camera_prim_path, camera):
        '''
        returns the locations and names of in view unobstructed keypoints

        '''
        viewport = get_active_viewport()

        viewport.set_active_camera(camera_prim_path)
        view_params = helpers.get_view_params(viewport) #sure hope this has the correct height and width
        
        skeleton_data = data #["skeleton_data"]
        
        inView = data.get("inView",[])
        # print("length of inview: ", len(inView))
        
        valid_points = []
        valid_bearing = []
        valid_range =[]
        valid_name = []
       
        width, height = camera.get_resolution()
        world_transform = omni.usd.get_world_transform_matrix(prim_utils.get_prim_at_path(camera_prim_path))
  
        occluded_joints = skeleton_data.get("jointOcclusions", []) #[True, False, False, True]
        occlusion_type = skeleton_data.get("occlusionTypes", []) # ["[{'class': 'window_front'}]", "[{'class': 'door_right'}]", "[{'class': 'loader_bucket'}]"]
     
        occlusion_type = [ast.literal_eval(item) for item in occlusion_type] # [[{'class': 'window_front'}], [{'class': 'door_right'}], [{'class': 'loader_bucket'}]]
        joint_names = skeleton_data.get("skelName", [])

        '''        for i in range(len(inView)):
            if(inView[i]):
                print(skeleton_data.get("skelPath")[i])
                print(occluded_joints[i])
                print(occlusion_type[i])
            '''
 
        skel_paths = skeleton_data.get("skelPath")
        points = np.zeros((len(skel_paths),3))

        for idx, skel_path in enumerate(skel_paths):
            path = "/".join(skel_path.split("/")[:-1])

            skel_prim = prim_utils.get_prim_at_path(path)
            skel_location = omni.usd.get_world_transform_matrix(skel_prim).ExtractTranslation()
            skel_location = np.array(skel_location)
            points[idx,:] = skel_location
            # point_locations.append(skel_location)

        if np.linalg.det(view_params["world_to_view"]) == 0.0:
            data["translations_2d"] = []
            data["in_view"] = False
            raise ValueError("View matrix determinant is 0.0, can't calculate 2D joint translations!")
        else:
            joint_pos2d = helpers.world_to_image(points, None, view_params)[:, :2]

            joint_pos2d *= np.array([view_params["width"], view_params["height"]])
            data["translations_2d"] = joint_pos2d.tolist()

        translations_2d = data.get("translations2d",[])
        
        point_locations = []
        distance_delta = np.zeros(len(inView))

        

        for idx, skel_path in enumerate(skel_paths):
            path = "/".join(skel_path.split("/")[:-1])

            skel_prim = prim_utils.get_prim_at_path(path)
            skel_location = omni.usd.get_world_transform_matrix(skel_prim).ExtractTranslation()
            skel_location = np.array(skel_location)
            point_locations.append(skel_location)
            
            # inView[idx] = True

            #this is called for skel prim in skel prim paths in the original code- we have only one prim at each path so it is ok to call it like this i believe
            #also only one joint per skeleton right now so translations 2d is (<num_joints>, 2) == (<num_skeletons>, 2)
            #note that in the original code the determinate of the view matrix is checked to see if it is 0.0- we are not doing this but i belive it is not necessary

                # Check if the current skeleton is in view of the viewport
            
            x = int(round(translations_2d[idx, 0]))
            y = int(round(translations_2d[idx, 1]))
            # Bounds check
            if not (0 <= x < width and 0 <= y < height):
                inView[idx] = False
                continue



            inView[idx] = True  # Passed all view checks 
            
       

            # Check for valid transform
            if world_transform is None:
                raise ValueError("World transform matrix is None, check camera prim path.")

            world_transform = np.array(world_transform)
            world_transform = world_transform.T #needed to convert from row to column major
            # print("world transform: ",world_transform)
            # Ensure it's 4x4
            if world_transform.shape != (4, 4):
                raise ValueError(f"Expected a 4x4 transform matrix, got {world_transform.shape}")

            # Ensure skel_location is valid
            if skel_location.shape != (3,):
                raise ValueError(f"Expected skel_location shape (3,), got {skel_location.shape}")

            # Convert to homogeneous coordinates
            skel_location_homogeneous = np.append(skel_location, 1)  # Shape (4,)

            # Matrix multiplication
            transformed_location = np.linalg.inv(world_transform) @ skel_location_homogeneous  # Shape (4,)

    
            if transformed_location[2] > 0: #idk why this is Y gotta check on that.... #may be different now
                inView[idx] = False
                continue



            # occluded_joints[idx] = True
            cam_x, cam_y, cam_z = omni.usd.get_world_transform_matrix(prim_utils.get_prim_at_path(camera_prim_path)).ExtractTranslation()
            # print(cam_x)
            # print(cam_y)
            # print(cam_z)
            skel_x,skel_y,skel_z = skel_location
            # Suppose you already have these:
            camera_position = Gf.Vec3f(cam_x, cam_y, cam_z)
            skel_position = Gf.Vec3f(skel_x, skel_y, skel_z)
            # print(skel_location)
            # Compute direction
            direction = (skel_position - camera_position).GetNormalized()
            #this right here is most of what we need to take the measurment, cam pose, skel pose, direction, name
            # Compute ray length
            max_distance = (skel_position - camera_position).GetLength()

            # Raycast
            hit = get_physx_scene_query_interface().raycast_closest(camera_position, direction, max_distance)
            # print(hit)
            # Check if the ray hit something *before* the skeleton
            if hit and hit['hit']:
                # hit_distance = (Gf.Vec3f(hit["position"][0],hit["position"][1],hit["position"][2]) - camera_position).GetLength()
                hit_distance = (Gf.Vec3f(*hit["position"]) - camera_position).GetLength()
                distance_delta[idx] = hit_distance-max_distance
                if hit_distance < max_distance - .5:  # small epsilon
                    # print("Skeleton_name: ", skel_path)
                    # print("hit info: ",hit)
                    # print(distance_delta[idx])
                    occluded_joints[idx] = True
                else:
                    occluded_joints[idx] = False
            else:
                distance_delta[idx] = 1111
                occluded_joints[idx] = False  # Nothing in the way

            if not occluded_joints[idx]:
                data_rgb= cv2.circle(data_rgb, (x, y), 8, (0, 255, 0, 255), -1)
                valid_points.append(skel_location)
                # print("landmark name: ", skel_path)
                # print("true landmark location: ",skel_location)
                valid_bearing.append(direction)
                valid_range.append(max_distance)
                valid_name.append(skel_path)


        self.addToGraph(camera_prim_path,valid_bearing,valid_range,valid_name,valid_points)

        # self.optimize()

        # point_locations = np.array(point_locations)
        
        # cv2.imwrite("Jetbot_view.png", data_rgb)
        
        
        # for i in range(len(inView) ):
        #     if(inView[i] and not occluded_joints[i]):
        #         # print(skeleton_data.get("skelPath")[i])
        #         # print(occluded_joints[i])
        #         # print(distance_delta[i])
        #         # # print(occlusion_type[i])
        #         valid_points.append(points[i])
        # # print("new data ----------------- \n")

        return
    


    

            