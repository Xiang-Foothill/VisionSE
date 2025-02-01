import numpy as np
import carla
import matplotlib.pyplot as plt
import sys
import data_utils as du
import Estimators
import Ego_motion as em
import video_utils as vu
import cv2

sys.path.insert(0,'C:/carla/PythonAPI/carla') 
# To import a basic agent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent

def play_game(upper_ticks = 500, deltaT = 0.04):
    # prepare the world and the client
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    world.actor_list = []
    client.set_timeout(10.0)
    
    world.data = {}
    world.data["T"] = deltaT
    world.data["real_Vl"] = []
    world.data["real_w"] = []
    world.data["estimated_Vl"] = []
    world.data["estimated_w"] = []
    
    ego = spawn_vehicle(world)

    world.estimator = Estimators.OP_estimator(deltaT = world.data["T"], h = world.data["sensor_height"], f = world.data["F"], pre_filter_discard=6.0, pre_filter_size=10, past_fusion_size = 20, past_fusion_amplifier = 1.2, past_fusion_on= True)

     # set the world to the synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = world.data["T"] # !!!! Note: to avoid the error of carla simulation, try to set the sensor_tick time as the same value as the world_tick time
    world.apply_settings(settings)

    set_spectator(world)

    # To start a basic agent
    agent = BasicAgent(ego)

    while True:
        ego.apply_control(agent.run_step())
        updata_spectator(world)
        world.tick()

        if judge_end(world, upper_ticks):
            break
    
    clear_world(world)
    f, (ax1, ax2) = plt.subplots(1, 2)

    # world.data["estimated_Vl"] = em.median_filter(world.data["estimated_Vl"])
    # world.data["estimated_w"] = em.median_filter(world.data["estimated_w"])
    ax1.plot(world.data["real_Vl"], label = "real_Vl")
    ax1.plot(world.data["estimated_Vl"], label = "est_vl")
    ax1.set_xlabel("frame number")
    ax1.set_ylabel("V_long (m / s)")
    ax2.plot(world.data["real_w"], label = "real_w")
    ax2.plot(world.data["estimated_w"], label = "est_w")
    ax2.set_xlabel("frame number")
    ax2.set_ylabel("w (rad / s)")
    ax1.legend()
    ax2.legend()
    plt.show()
    
def judge_end(world, upper_ticks):
    return len(world.data["real_Vl"]) > upper_ticks

def clear_world(world):
    for actor in world.actor_list:
        actor.destroy()

def updata_spectator(world):
    """update the position of the spectator to the position of the camera"""
    spectator = world.get_spectator()
    sensor = world.actor_list[1]
    sensor_transform = sensor.get_transform()
    spectator.set_transform(sensor_transform)

def set_spectator(world):
    """move the spectator to a high position right above the whole map"""
    spectator = world.get_spectator()
    target_pos = carla.Transform(carla.Location(0.0, 0.0, 60.0), carla.Rotation(-90.0, 0.0, 0.0))
    spectator.set_transform(target_pos)

def spawn_vehicle(world):
    #spawn the vehicle at a random position
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.find("vehicle.audi.a2")
    world_map = world.get_map()
    spawn_points = world_map.generate_waypoints(distance = 0.5)
    np.random.seed(10) # set the value of the seed so that the experiment is repeatable
    point_index = np.random.randint(low = 0, high = len(spawn_points))
    ego = world.spawn_actor(blueprint, spawn_points[point_index].transform)

    # attach a rgb camera to the vehicle
    cam = blueprint_library.find("sensor.camera.rgb")
    imu = blueprint_library.find("sensor.other.imu")
    IM_W = 640
    IM_H = 480
    deltaT = world.data["T"]

    cam.set_attribute("image_size_x", f"{IM_W}")
    cam.set_attribute("image_size_y", f"{IM_H}")
    cam.set_attribute('sensor_tick', f"{deltaT}")


    sensor_height = 1.2
    cam_2_v = carla.Transform(carla.Location(x = 2.9, z = sensor_height))
    rgb_sensor = world.spawn_actor(cam, cam_2_v, attach_to = ego)

    world.actor_list.append(ego)
    world.actor_list.append(rgb_sensor)

    world.data["F"] = findFocal(cam)
    world.data["sensor_height"] = sensor_height

    f_recall = make_exp_recall(world)
    rgb_sensor.listen(f_recall)

    return ego

def make_exp_recall(world):
    """make the recall function used for real-time experiment
    once the image is passed in, use the functions from OP_flow to do real_time speed estimation"""
    # prepare the world parameters for real_time speed estimation
    def exp_recall(image):
        IM_array = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        IM_array = du.BGRA2RGB(IM_array)
        IM_array = np.uint8(IM_array)

        if world.estimator.preImg is None:
            world.estimator.setPreImg(IM_array)
            return
        
        ego = world.actor_list[0]
        V = ego.get_velocity()
        real_omega = np.deg2rad(ego.get_angular_velocity().z)
        V_x, V_y = V.x, V.y
        real_V = (V_x ** 2 + V_y ** 2) ** 0.5

        world.data["real_Vl"].append(real_V)
        world.data["real_w"].append(real_omega)
        est_Vl, est_w, error = world.estimator.estimate(IM_array)
        world.data["estimated_Vl"].append(est_Vl)
        world.data["estimated_w"].append(est_w)

    return exp_recall


def findFocal(sensor):
    """find the pixel focal length of a carla rgb sensor"""
    fov = sensor.get_attribute('fov').as_float()
    width = sensor.get_attribute('image_size_x').as_int()

    F = (width / 2) / np.tan(np.deg2rad(fov / 2))

    return F

def main():
   play_game(1000, 0.04)

if __name__ == "__main__":
    main()