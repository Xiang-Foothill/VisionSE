import numpy as np
import carla
import matplotlib.pyplot as plt
import sys
import data_utils as du
import pickle
import os

sys.path.insert(0,'C:/carla/PythonAPI/carla') 
# To import a basic agent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent

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
    ego = world.spawn_actor(blueprint, spawn_points[0].transform)

    # attach a rgb camera to the vehicle
    cam = blueprint_library.find("sensor.camera.rgb")

    IM_W = 640
    IM_H = 480
    T = 0.05

    cam.set_attribute("image_size_x", f"{IM_W}")
    cam.set_attribute("image_size_y", f"{IM_H}")
    cam.set_attribute('sensor_tick', f"{T}")
    sensor_height = 1.2
    cam_2_v = carla.Transform(carla.Location(x = 2.9, z = sensor_height))
    sensor = world.spawn_actor(cam, cam_2_v, attach_to = ego)
    
    world.actor_list.append(ego)
    world.actor_list.append(sensor)

    world.data["F"] = findFocal(cam)
    world.data["sensor_height"] = sensor_height
    world.data["T"] = T
    f_recall = make_recall(world)
    sensor.listen(f_recall)

    return ego

def findFocal(sensor):
    """find the pixel focal length of a carla rgb sensor"""
    fov = sensor.get_attribute('fov').as_float()
    width = sensor.get_attribute('image_size_x').as_int()

    F = (width / 2) / np.tan(np.deg2rad(fov / 2))

    return F

def updata_spectator(world):
    """update the position of the spectator to the position of the camera"""
    spectator = world.get_spectator()
    sensor = world.actor_list[1]
    sensor_transform = sensor.get_transform()
    spectator.set_transform(sensor_transform)

def make_recall(world):
    """generate the data_recall function applied when the RGB sensor is listening"""
    def data_recall(image):
        states = world.data["states"]
        images = world.data["images"]
        IM_array = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        IM_array = du.BGRA2RGB(IM_array)
        images.append(IM_array)

        ego = world.actor_list[0]
        V = ego.get_velocity()
        omega = np.deg2rad(ego.get_angular_velocity().z)
        V_x, V_y = V.x, V.y
        states.append(np.asarray([V_x, V_y, omega]))

    return data_recall

def prepareData(world):
    world.data = {}
    world.data["images"] = []
    world.data["states"] = []

def clear_world(world):
    for actor in world.actor_list:
        actor.destroy()
    world.data["states"] = np.asarray(world.data["states"])
    world.data["images"] = np.asarray(world.data["images"])

def judge_end(world):
    upper_ticks = 300 # maximum number of ticks allowed for this experiment
    states = world.data["states"]
    return len(states) > upper_ticks

def play_game():
    # prepare the world and the client
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    world.actor_list = []
    prepareData(world)
    client.set_timeout(10.0)
    ego = spawn_vehicle(world)
    set_spectator(world)

    # To start a basic agent
    agent = BasicAgent(ego)

    while True:
        ego.apply_control(agent.run_step())
        updata_spectator(world)

        if judge_end(world):
            break
    
    clear_world(world)
    cur_path = os.getcwd()
    SE_root = os.path.dirname(cur_path)
    path_to_save =  SE_root + "/VideoSet/" + "carlaData1.pkl"
    saveData(world.data, path_to_save)
    # du.random_image_test(world.data["images"])

def saveData(data, path):
    with open(path, "wb") as file:
        pickle.dump(data, file)

def __main__():
    play_game()

if __name__ == "__main__":
    __main__()