import numpy as np
import carla
import matplotlib.pyplot as plt
import sys

sys.path.insert(0,'C:/carla/PythonAPI/carla') 
# To import a basic agent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent

def set_spectator(world):
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
    cam.set_attribute("image_size_x", f"{IM_W}")
    cam.set_attribute("image_size_y", f"{IM_H}")
    cam_2_v = carla.Transform(carla.Location(x = 2.5, z = 0.7))
    sensor = world.spawn_actor(cam, cam_2_v, attach_to = ego)
    
    world.actor_list.append(ego)
    world.actor_list.append(sensor)

    return ego

def updata_spectator(world):
    """update the position of the spectator to the position of the camera"""
    spectator = world.get_spectator()
    sensor = world.actor_list[1]
    sensor_transform = sensor.get_transform()
    spectator.set_transform(sensor_transform)

def play_game():
    # prepare the world and the client
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    world.actor_list = []
    client.set_timeout(10.0)
    ego = spawn_vehicle(world)

    set_spectator(world)

    # To start a basic agent
    agent = BasicAgent(ego)

    while True:
        ego.apply_control(agent.run_step())
        updata_spectator(world)

def __main__():
    play_game()

if __name__ == "__main__":
    __main__()