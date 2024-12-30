import numpy as np
import carla
import matplotlib.pyplot as plt
import sys
import data_utils as du
import pickle
import os
import OP_flow

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
    np.random.seed(14) # set the value of the seed so that the experiment is repeatable
    point_index = np.random.randint(low = 0, high = len(spawn_points))
    ego = world.spawn_actor(blueprint, spawn_points[point_index].transform)

    # attach a rgb camera to the vehicle
    cam = blueprint_library.find("sensor.camera.rgb")
    imu = blueprint_library.find("sensor.other.imu")
    IM_W = 640
    IM_H = 480
    T = 0.04
    imu_acc_noise_x = 2.0 # standard deviation parameter for the imu's x-direction noise measurement
    imu_acc_noise_y = 1.0 # standard deviation parameter for the imu's y-direction noise measurement
    imu_agu_noise_z = 0.3 # standard deviation parameter for the imu's angular velocity noise measurement

    cam.set_attribute("image_size_x", f"{IM_W}")
    cam.set_attribute("image_size_y", f"{IM_H}")
    cam.set_attribute('sensor_tick', f"{T}")
    imu.set_attribute("sensor_tick", f"{T}")
    imu.set_attribute("noise_accel_stddev_x", f"{imu_acc_noise_x}")
    imu.set_attribute("noise_accel_stddev_y", f"{imu_acc_noise_y}")
    imu.set_attribute("noise_gyro_stddev_z", f"{imu_agu_noise_z}")
    imu.set_attribute("noise_seed", f"{0}")

    sensor_height = 1.2
    cam_2_v = carla.Transform(carla.Location(x = 2.9, z = sensor_height))
    imu_2_v = carla.Transform(carla.Location(x = 0.0, y = 0.0, z = 0.0))
    rgb_sensor = world.spawn_actor(cam, cam_2_v, attach_to = ego)
    imu_sensor = world.spawn_actor(imu, imu_2_v, attach_to = ego)

    world.actor_list.append(ego)
    world.actor_list.append(rgb_sensor)
    world.actor_list.append(imu_sensor)

    world.data["F"] = findFocal(cam)
    world.data["sensor_height"] = sensor_height

    world.data["T"] = T
    if world.mode == "data":
        rgb_recall = make_data_recall(world)
        imu_recall = make_imu_data_recall(world)
        rgb_sensor.listen(rgb_recall)
        imu_sensor.listen(imu_recall)
    else:
        f_recall = make_exp_recall(world)
        rgb_sensor.listen(f_recall)

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

def make_data_recall(world):
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

def make_imu_data_recall(world):
    """generate the data_recall function applied when the imu sensor is listening"""
    def imu_data_recall(imuMeasurement):
        imu_data = world.data["imu"]
        acceleration_raw = imuMeasurement.accelerometer
        angular_raw = imuMeasurement.gyroscope
        al = acceleration_raw.x
        w = angular_raw.z
        imu_data.append(np.asarray([al, w]))
    
    return imu_data_recall

def make_exp_recall(world):
    """make the recall function used for real-time experiment
    once the image is passed in, use the functions from OP_flow to do real_time speed estimation"""
    # prepare the world parameters for real_time speed estimation
    def exp_recall(image):
        IM_array = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        IM_array = du.BGRA2RGB(IM_array)
        IM_array = np.uint8(IM_array)
        
        ego = world.actor_list[0]
        V = ego.get_velocity()
        real_omega = np.deg2rad(ego.get_angular_velocity().z)
        V_x, V_y = V.x, V.y
        real_V = (V_x ** 2 + V_y ** 2) ** 0.5

        world.data["realV"].append(real_V)
        world.plot["realV"] = real_V
        # iteract with the functions from OP_flow to estimate the value of speed
        params = world.data["parameters"]
        if params["preImg"] is None:
            opV = real_V
            params["preV"] = opV
            params["preImg"] = IM_array
        else:
            params["nextImg"] = IM_array
            opV, W, preV, preW, preVE, preWE, filtered_V = OP_flow.full_estimator(**params)
            params["preV"] = preV
            params["preW"] = preW
            params["preVE"] = preVE
            params["preWE"] = preWE
            params["preImg"] = IM_array
        
        world.data["opV"].append(opV)
        world.data["filtered_V"].append(filtered_V)
        # world.data["V_std"].append(V_std)

        print(f"the OP_flow estimated speed is {opV}, the real_time speed is {real_V}")
    return exp_recall

def prepareData(world):
    world.data["images"] = []
    world.data["states"] = []
    world.data["imu"] = []

def clear_world(world):
    for actor in world.actor_list:
        actor.destroy()
    # world.data["states"] = np.asarray(world.data["states"])
    # world.data["images"] = np.asarray(world.data["images"])

def judge_end(world):
    upper_ticks = 500 # maximum number of ticks allowed for this experiment
    if world.mode == "realTime":
        judge_array = world.data["realV"]
    elif world.mode == "data":
        judge_array = world.data["states"]
    return len(judge_array) > upper_ticks

def prepareExp(world):
    world.data["realV"] = []
    world.data["opV"] = []
    world.data["V_std"] = []
    world.data["filtered_V"] = []
    world.plot = {}
    world.plot["index"] = 0
    world.plot["N"] = 12
    world.plot["row"] = 5
    fig, axs = plt.subplots(world.plot["N"] // world.plot["row"] + 1, world.plot["row"])
    world.plot["axs"] = axs
    # prepare the paprameters for realTime experiment iteration
    world.data["parameters"] = dict(deltaT = world.data["T"], h = world.data["sensor_height"], f = world.data["F"],
                                    preImg = None, preV = 0.0, V_discard = 20,
    W_discard = 5.0,
    V_noise = 0.05,
    W_noise = 0.01,
    preVE = 0.0,
    preWE = 0.0,
    preW = 0.0,
    history = world.data["opV"],
    mode = "onlyV",
    plot_tool = world.plot,
    with_std = False)

def play_game(mode):
    # prepare the world and the client
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    world.actor_list = []
    world.mode = mode
    client.set_timeout(10.0)
    
    world.data = {}
    if world.mode == "data":
        prepareData(world)
    if world.mode == "realTime":
        prepareExp(world)
    
    ego = spawn_vehicle(world)
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

        if judge_end(world):
            break
    
    clear_world(world)
    if world.mode == "data":
        cur_path = os.getcwd()
        SE_root = os.path.dirname(cur_path)
        path_to_save =  SE_root + "/VideoSet/" + "ladder2.pkl"
        saveData(world.data, path_to_save)

    if world.mode == "realTime":
        # now plot the data
        realVs = world.data["realV"]
        opVs = world.data["opV"]
        plt.plot(realVs, label = "realV")
        # plt.plot(opVs, label = "opV")
        plt.plot(world.data["V_std"], label = "V_std")
        plt.plot(world.data["filtered_V"], label = "filtered_V")
        plt.legend()
        plt.show()
    # du.random_image_test(world.data["images"])

def saveData(data, path):
    with open(path, "wb") as file:
        pickle.dump(data, file)

def __main__():
    play_game(mode = "data")

if __name__ == "__main__":
    __main__()