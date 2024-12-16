# main.py

import sys
import math
import random
import ctypes
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import glfw
from OpenGL.GL import *
import glm
import cv2
import numpy as np

from shaders import compile_shader
from formations import generate_cube_positions, generate_sphere_positions, generate_cylinder_positions
from camera import Camera
from particles import ParticleSystem
from uav import UAV
from uav_behavior import cohesion_force, separation_force, alignment_force
from image_processing import load_image, edge_detection, get_edge_points

############################################
# Global Parameters
############################################

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

deltaTime = 0.0
lastFrame = 0.0

swarm = []
swarmSize = 200
maxSpeed = 5.0

BOUND_MIN = glm.vec3(-20.0, 0.0, -20.0)
BOUND_MAX = glm.vec3(20.0, 20.0, 20.0)

currentFormation = "random"
target_positions = []
transition_speed = 2.0
inTransition = False

initial_positions = []

# 定義全域變數
keys = {}
firstMouse = True
lastX = WINDOW_WIDTH / 2.0
lastY = WINDOW_HEIGHT / 2.0

# Initialize camera and particle_system as global variables
camera = None
particle_system = None

# Edge points extracted from image
image_edge_points = []

# 過渡相關變數
transition_duration = 5.0  # 過渡持續時間（秒），增加到5秒以確保過渡可見
transition_elapsed = 0.0    # 過渡已經過的時間
initial_positions_transition = []  # 過渡開始時的初始位置

# 新增狀態標誌
inFormation = False

############################################
# Functions
############################################

def ease_in_out(t):
    return t * t * (3 - 2 * t)

def init_swarm():
    global swarm, initial_positions, inTransition, currentFormation, inFormation
    swarm = []
    initial_positions = []
    for _ in range(swarmSize):
        pos = glm.vec3(random.uniform(-5, 5),
                      random.uniform(0, 10),
                      random.uniform(-5, 5))
        vel = glm.normalize(glm.vec3(random.uniform(-1, 1),
                                     random.uniform(-1, 1),
                                     random.uniform(-1, 1))) * 0.5
        uav = UAV(pos, vel)
        swarm.append(uav)
        initial_positions.append(glm.vec3(pos))
    currentFormation = "random"
    inTransition = False
    inFormation = False  # 初始化時不在固定隊形中
    print("Swarm initialized with random formation.")

def begin_transition(new_formation):
    global inTransition, target_positions, currentFormation, inFormation
    if new_formation == "cube":
        target_positions = generate_cube_positions(swarmSize)
    elif new_formation == "sphere":
        target_positions = generate_sphere_positions(swarmSize, radius=10.0)
    elif new_formation == "cylinder":
        target_positions = generate_cylinder_positions(swarmSize, radius=10.0, height=20.0)
    elif new_formation == "random":
        target_positions = initial_positions.copy()
    elif new_formation == "image":
        if not image_edge_points:
            print("No image edge points available for image formation.")
            return
        desired_count = swarmSize
        total_points = len(image_edge_points)

        if total_points > desired_count:
            step = total_points // desired_count
            selected_points = image_edge_points[::step][:desired_count]
        else:
            selected_points = image_edge_points.copy()
            while len(selected_points) < desired_count:
                selected_points += image_edge_points.copy()
            selected_points = selected_points[:desired_count]

        # 添加隨機偏移，避免目標位置重疊
        for i in range(len(selected_points)):
            offset = glm.vec3(random.uniform(-0.5, 0.5),  # 增加偏移量
                              random.uniform(-0.5, 0.5),
                              random.uniform(-0.5, 0.5))
            selected_points[i] += offset

        target_positions = selected_points

    currentFormation = new_formation
    inTransition = True
    inFormation = False  # 開始過渡，設置為 False

    # 保存過渡開始時的初始位置
    global transition_elapsed, initial_positions_transition
    transition_elapsed = 0.0
    initial_positions_transition = [glm.vec3(u.position) for u in swarm]

    # 生成過渡粒子特效
    for u in swarm:
        particle_system.spawn_effect(u.position)

    print(f"Transition to {new_formation} formation started.")

    # 打印部分目標位置以確認
    if new_formation == "image":
        print("Sample target positions for Image formation:")
        for i in range(5):
            if i < len(target_positions):
                distance = glm.length(target_positions[i] - initial_positions_transition[i])
                print(f"Target {i}: {target_positions[i]}, Start: {initial_positions_transition[i]}, Distance: {distance:.2f}")

def arrange_particles_on_image():
    global swarm, target_positions, inTransition, image_edge_points, inFormation, initial_positions_transition
    if not image_edge_points:
        print("No image edge points to arrange.")
        return

    print("Arranging particles to Image formation.")

    desired_count = swarmSize
    total_points = len(image_edge_points)

    if total_points > desired_count:
        step = total_points // desired_count
        selected_points = image_edge_points[::step][:desired_count]
    else:
        selected_points = image_edge_points.copy()
        while len(selected_points) < desired_count:
            selected_points += image_edge_points.copy()
        selected_points = selected_points[:desired_count]

    # 添加隨機偏移，避免目標位置重疊
    for i in range(len(selected_points)):
        offset = glm.vec3(random.uniform(-0.5, 0.5),  # 增加偏移量
                          random.uniform(-0.5, 0.5),
                          random.uniform(-0.5, 0.5))
        selected_points[i] += offset

    # 設定目標位置為圖片的點
    target_positions = selected_points
    inTransition = True
    inFormation = False

    # 保存過渡開始時的初始位置
    global transition_elapsed, initial_positions_transition
    transition_elapsed = 0.0
    initial_positions_transition = [glm.vec3(u.position) for u in swarm]

    print(f"Number of target positions set: {len(target_positions)}")

    # 產生特效
    for u in swarm:
        particle_system.spawn_effect(u.position)

    # 打印部分目標位置以確認
    print("Sample target positions:")
    for i in range(5):
        if i < len(target_positions):
            distance = glm.length(target_positions[i] - initial_positions_transition[i])
            print(f"Target {i}: {target_positions[i]}, Start: {initial_positions_transition[i]}, Distance: {distance:.2f}")

def update_swarm(dt):
    global inTransition, transition_elapsed, initial_positions_transition, inFormation
    if inTransition:
        transition_elapsed += dt
        t = min(transition_elapsed / transition_duration, 1.0)  # 計算插值比例，最大為1.0
        t = ease_in_out(t)  # 應用ease-in-ease-out

        # 調試輸出
        print(f"Transition Progress: t = {t:.2f}, transition_elapsed = {transition_elapsed:.2f}s")

        all_arrived = True
        for i, u in enumerate(swarm):
            if i >= len(target_positions):
                break
            start_pos = initial_positions_transition[i]
            target_pos = target_positions[i]

            # 線性插值
            new_pos = glm.mix(start_pos, target_pos, t)
            direction = target_pos - new_pos
            dist = glm.length(direction)
            if dist > 0.1:
                all_arrived = False

            u.position = new_pos

        # **在過渡期間應用分離行為**
        for u in swarm:
            separation = separation_force(u, swarm)
            u.apply_force(separation * 1.5)  # 增加分離力的權重

        if t >= 1.0 or all_arrived:
            inTransition = False
            inFormation = True  # 過渡完成，設置為 True
            print("Transition completed.")
    elif not inFormation:
        # 只有在不處於固定隊形時，才應用群體行為
        # Apply flocking behaviors: Cohesion, Separation, Alignment
        for u in swarm:
            cohesion = cohesion_force(u, swarm)
            separation = separation_force(u, swarm)
            alignment = alignment_force(u, swarm)
            u.apply_force(cohesion)
            u.apply_force(separation)
            u.apply_force(alignment)

        # Update UAVs
        current_time = glfw.get_time()
        for u in swarm:
            u.update(deltaTime, maxSpeed, target_positions, current_time)

        for u in swarm:
            keep_within_bounds(u)
    else:
        # 在固定隊形中，保持UAV在目標位置並添加漂浮偏移
        current_time = glfw.get_time()
        for i, u in enumerate(swarm):
            if i >= len(target_positions):
                break
            target_pos = target_positions[i]
            floating_offset = u.get_floating_offset(current_time)
            u.position = target_pos + floating_offset  # 添加漂浮偏移

def keep_within_bounds(u):
    if u.position.x < BOUND_MIN.x or u.position.x > BOUND_MAX.x:
        u.velocity.x = -u.velocity.x
    if u.position.y < BOUND_MIN.y or u.position.y > BOUND_MAX.y:
        u.velocity.y = -u.velocity.y
    if u.position.z < BOUND_MIN.z or u.position.z > BOUND_MAX.z:
        u.velocity.z = -u.velocity.z

def mouse_callback(window, xpos, ypos):
    global firstMouse, lastX, lastY, camera
    if firstMouse:
        lastX = xpos
        lastY = ypos
        firstMouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos  # Reversed since y-coordinates go from bottom to top
    lastX = xpos
    lastY = ypos

    camera.process_mouse_movement(xoffset, yoffset)

def key_callback(window, key, scancode, action, mods):
    global keys, inTransition, currentFormation, target_positions, image_edge_points, inFormation
    if action == glfw.PRESS:
        keys[key] = True
    elif action == glfw.RELEASE:
        keys[key] = False

    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    # Press F to form Cube
    if key == glfw.KEY_F and action == glfw.PRESS:
        print("F key pressed: Transition to Cube formation.")
        begin_transition("cube")

    # Press G to form Sphere
    if key == glfw.KEY_G and action == glfw.PRESS:
        print("G key pressed: Transition to Sphere formation.")
        begin_transition("sphere")

    # Press H to form Cylinder
    if key == glfw.KEY_H and action == glfw.PRESS:
        print("H key pressed: Transition to Cylinder formation.")
        begin_transition("cylinder")

    # Press R to reset swarm to initial random distribution
    if key == glfw.KEY_R and action == glfw.PRESS:
        print("R key pressed: Transition to Random formation.")
        begin_transition("random")

    # Press P to upload image and store edge points
    if key == glfw.KEY_P and action == glfw.PRESS:
        print("P key pressed: Uploading image.")
        upload_and_process_image()

    # **修改鍵 (如 O) 來開始過渡到圖片隊形**
    if key == glfw.KEY_O and action == glfw.PRESS:
        print("O key pressed: Attempting to transition to Image formation.")
        # 修改條件：只要不是正在過渡中，就允許開始新的過渡
        if image_edge_points and not inTransition:
            # 現在才開始過渡到圖片隊形
            begin_transition("image")
        else:
            if not image_edge_points:
                print("O key pressed but no image edge points available.")
            if inTransition:
                print("O key pressed but currently in transition.")
            # 移除 "already in formation" 的判斷
            # 因為我們允許從任何隊形過渡到圖片隊形，只要不在過渡中

def upload_and_process_image():
    global image_edge_points
    # 使用Tkinter的文件對話框選擇圖片
    Tk().withdraw()  # 不顯示主視窗
    image_path = askopenfilename(title="選擇圖片", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not image_path:
        print("未選擇任何圖片。")
        return
    print(f"選擇的圖片：{image_path}")

    # 加載並處理圖片
    try:
        img = load_image(image_path, max_width=800, max_height=800)
    except Exception as e:
        print(f"圖片加載失敗：{e}")
        return

    edges = edge_detection(img, low_threshold=100, high_threshold=200)
    if np.count_nonzero(edges) == 0:
        print("未檢測到任何邊緣。")
        return

    edge_points = get_edge_points(edges, scale=0.1)  # 根據需要調整scale
    if not edge_points:
        print("未提取到任何邊緣點。")
        return

    image_edge_points = edge_points
    print(f"提取到的邊緣點數量：{len(image_edge_points)}")

    # **這裡不立刻開始過渡**
    # 不呼叫 arrange_particles_on_image() 或 begin_transition("image")
    # 只儲存 image_edge_points，等待您需要的時候再開始過渡

    # 添加調試輸出確認點的分佈
    print("Sample edge points:")
    for i in range(5):
        if i < len(image_edge_points):
            print(f"Edge Point {i}: {image_edge_points[i]}")

def process_input(window):
    camera_speed = 20.0 * deltaTime
    if keys.get(glfw.KEY_W, False):
        camera.position += camera.front * camera_speed
    if keys.get(glfw.KEY_S, False):
        camera.position -= camera.front * camera_speed
    if keys.get(glfw.KEY_A, False):
        camera.position -= camera.right * camera_speed
    if keys.get(glfw.KEY_D, False):
        camera.position += camera.right * camera_speed
    if keys.get(glfw.KEY_Q, False):
        camera.position += camera.up * camera_speed
    if keys.get(glfw.KEY_E, False):
        camera.position -= camera.up * camera_speed

############################################
# Initialize GLFW and OpenGL
############################################

def main():
    global deltaTime, lastFrame, initial_positions, particle_system, camera

    if not glfw.init():
        print("Failed to initialize GLFW")
        sys.exit(-1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "UAV Swarm Show (Python)", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create GLFW window")
        sys.exit(-1)

    glfw.make_context_current(window)
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Compile shaders
    vertex_src = """
    #version 330 core
    layout(location=0) in vec3 aPos;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform vec3 uColor;

    out vec3 FragColor;

    void main(){
        FragColor = uColor;
        gl_Position = projection * view * model * vec4(aPos,1.0);
    }
    """

    fragment_src = """
    #version 330 core
    in vec3 FragColor;
    out vec4 color;

    void main(){
        color = vec4(FragColor,1.0);
    }
    """

    uav_shader = compile_shader(vertex_src, fragment_src)

    # Particle shaders
    particle_vertex_src = """
    #version 330 core
    layout(location=0) in vec3 aPos;

    uniform mat4 view;
    uniform mat4 projection;

    void main(){
        gl_Position = projection * view * vec4(aPos,1.0);
        gl_PointSize = 5.0;
    }
    """

    particle_fragment_src = """
    #version 330 core
    out vec4 FragColor;

    void main(){
        // Soft edges for particles
        float dist = length(gl_PointCoord - vec2(0.5));
        float alpha = 1.0 - smoothstep(0.4, 0.5, dist);
        FragColor = vec4(1.0, 1.0, 0.5, alpha);
    }
    """

    particle_shader = compile_shader(particle_vertex_src, particle_fragment_src)

    # Create cube VAO for UAV
    cubeVertices = [
        -0.2, -0.2, -0.2,
         0.2, -0.2, -0.2,
         0.2,  0.2, -0.2,
         0.2,  0.2, -0.2,
        -0.2,  0.2, -0.2,
        -0.2, -0.2, -0.2,

        -0.2, -0.2,  0.2,
         0.2, -0.2,  0.2,
         0.2,  0.2,  0.2,
         0.2,  0.2,  0.2,
        -0.2,  0.2,  0.2,
        -0.2, -0.2,  0.2,

        -0.2,  0.2,  0.2,
        -0.2,  0.2, -0.2,
        -0.2, -0.2, -0.2,
        -0.2, -0.2, -0.2,
        -0.2, -0.2,  0.2,
        -0.2,  0.2,  0.2,

         0.2,  0.2,  0.2,
         0.2,  0.2, -0.2,
         0.2, -0.2, -0.2,
         0.2, -0.2, -0.2,
         0.2, -0.2,  0.2,
         0.2,  0.2,  0.2,

        -0.2, -0.2, -0.2,
         0.2, -0.2, -0.2,
         0.2, -0.2,  0.2,
         0.2, -0.2,  0.2,
        -0.2, -0.2,  0.2,
        -0.2, -0.2, -0.2,

        -0.2,  0.2, -0.2,
         0.2,  0.2, -0.2,
         0.2,  0.2,  0.2,
         0.2,  0.2,  0.2,
        -0.2,  0.2,  0.2,
        -0.2,  0.2, -0.2
    ]

    # Convert to GLfloat array
    cubeVertices = (GLfloat * len(cubeVertices))(*cubeVertices)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)

    # Initialize camera
    camera = Camera()
    camera.position = glm.vec3(0.0, 15.0, 30.0)  # 調整相機位置以便更好地觀看過渡

    # Initialize swarm and particle system
    init_swarm()
    particle_system = ParticleSystem()

    # Store initial positions
    initial_positions = [glm.vec3(u.position) for u in swarm]

    print(f"After init_swarm: inFormation={inFormation}")

    while not glfw.window_should_close(window):
        currentFrame = glfw.get_time()
        deltaTime = currentFrame - lastFrame
        lastFrame = currentFrame

        glfw.poll_events()
        process_input(window)

        glClearColor(0.05, 0.05, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Update swarm behaviors
        update_swarm(deltaTime)

        # Update particles
        particle_system.update(deltaTime)

        # Get view and projection matrices
        view = camera.get_view_matrix()
        projection = glm.perspective(glm.radians(45.0), WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 100.0)

        # Draw UAVs
        glUseProgram(uav_shader)
        glUniformMatrix4fv(glGetUniformLocation(uav_shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(uav_shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))

        glBindVertexArray(VAO)
        for u in swarm:
            model = glm.mat4(1.0)
            model = glm.translate(model, u.position)
            glUniformMatrix4fv(glGetUniformLocation(uav_shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
            color = glm.vec3(0.8, 0.2, 1.0)
            glUniform3fv(glGetUniformLocation(uav_shader, "uColor"), 1, glm.value_ptr(color))
            glDrawArrays(GL_TRIANGLES, 0, 36)
        glBindVertexArray(0)

        # Draw particles
        glUseProgram(particle_shader)
        glUniformMatrix4fv(glGetUniformLocation(particle_shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(particle_shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
        particle_positions = particle_system.get_positions()
        if len(particle_positions) > 0:
            positions = []
            for pos in particle_positions:
                positions.extend([pos.x, pos.y, pos.z])
            positions = (GLfloat * len(positions))(*positions)

            particle_VAO = glGenVertexArrays(1)
            particle_VBO = glGenBuffers(1)
            glBindVertexArray(particle_VAO)
            glBindBuffer(GL_ARRAY_BUFFER, particle_VBO)
            glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(positions), positions, GL_DYNAMIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            glDrawArrays(GL_POINTS, 0, len(particle_positions))
            glBindVertexArray(0)
            glDeleteBuffers(1, [particle_VBO])
            glDeleteVertexArrays(1, [particle_VAO])

        glfw.swap_buffers(window)

    # Cleanup
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteProgram(uav_shader)
    glDeleteProgram(particle_shader)
    glfw.terminate()

if __name__ == "__main__":
    main()
