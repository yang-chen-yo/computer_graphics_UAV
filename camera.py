import glm
import math

class Camera:
    def __init__(self, position=glm.vec3(0.0, 10.0, 30.0), up=glm.vec3(0.0, 1.0, 0.0), yaw=-90.0, pitch=-20.0):
        self.position = position
        self.world_up = up
        self.yaw = yaw
        self.pitch = pitch
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = up
        self.right = glm.normalize(glm.cross(self.front, self.world_up))
        self.update_camera_vectors()

    def process_mouse_movement(self, xoffset, yoffset, sensitivity=0.1):
        xoffset *= sensitivity
        yoffset *= sensitivity

        self.yaw += xoffset
        self.pitch += yoffset

        # 限制 pitch
        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0

        self.update_camera_vectors()

    def update_camera_vectors(self):
        # 計算前方向量
        front = glm.vec3()
        front.x = math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        front.y = math.sin(math.radians(self.pitch))
        front.z = math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        self.front = glm.normalize(front)
        # 重新計算右向量和上向量
        self.right = glm.normalize(glm.cross(self.front, self.world_up))
        self.up = glm.normalize(glm.cross(self.right, self.front))

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.front, self.up)
