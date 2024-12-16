import glm
import random
import math

class UAV:
    def __init__(self, pos, vel):
        self.position = glm.vec3(pos)
        self.velocity = glm.vec3(vel)
        self.acceleration = glm.vec3(0.0, 0.0, 0.0)
        self.target_index = 0  # 新增目標索引
        
        # 超小且規整的漂浮相關參數
        self.float_amplitude = glm.vec3(
            random.uniform(0.01, 0.02),  # x軸振幅
            random.uniform(0.01, 0.02),  # y軸振幅
            random.uniform(0.01, 0.02)   # z軸振幅
        )
        self.float_frequency = glm.vec3(
            random.uniform(0.5, 0.7),  # x軸頻率
            random.uniform(0.5, 0.7),  # y軸頻率
            random.uniform(0.5, 0.7)   # z軸頻率
        )
        self.float_phase = glm.vec3(
            random.uniform(0, 2 * math.pi),  # x軸相位
            random.uniform(0, 2 * math.pi),  # y軸相位
            random.uniform(0, 2 * math.pi)   # z軸相位
        )

    def apply_force(self, force):
        self.acceleration += force

    def update(self, dt, max_speed, target_positions, current_time):
        if target_positions:
            target = target_positions[self.target_index]
            direction = target - self.position
            if glm.length(direction) < 0.1:
                self.target_index = (self.target_index + 1) % len(target_positions)
            else:
                steer = glm.normalize(direction) * max_speed
                self.velocity += steer * dt
        # 確保速度不超過最大速度
        speed = glm.length(self.velocity)
        if speed > max_speed:
            self.velocity = glm.normalize(self.velocity) * max_speed
        self.position += self.velocity * dt
        # 重置加速度
        self.acceleration = glm.vec3(0.0, 0.0, 0.0)

    def distance_to(self, other):
        return glm.length(self.position - other.position)
    
    def get_floating_offset(self, current_time):
        """
        計算漂浮偏移量，基於正弦波動作。
        """
        offset = glm.vec3(
            self.float_amplitude.x * math.sin(self.float_frequency.x * current_time + self.float_phase.x),
            self.float_amplitude.y * math.sin(self.float_frequency.y * current_time + self.float_phase.y),
            self.float_amplitude.z * math.sin(self.float_frequency.z * current_time + self.float_phase.z)
        )
        return offset
