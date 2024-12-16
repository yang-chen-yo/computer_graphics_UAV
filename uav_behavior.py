# uav_behavior.py

import glm

def cohesion_force(u, swarm):
    perception_radius = 5.0
    steering = glm.vec3(0.0, 0.0, 0.0)
    total = 0
    for other in swarm:
        if other != u and glm.length(other.position - u.position) < perception_radius:
            steering += other.position
            total += 1
    if total > 0:
        steering /= total
        steering = glm.normalize(steering - u.position)
        steering *= 0.05  # 聚合力的強度
    return steering

def separation_force(u, swarm):
    perception_radius = 2.5  # 增加感知範圍
    steering = glm.vec3(0.0, 0.0, 0.0)
    total = 0
    for other in swarm:
        distance = glm.length(other.position - u.position)
        if other != u and distance < perception_radius:
            diff = u.position - other.position
            if distance != 0:
                diff /= distance  # 避免除以零
            steering += diff
            total += 1
    if total > 0:
        steering /= total
        steering = glm.normalize(steering)
        steering *= 0.2  # 增加分離力的強度
    return steering


def alignment_force(u, swarm):
    perception_radius = 5.0
    steering = glm.vec3(0.0, 0.0, 0.0)
    total = 0
    for other in swarm:
        if other != u and glm.length(other.position - u.position) < perception_radius:
            steering += other.velocity
            total += 1
    if total > 0:
        steering /= total
        steering = glm.normalize(steering) * 0.05  # 對齊力的強度
    return steering
