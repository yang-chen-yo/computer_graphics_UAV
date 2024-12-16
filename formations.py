import math
import glm

def generate_cube_positions(count, spacing=1.5):
    n = math.ceil(count ** (1.0/3.0))
    formation_positions = []
    start_offset = -(n - 1) * spacing * 0.5
    created = 0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if created < count:
                    pos = glm.vec3(start_offset + x * spacing,
                                   5.0 + start_offset + y * spacing,
                                   start_offset + z * spacing)
                    formation_positions.append(pos)
                    created += 1
                else:
                    break
    return formation_positions

def generate_sphere_positions(count, radius=10.0):
    formation_positions = []
    offset = 2.0 / count
    increment = math.pi * (3.0 - math.sqrt(5.0))  # 黃金角
    for i in range(count):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - y * y)
        phi = i * increment
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        formation_positions.append(glm.vec3(x * radius, y * radius + 5.0, z * radius))
    return formation_positions

def generate_cylinder_positions(count, radius=10.0, height=20.0):
    formation_positions = []
    layers = math.ceil(math.sqrt(count))
    per_layer = math.ceil(count / layers)
    dh = height / layers
    for i in range(layers):
        layer_y = 5.0 + (-height / 2.0 + i * dh)
        for j in range(per_layer):
            if len(formation_positions) >= count:
                break
            angle = (2 * math.pi * j) / per_layer
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            formation_positions.append(glm.vec3(x, layer_y, z))
    return formation_positions[:count]
