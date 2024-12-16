import glm
import random

class Particle:
    def __init__(self, pos, vel, life):
        self.position = glm.vec3(pos)
        self.velocity = glm.vec3(vel)
        self.life = life

class ParticleSystem:
    def __init__(self):
        self.particles = []

    def spawn_effect(self, center):
        for _ in range(20):
            offset = glm.vec3(random.uniform(-0.5, 0.5),
                             random.uniform(-0.5, 0.5),
                             random.uniform(-0.5, 0.5))
            vel = glm.vec3(random.uniform(-1.0, 1.0),
                           random.uniform(0.5, 2.0),
                           random.uniform(-1.0, 1.0))
            p = Particle(center + offset, vel, 1.0)
            self.particles.append(p)

    def spawn_particles_at_positions(self, positions):
        """
        在指定的位置上生成粒子。
        """
        for pos in positions:
            p = Particle(pos, glm.vec3(0.0, 0.0, 0.0), 5.0)  # 粒子生命週期可根據需要調整
            self.particles.append(p)

    def update(self, dt):
        alive = []
        for p in self.particles:
            p.position += p.velocity * dt
            p.life -= dt
            p.velocity.y -= dt * 0.5  # 重力效果
            if p.life > 0:
                alive.append(p)
        self.particles = alive

    def get_positions(self):
        return [p.position for p in self.particles]
