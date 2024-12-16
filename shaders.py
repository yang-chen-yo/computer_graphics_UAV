from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram

def compile_shader(vertex_src, fragment_src):
    """
    Compiles vertex and fragment shaders and links them into a program.
    """
    try:
        vertex_shader = compileShader(vertex_src, GL_VERTEX_SHADER)
        fragment_shader = compileShader(fragment_src, GL_FRAGMENT_SHADER)
        shader_program = compileProgram(vertex_shader, fragment_shader)
        return shader_program
    except RuntimeError as e:
        print(f"Shader compilation error: {e}")
        return None
