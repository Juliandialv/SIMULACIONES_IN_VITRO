"""Test module"""
import open3d as o3d

og_mesh = o3d.io.read_triangle_mesh("./data/raw/modelsSynth/P/P_3_25.obj")
sm_mesh = og_mesh.subdivide_loop(number_of_iterations=2)
o3d.io.write_triangle_mesh("./data/raw/P_3_25_sm.obj", sm_mesh)
