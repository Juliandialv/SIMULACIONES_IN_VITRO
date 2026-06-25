"""Test module"""
import open3d as o3d
from pathlib import Path

folder_path = Path("./data/raw/modelsSynth/P")

for file_path in folder_path.iterdir():
    if file_path.is_file():
        og_mesh = o3d.io.read_triangle_mesh(file_path)
        sm_mesh = og_mesh.subdivide_loop(number_of_iterations=2)
        o3d.io.write_triangle_mesh("./data/raw/" + file_path.stem + "_sm.obj" , sm_mesh)


#og_mesh = o3d.io.read_triangle_mesh("./data/raw/modelsSynth/P/P_8_12.obj")
#sm_mesh = og_mesh.subdivide_loop(number_of_iterations=2)
#o3d.io.write_triangle_mesh("./data/raw/P_8_12_sm.obj", sm_mesh)
