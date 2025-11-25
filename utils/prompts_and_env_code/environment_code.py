import os

script_dir = os.path.dirname(os.path.abspath(__file__))
walker_path = os.path.join(script_dir, "env_code_txt_files", "env_code_Walker_2d_v5.txt")
humanoid_path = os.path.join(script_dir, "env_code_txt_files", "env_code_humanoid_v5.txt")

with open(walker_path, "r", encoding="utf-8") as f:
    walker_2d_v5_code = f.read()

walker_2d_v5_description = """The walker is a two-dimensional bipedal robot consisting of seven main body parts - 
a single torso at the top (with the two legs splitting after the torso), two thighs in the middle below the torso, 
two legs below the thighs, and two feet attached to the legs on which the entire body rests. The goal is to walk in the 
forward (right) direction by applying torque to the six hinges connecting the seven body parts."""

with open(humanoid_path, "r", encoding="utf-8") as f:
    Humanoid_v5_code = f.read()

Humanoid_v5_description = ("The 3D bipedal robot is designed to simulate a human. It has a torso (abdomen) with a pair "
                           "of legs and arms, and a pair of tendons connecting the hips to the knees."
                           " The legs each consist of three body parts (thigh, shin, foot), and the arms consist of "
                           "two body parts (upper arm, forearm). The goal of the environment is to walk forward "
                           "as fast as possible without falling over.")
