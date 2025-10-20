def return_reward_string_placeholder(query):
    example_return_from_llm = """A first example for a reward function could be ```def walker2d_less_speed(observation, action, original_reward, terminated, truncated, info):
    forward_vel = info.get('x_velocity', 0.0)
    return original_reward + 0 * forward_vel``` some more text.... ```def walker2d_less_speed(observation, action, original_reward, terminated, truncated, info):
    forward_vel = info.get('x_velocity', 0.0)
    return original_reward + 0 * forward_vel``` some more text.... ```def walker2d_less_speed(observation, action, original_reward, terminated, truncated, info):
    forward_vel = info.get('x_velocity', 0.0)
    return original_reward + 0 * forward_vel``` some more text.... ```def walker2d_less_speed(observation, action, original_reward, terminated, truncated, info):
    forward_vel = info.get('x_velocity', 0.0)
    return original_reward + 0 * forward_vel``` some more text.... ```def walker2d_less_speed(observation, action, original_reward, terminated, truncated, info):
    forward_vel = info.get('x_velocity', 0.0)
    return original_reward + 0 * forward_vel``` some more text.... ```def walker2d_less_speed(observation, action, original_reward, terminated, truncated, info):
    forward_vel = info.get('x_velocity', 0.0)
    return original_reward + 0 * forward_vel``` the next one is formatted incorrectly.... ```def walker2d_less_speed(observation, action, original_reward, terminated, truncated, info):
    forward_vel = info.get('x_velocity', 0.0)
    mistake original_reward + 0 * forward_vel``` the next one is formatted incorrectly.... ```def walker2d_less_speed(observation, action, original_reward, terminated, truncated, info):
    forward_vel = info.get('x_velocity', 0.0)
    mistake original_reward + 0 * forward_vel``` some more text.... ```def walker2d_less_speed(observation, action, original_reward, terminated, truncated, info):
    forward_vel = info.get('x_velocity', 0.0)
    return original_reward + 0 * forward_vel``` some more text.... ```def walker2d_less_speed(observation, action, original_reward, terminated, truncated, info):
    forward_vel = info.get('x_velocity', 0.0)
    return original_reward + 0 * forward_vel``` some more text.... ```def walker2d_less_speed(observation, action, original_reward, terminated, truncated, info):
    forward_vel = info.get('x_velocity', 0.0)
    return original_reward + 0 * forward_vel``` some more text.... ```def walker2d_less_speed(observation, action, original_reward, terminated, truncated, info):
    forward_vel = info.get('x_velocity', 0.0)
    return original_reward + 0 * forward_vel``` some more text.... ```def walker2d_less_speed(observation, action, original_reward, terminated, truncated, info):
    forward_vel = info.get('x_velocity', 0.0)
    return original_reward + 0 * forward_vel``` some more text.... ```def walker2d_less_speed(observation, action, original_reward, terminated, truncated, info):
    forward_vel = info.get('x_velocity', 0.0)
    return original_reward + 0 * forward_vel``` some more text.... ```def walker2d_less_speed(observation, action, original_reward, terminated, truncated, info):
    forward_vel = info.get('x_velocity', 0.0)
    return original_reward + 0 * forward_vel``` some more text.... ```def walker2d_less_speed(observation, action, original_reward, terminated, truncated, info):
    forward_vel = info.get('x_velocity', 0.0)
    return original_reward + 0 * forward_vel``` some text to end with"""
    return example_return_from_llm


def reflect(fitness_score_string):
    return "reflection"
