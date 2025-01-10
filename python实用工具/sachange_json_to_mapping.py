# import json

# # JSON字符串
# json_string = '''
# {
#   "task_cohesion": {
#     "task_sequence": [
#       "move-hand(drawer_bottom_knob)",
#       "grasp-object()",
#       "open_by_slide(pulling)",
#       "release-object()",
#       "detach_from_plane(blue_cube)",
#       "move_hand(drawer)",
#       "attach_to_plane()",
#       "release_object()"
#     ]
#   }
# }
# '''
# # load JSON
# json_data = json.loads(json_string)

# # GET task_sequence
# task_sequence = json_data["task_cohesion"]["task_sequence"]

# for task in task_sequence:
#     print(task)

import json

# 新的 JSON 字符串
json_string = '''
{
  "task_cohesion": {
    "task_sequence": [
      "move-hand(drawer_bottom_knob)",
      "grasp-object()",
      "open_by_slide(pulling)",
      "release-object()",
      "detach_from_plane(blue_cube)",
      "move_hand(drawer)",
      "attach_to_plane()",
      "release_object()"
    ]
  }
}
'''

# 模拟一些函数
def move_hand(location):
    print(f"Executing move_hand({location})")

def grasp_object():
    print("Executing grasp_object()")

def open_by_slide(action):
    print(f"Executing open_by_slide({action})")

def release_object():
    print("Executing release_object()")

def detach_from_plane(object_name):
    print(f"Executing detach_from_plane({object_name})")

def attach_to_plane():
    print("Executing attach_to_plane()")

# 解析新的 JSON 字符串
json_data = json.loads(json_string)

# 提取新的 task_sequence
task_sequence = json_data["task_cohesion"]["task_sequence"]

# 定义一个字典，将字符串动作映射到对应的函数
action_mapping = {
    "move-hand": move_hand,
    "grasp-object": grasp_object,
    "open_by_slide": open_by_slide,
    "release-object": release_object,
    "detach_from_plane": detach_from_plane,
    "move_hand": move_hand,
    "attach_to_plane": attach_to_plane,
    "release_object": release_object
}

# 按照顺序调用对应的函数
for task in task_sequence:
    action_parts = task.split("(")
    action_name = action_parts[0]
    action_parameter = action_parts[1][:-1] if len(action_parts) > 1 else None

    if action_name in action_mapping:
        if action_parameter:
            action_mapping[action_name](action_parameter)
        else:
            action_mapping[action_name]()
    else:
        print(f"Unknown action: {action_name}")
