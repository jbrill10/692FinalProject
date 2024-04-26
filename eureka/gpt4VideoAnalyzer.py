import cv2
import numpy as np
import base64
import requests

import os
import glob

# Convert video into a collection of frames
def extract_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total Frames:", total_frames)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int16).tolist()
    print("Selected Frames:", frame_indices)
    frames = []
    output_path = "frame_outputs/"
    os.makedirs(output_path, exist_ok=True)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            #cv2.imshow("video", frame)
            save_path = output_path + "frame"+str(idx)+".jpeg"
            cv2.imwrite(save_path, frame)
            frames.append(save_path)
        else:
            print(f"Failed to read frame {idx}")

    return frames

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Create a base64 payload
def create_image_payload(video_path, num_frames):
    frame_data = extract_frames(video_path, num_frames)
    if not frame_data:
        return None
    content = []
    for i in range(num_frames):
        base64_image = encode_image(frame_data[i])
        data = {"type":"image_url",
                "image_url":{
                    "url":f"data:image/jpeg;base64,{base64_image}"
                    }
                }
        content.append(data)
    
    return content


def generate_full_payload(model, prompt, video_path, num_frames, max_tokens):
    payload = {}
    payload["model"] = model
    payload_message = [{"role":"user"}]
    payload_content = [{
          "type": "text",
          "text": prompt
        }]
    image_data = create_image_payload(video_path, num_frames)
    if image_data is None:
        return None
    for data in image_data:
        payload_content.append(data)
    payload_message[0]["content"] = payload_content
    payload["messages"] = payload_message
    payload["max_tokens"] = max_tokens

    return payload


def get_most_recent_video(base_dir="/home/creighton/Eureka/eureka"):
    # Get a list of all dated folders in outputs/eureka
    dated_folders = os.listdir(os.path.join(base_dir, "outputs", "eureka"))
    if not dated_folders:
        return None

    # Sort the dated folders in descending order
    dated_folders.sort(reverse=True)

    for dated_folder in dated_folders:
        # Get a list of all policy folders within the current dated folder
        policy_folders = glob.glob(os.path.join(base_dir, "outputs", "eureka", dated_folder, "policy-*"))
        if not policy_folders:
            continue

        # Sort the policy folders in descending order
        policy_folders.sort(reverse=True)

        # Iterate through the policy folders until a video file is found
        for policy_folder in policy_folders:
            video_files = glob.glob(os.path.join(policy_folder, "videos", "HumanoidGPT_*", "rl-video-step-*.mp4"))
            if video_files:
                return max(video_files)

    # No video files found in any dated folder
    return None


def get_ai_feedback(prompt_type="feedback", 
                    human_video_path="human_running_1.mp4", 
                    api_key="key-here",
                    model = "gpt-4-vision-preview"):

    #descriptor_prompt = "These are some frames from a video showing a human performing a particular motion.\n State what the target motion is (do not specifiy if the person is in-place or moving).\n Another AI agent is attempting to replicate this motion. Provide a description of how the motion is being performed by commenting on how the person's limbs are moving. Do not comment on the scene itself or the person's characteristics, and provide aggregate information instead of providing information about each frame. Only describe the motion in detail based on how the individual body parts are moving"

    descriptor_prompt = "These are some frames from a video showing a human performing a particular motion.\n State what the target motion is (do not specifiy if the person is in-place or moving).\n Another AI agent is attempting to replicate this motion. Provide a description of how the motion is being performed by commenting on how the person's key limbs and joints are moving so that the AI can replicate it. Do not comment on the scene itself or the person's characteristics, and provide aggregate information instead of providing information about each frame. Only describe the motion in detail based on how the individual body parts are moving"


    simulation_prompt = "These are some frames from a video of an AI agent attempting to " 

    # human_description = """
    # The legs alternate between a stance phase and a swing phase. During the stance phase, one foot is in contact with the ground, providing support and propulsion. The knee is generally slightly bent upon initial contact, then extends during the support phase, and flexes again to prepare for the swing phase.\n- The swing phase involves the leg that is not in contact with the ground swinging forward to prepare for its stance phase. This swinging leg moves through hip flexion followed by knee flexion to advance the leg with minimal effort and aerodynamic resistance.\n- The arms are bent at the elbows and move in a reciprocal motion with the legs. As one leg moves forward during its swing phase, the opposite arm swings forward, and vice versa. This arm movement helps to balance the body and maintain momentum.\n- The torso remains upright with a slight forward lean. This position helps to maintain balance and facilitate forward motion.\n- The head is held up, generally looking forward to anticipate terrain and maintain balance.\n\nThe movement is rhythmic, and each cycle of leg and arm movement constitutes one stride.
    # """

    # human_description = """
    # Throughout the motion, the arms alternate with the legs, moving back and forth in a rhythmic pattern.
    # The elbows are bent, and the arms swing approximately from the height of the hips to the chest.
    # The hands maintain a relaxed, partially closed position.
    # Simultaneously, the legs alternate in a running action, with one leg bending at the knee and lifting towards the torso while the other extends back down and slightly behind.
    # This action resembles the motion of running, with the knee driving upwards and then the foot pushing down, creating the dynamic movement of running.
    # The elevation of the knee seems to be fairly high, consistent with a jogging or running motion. The torso maintains an upright position, and the head faces forward throughout the cycle of movement.
    # """

    human_description = """
    During the running motion, the person alternates extending one leg forward while the other leg pushes off the ground from behind. The forward leg bends at the knee as it comes up to prepare for the foot to make contact with the ground, while the back leg extends as it pushes off, contributing to forward propulsion. The person's arms are also in motion, swinging opposite to the legs to help maintain balance and momentum; when the left leg is forward, the right arm swings forward and the left arm swings back, and vice versa for the opposite side. The torso is generally upright and slightly leaning forward to aid in the forward motion.\n\nThe movement of the limbs in this sequence is continuous and there is a rhythm established by the alternating positions of the legs and arms. This type of motion involves a complex coordination of various muscle groups and body mechanics to achieve fluid, efficient running.
    """

    # human_description = """
    # The target motion being performed appears to be a walking sequence accompanied by the action of holding a phone to the ear with one hand.\n\nTo replicate this motion, the AI should consider the following key movements:\n\nThe gait consists of a step-by-step sequence involving a transfer of weight from one foot to the other. The swing leg moves forward, with the knee bending to raise the foot slightly off the ground while the stance leg supports the body's weight. As the swing leg moves forward, the foot eventually comes down, contacting the floor typically with the heel first, rolling through the foot and pushing off with the toes. This is repeated in an alternating pattern for each leg.\n\nConcurrently, the upper body maintains an upright posture, and the shoulders should have a slight counter-rotational movement to the legs. The arms swing gently in opposition to the legs, which means when one leg steps forward, the opposite arm swings forward, and the other arm swings back, altogether facilitating balance and momentum.\n\nThe arm being used to hold the phone should lift at the shoulder, bending the elbow at approximately a right angle to bring the hand close to the ear as if holding a phone. That arm maintains this position steadily during the walking motion, and the head may tilt slightly towards the shoulder that is holding the phone, simulating the act of pressing the phone against the ear.
    # """

    goal_prompt = "\nThe original motion was described as follows:" + human_description

    target_prompt = "\nCritique the motion of the AI agent by providing a description about how the simulations movements match and vary from the original motion's description. Do not comment on the scene or the agent, just focus on the motion of the agent. Direction of motion does not matter, only how the limbs move."

    # Get policy rollout video most recently saved by eureka training. 
    # Only use when running one eureka experiment at a time! 
    simulation_video_path = get_most_recent_video()

    num_frames = 10 # 5
    max_tokens = 300

    action_name = "run like a human"

    if prompt_type=="feedback":
        input_prompt = simulation_prompt + action_name + goal_prompt + target_prompt
        video_path = simulation_video_path
        print("Video path: ", video_path)


    else:
        input_prompt = descriptor_prompt
        video_path = human_video_path

    # Auth header:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    
    payload = generate_full_payload(model, input_prompt, video_path, num_frames, max_tokens)
    if payload is None:
        return None
    print("Posting to OAI")
    response_json = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    print("======")

    response = response_json["choices"][0]["message"]["content"]

    print(response)

    with open("gpt4v_responses.txt", "a") as f:
        f.write(response)
        f.write("\n")

    return response

if __name__ == "__main__":
    get_feedback()