from argparse import ArgumentParser

from tasks.eval.model_utils import load_pllava
from tasks.eval.eval_utils import (
    ChatPllava,
    conv_plain_v1,
    Conversation,
    conv_templates
)

SYSTEM = """You are a powerful Video Magic ChatBot, a large vision-language assistant. 
You are able to understand the video content that the user provides and assist the user in a video-language related task.
The user might provide you with the video and maybe some extra noisy information to help you out or ask you a question. Make use of the information in a proper way to be competent for the job.
### INSTRUCTIONS:
1. Follow the user's instruction.
2. Be critical yet believe in yourself.
"""

INIT_CONVERSATION: Conversation = conv_plain_v1.copy()

def init_model(args):
    print('Initializing PLLaVA')
    model, processor = load_pllava(
        args.pretrained_model_name_or_path,
        args.num_frames,
        use_lora=args.use_lora,
        weight_dir=args.weight_dir,
        lora_alpha=args.lora_alpha)
    chat = ChatPllava(model, processor)
    return chat

def process_input(args, chat):
    chat_state = INIT_CONVERSATION.copy()
    img_list = []

    if args.video:
        llm_message, img_list, chat_state = chat.upload_video(args.video, chat_state, img_list, args.num_segments)
    elif args.image:
        llm_message, img_list, chat_state = chat.upload_img(args.image, chat_state, img_list)
    else:
        raise ValueError("You must provide either an image or video file.")

    return llm_message, chat_state, img_list

def get_response(chat, chat_state, img_list, question, num_beams, temperature):
    chat_state = chat.ask(question, chat_state, SYSTEM)
    llm_message, _, chat_state = chat.answer(
        conv=chat_state,
        img_list=img_list,
        max_new_tokens=200,
        num_beams=num_beams,
        temperature=temperature
    )
    return llm_message


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=False
                        , default="/mnt/disk4/mikecheung/model/llava-v1.6-vicuna-7b-hf")
    parser.add_argument("--num_frames", type=int, required=False, default=4)
    parser.add_argument("--use_lora", action='store_true')
    parser.add_argument("--weight_dir", type=str, required=False, default=None)
    parser.add_argument("--conv_mode", type=str, required=False, default="plain")
    parser.add_argument("--lora_alpha", type=int, required=False, default=None)
    parser.add_argument("--video", type=str, help="Path to the video file", default="video.mp4")
    parser.add_argument("--image", type=str, help="Path to the image file", default="llava_v1_5_radar.jpg")
    parser.add_argument("--question", type=str, help="Question to ask the model", required=False,
                        default="What is shown in this video?")
    parser.add_argument("--num_segments", type=int, default=8, help="Number of video segments")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search numbers")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    chat = init_model(args)
    INIT_CONVERSATION = conv_templates[args.conv_mode]

    llm_message, chat_state, img_list = process_input(args, chat)
    response = get_response(chat, chat_state, img_list, args.question, args.num_beams, args.temperature)

    print(f"Response: {response}")

##################################################################################


# chat_state = INIT_CONVERSATION.copy() if chat_state is None else chat_state
# img_list = [] if img_list is None else img_list
# if video:
# llm_message, img_list, chat_state = chat.upload_video(gr_video, chat_state, img_list, num_segments)
# llm_message, img_list, chat_state = chat.upload_img(gr_img, chat_state, img_list)
#
# chat_state = chat.ask(user_message, chat_state, system)
# chatbot = chatbot + [[user_message, None]]
#
# llm_message, llm_message_token, chat_state = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=200,
#                                                              num_beams=num_beams, temperature=temperature)
# llm_message = llm_message.replace("<s>", "")  # handle <s>
# chatbot[-1][1] = llm_message
# print(chat_state)
# print(f"Answer: {llm_message}")
# return chatbot, chat_state, img_list
