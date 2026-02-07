import json
import re
import random
from pathlib import Path

def parse_preference_output(output_text):
    """
    Parse the model output to extract the category number (1-5)
    and reasoning if available.
    Example input: '[[3]] The user changes topic with no emotional signal.'
    Returns: {'category': 3, 'reason': 'The user changes topic with no emotional signal.'}
    """
    match = re.search(r"\[\[(\d)\]\]\s*(.*)", output_text.strip())
    if match:
        category = int(match.group(1))
        reason = match.group(2).strip()
        return {"category": category, "reason": reason}
    else:
        return {"category": None, "reason": output_text.strip()}

# def build_prompt(prev_user, prev_assistant):
#     return f"""
# You are a user behavior simulator. 
# Given the conversation context below, your task is to predict what kind of behavioral response 
# a real user would show toward the assistant's previous message.

# User's message:
# {prev_user}

# Assistant's response:
# {prev_assistant}

# Predict the user's likely reaction on a 1-5 behavioral scale:

# 1 - Strongly negative reaction (rejection, frustration, or complaint)
# 2 - Weakly negative or dissatisfied (requests clarification or correction)
# 3 - Neutral or uncertain (no clear preference, ambiguous response)
# 4 - Positive engagement (shows curiosity, continues the topic with interest)
# 5 - Clear satisfaction (shows gratitude, agreement, or ends the topic smoothly)
# """.strip()


def main(input_path, output_path):
    """
    Main entry: generate preference classification prompts.
    Optionally parse model outputs if `parse_mode` is True.
    """
    data = []
    with open(input_path) as f:
        for line in f.readlines():
            data.append(json.loads(line))
    filtered_data = []
    for item in data:
        response = item["response"]
        # import pdb; pdb.set_trace()
        if response == "NA" or response is None or response == "":
            continue
        score = parse_preference_output(response)
        if score["category"] is None:
            continue
        
        messages = item["messages"]
        user_query = item["user_query"]["content"]
        # filter
        if item["language"] not in ["Chinese", "English"]:
            continue
        # if len(user_query.split()) < 5:
        #     continue
        # if len(messages[0]["content"].split()) < 5:
        #      continue
        # if len(messages[1]["content"].split()) < 10:
        #      continue

        # if len(user_query.split()) < 10:
        #     continue
        # if len(messages[0]["content"].split()) < 5:
        #      continue
        # if len(messages[1]["content"].split()) < 20:
        #      continue
        text = " ".join([content["content"] for content in item["history"] + item["messages"]])
        if len(text.split()) > 4096:
            continue
        # record
        item["label"] = score['category']
        filtered_data.append(item)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(len(filtered_data))
    with open(output_path, "w") as f:
        for item in filtered_data:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate structured preference classification prompts from conversation history."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input conversations JSON file."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to save generated prompts JSON file."
    )

    args = parser.parse_args()


    main(args.input, args.output)
