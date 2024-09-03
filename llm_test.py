import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from tqdm import tqdm
import json
import torch.nn.functional as F


login(token="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")


model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")


input_paragraph = """
When the host of an active meeting invites a contact directly to join, the invitee can send a message when declining the invite, to provide context around the reason for declining. Some preset responses are provided by Zoom, but custom text can be used as well. The message will appear as a chat message in Zoom Team Chat for the person requesting you join.
"""

# Prompt for generating DOT code
prompt = f"Create a dot code script that generates a figure representing the following paragraph as a flowchart:\n{input_paragraph}\n\nThe flowchart should include decision points and actions clearly. It should keep the conditions in mind and have a detailed flowchart with the correct flow."


conversation = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
)


device = 'cuda'
inputs.to(model.device)
outputs = model.generate(**inputs, max_new_tokens=5000)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt_end_idx = output_text.find(prompt.strip()) + len(prompt.strip())
generated_text = output_text[prompt_end_idx:].strip()

#generated_text = "I\'m glad you\'d like to create a flowchart for the given scenario using Dot code. Here\'s a simple representation of the flowchart:\n\n```\ndigraph MeetingInvitation {\n    rankdir=LR;\n    node [shape=rectangle, style=filled, fillcolor=lightgray];\n\n    Host ["Host of Active Meeting"]\n    Invite ["Invites Contact"]\n    Accept ["Accepts Invitation"]\n    Decline ["Declines Invitation"]\n    Message ["Sends Message"]\n    Chat ["Appears as a Chat Message in Zoom Team Chat"]\n\n    Host -> Invite\n    Invite -> Accept [label="Accepts Invitation"]\n    Invite -> Decline [label="Declines Invitation"]\n    Decline -> Message [label="Sends a message"]\n    Message -> Chat [label="Appears as a Chat Message"]\n\n    Accept [shape=point, style=invis]\n    Decline [shape=point, style=invis]\n}\n```\n\nThis Dot code creates a graph with nodes representing the different steps in the process and edges connecting them to show the flow. The `Host`, `Invite`, `Accept`, `Decline`, `Message`, and `Chat` nodes represent the different steps in the process. The edges connecting the nodes show the flow from one step to the next.\n\nThe `Accept` and `Decline` nodes are set to be invisible points, as they don\'t have any specific actions associated with them. Instead, the flow moves directly to the next step based on the decision made at the `Decline` node.\n\nYou can visualize this Dot code using Graphviz, which is a popular tool for creating and viewing graphs and flowcharts."

from graphviz import Source

# Define the corrected Dot code
dot_code = '''
digraph MeetingInvitation {
    rankdir=LR;
    node [shape=rectangle, style=filled, fillcolor=lightgray];

    Host [label="Host of Active Meeting"]
    Invite [label="Invites Contact"]
    Accept [label="Accepts Invitation"]
    Decline [label="Declines Invitation"]
    Message [label="Sends Message"]
    Chat [label="Appears as a Chat Message in Zoom Team Chat"]

    Host -> Invite
    Invite -> Accept [label="Accepts Invitation"]
    Invite -> Decline [label="Declines Invitation"]
    Decline -> Message [label="Sends a message"]
    Message -> Chat [label="Appears as a Chat Message"]

    Accept [shape=point, style=invisible]
    Decline [shape=point, style=invisible]
}
'''

# Create a Source object
source = Source(dot_code)

# Render the graph to a file (e.g., PNG format) and display it
source.render('meeting_invitation', format='png', cleanup=False)
source.view()
