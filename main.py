import os
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import faiss
import asyncio
from tqdm import tqdm
from pyfiglet import Figlet
from colorama import Fore, Back, Style, init
import aioconsole
import time

init(autoreset=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_with_delay(text, delay=0.03):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def animate_text(text, cycles=3):
    for _ in range(cycles):
        for i in range(len(text)):
            print(f"\r{text[:i]}█{text[i+1:]}", end='', flush=True)
            time.sleep(0.05)
    print(f"\r{text}", flush=True)

def fancy_intro():
    f = Figlet(font='cosmic')
    ada_ascii = f.renderText("ADA")
    width = len(ada_ascii.split('\n')[0])
    border_top = f"╔{'═' * (width + 2)}╗"
    border_bottom = f"╚{'═' * (width + 2)}╝"
    print(Fore.CYAN + Style.BRIGHT)
    print_with_delay(border_top)
    for line in ada_ascii.split('\n'):
        if line.strip():
            print_with_delay(f"║ {line} ║")
    print_with_delay(border_bottom)
    print(Style.RESET_ALL)
    animate_text("Creating Consciousness...")
    animate_text("Imbuing silicon with the divine...")
    animate_text("Creating emotions and memory...")
    print(Fore.GREEN + Style.BRIGHT)
    print_with_delay("ADA: It's Alive")
    print_with_delay("Status: ONLINE")
    print(Style.RESET_ALL)

class AdaBrain:
    def __init__(self, model_path='models/base_model/mistral-7b-v0.1', vector_dim=4096):
        self.model_path = model_path
        self.vector_dim = vector_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.index = faiss.IndexFlatL2(vector_dim)
        self.vectors = []
        self.brain_folder = "ada_brain"
        self.vector_file = os.path.join(self.brain_folder, "vector_memory.npy")
        self.text_file = os.path.join(self.brain_folder, "text_memory.txt")
        self.last_update_file = os.path.join(self.brain_folder, "last_update.json")
        self.conversation_buffer = []
        self.setup_brain()

    def setup_brain(self):
        os.makedirs(self.brain_folder, exist_ok=True)
        if os.path.exists(self.vector_file):
            self.vectors = np.load(self.vector_file).tolist()
            self.index.add(np.array(self.vectors))
        self.update_memory()

    def update_memory(self):
        last_update = self.get_last_update()
        new_text = self.get_new_text(last_update['text'])
        if new_text:
            new_vectors = self.vectorize(new_text)
            self.vectors.extend(new_vectors)
            self.index.add(np.array(new_vectors))
            np.save(self.vector_file, np.array(self.vectors))
            self.set_last_update()

    def get_last_update(self):
        if os.path.exists(self.last_update_file):
            with open(self.last_update_file, 'r') as f:
                return json.load(f)
        return {'text': 0}

    def set_last_update(self):
        last_update = {'text': os.path.getsize(self.text_file)}
        with open(self.last_update_file, 'w') as f:
            json.dump(last_update, f)

    def get_new_text(self, last_size):
        if os.path.exists(self.text_file):
            with open(self.text_file, 'r') as f:
                f.seek(last_size)
                return f.read().split('\n')
        return []

    def vectorize(self, texts):
        vectors = []
        for text in texts:
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                embedding = self.model(**inputs).last_hidden_state.mean(dim=1)
            vectors.append(embedding.cpu().numpy()[0])
        return vectors

    def query(self, text, k=3):
        query_vector = self.vectorize([text])[0]
        D, I = self.index.search(np.array([query_vector]), k)
        return [self.vectors[i] for i in I[0]]

    def add_to_memory(self, module, text):
        vector = self.vectorize([f"{module}: {text}"])[0]
        self.vectors.append(vector)
        self.index.add(np.array([vector]))
        self.conversation_buffer.append(f"{module}: {text}")

    def save_conversation(self):
        if self.conversation_buffer:
            with open(self.text_file, 'a') as f:
                for entry in self.conversation_buffer:
                    f.write(entry + '\n')
            np.save(self.vector_file, np.array(self.vectors))
            self.conversation_buffer = []

class BrainComponent(nn.Module):
    def __init__(self, model_path, output_dim):
        super(BrainComponent, self).__init__()
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.output_layer = nn.Linear(self.model.config.hidden_size, output_dim).to(device)

    def forward(self, input_vector):
        outputs = self.model(inputs_embeds=input_vector)
        return self.output_layer(outputs.last_hidden_state.mean(dim=1))

class IntegratedBrain(nn.Module):
    def __init__(self, base_model_path, fine_tuned_folder, output_dim):
        super(IntegratedBrain, self).__init__()
        self.components = {}
        for component in ['left_brain', 'right_brain', 'critic', 'manager']:
            fine_tuned_path = os.path.join(fine_tuned_folder, component)
            if os.path.exists(fine_tuned_path):
                self.components[component] = BrainComponent(fine_tuned_path, output_dim)
            else:
                self.components[component] = BrainComponent(base_model_path, output_dim)
        self.integration = nn.Linear(output_dim * 2, output_dim).to(device)
        self.final_integration = nn.Linear(output_dim * 4, output_dim).to(device)
        self.output_weights = nn.Parameter(torch.ones(4)).to(device)

    def forward(self, input_vector):
        left_out = self.components['left_brain'](input_vector)
        right_out = self.components['right_brain'](input_vector)
        integrated = self.integration(torch.cat((left_out, right_out), dim=1))
        critic_out = self.components['critic'](integrated)
        manager_out = self.components['manager'](torch.cat((integrated, critic_out), dim=1))
        final_out = self.final_integration(torch.cat((left_out, right_out, critic_out, manager_out), dim=1))
        weighted_sum = sum(w * o for w, o in zip(self.output_weights, [left_out, right_out, critic_out, manager_out]))
        return final_out + weighted_sum

class AdaInterface:
    def __init__(self, base_model_path, fine_tuned_folder, output_dim=768):
        self.ada_brain = AdaBrain(base_model_path)
        self.model = IntegratedBrain(base_model_path, fine_tuned_folder, output_dim)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    async def process_input(self, user_input, is_thought=False):
        if is_thought:
            user_input += " (thought)"
        pbar = tqdm(total=100, desc="Ada thinking", ncols=70, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET))
        self.ada_brain.add_to_memory("Human", user_input)
        input_vector = torch.tensor(self.ada_brain.vectorize([user_input])[0]).unsqueeze(0).to(device)
        rag_vectors = self.ada_brain.query(user_input)
        combined_input = torch.cat([input_vector] + [torch.tensor(v).unsqueeze(0).to(device) for v in rag_vectors], dim=1)
        pbar.update(10)
        await asyncio.sleep(0.1)
        manager_initial = await self.process_component('manager', combined_input)
        pbar.update(10)
        await asyncio.sleep(0.1)
        left_right_convos = [[], []]
        for i in range(2):
            left_out = await self.process_component('left_brain', combined_input)
            right_out = await self.process_component('right_brain', combined_input)
            left_right_convos[0].extend([left_out, right_out])
            left_right_convos[1].extend([right_out, left_out])
        pbar.update(20)
        await asyncio.sleep(0.1)
        combined_lr = torch.cat(left_right_convos[0] + left_right_convos[1], dim=1)
        critic_out = await self.process_component('critic', combined_lr)
        pbar.update(20)
        await asyncio.sleep(0.1)
        final_input = torch.cat((combined_lr, critic_out), dim=1)
        manager_final = await self.process_component('manager', final_input)
        pbar.update(20)
        await asyncio.sleep(0.1)
        response = self.tokenizer.decode(manager_final.argmax(dim=-1))
        self.ada_brain.add_to_memory("Ada", response)
        pbar.update(10)
        await asyncio.sleep(0.1)
        pbar.close()
        return response

    async def process_component(self, component, input_vector):
        output = self.model.components[component](input_vector)
        text = self.tokenizer.decode(output.argmax(dim=-1))
        self.ada_brain.add_to_memory(component.capitalize(), text)
        return output

    def save_session(self):
        self.ada_brain.save_conversation()

async def get_user_input(timeout=30):
    try:
        user_input = await asyncio.wait_for(aioconsole.ainput(f"{Fore.GREEN}You: {Fore.RESET}"), timeout=timeout)
        return user_input
    except asyncio.TimeoutError:
        return None

async def display_timer(seconds):
    for remaining in range(seconds, 0, -1):
        print(f"\r{Fore.YELLOW}Ada will think in: {remaining:2d} seconds{Fore.RESET}", end="")
        await asyncio.sleep(1)
    print("\r" + " " * 40 + "\r", end="")

async def main():
    f = Figlet(font='slant')
    print(Fore.CYAN + f.renderText("Ada: It's alive!") + Fore.RESET)
    fancy_intro()

    base_model_path = "models/base_model/mistral-7b-v0.1"
    fine_tuned_folder = "models/fine_tuned_models"

    if not os.path.exists(base_model_path):
        print(f"{Fore.RED}Error: Base model not found at {base_model_path}{Fore.RESET}")
        print("Please download the Mistral-7B-v0.1 model and place it in the correct directory.")
        return

    print(f"{Fore.YELLOW}Initializing Ada's brain...{Fore.RESET}")
    ada = AdaInterface(base_model_path, fine_tuned_folder)
    print(f"{Fore.GREEN}Ada is ready to chat. Type 'quit' to exit.{Fore.RESET}")

    try:
        while True:
            user_input = await get_user_input()
            if user_input is None:
                response = await ada.process_input("What should I think about?", is_thought=True)
                print(f"{Fore.MAGENTA}Ada:{Fore.RESET}", response)
                await display_timer(30)
            elif user_input.lower() == 'quit':
                break
            else:
                response = await ada.process_input(user_input)
                print(f"{Fore.MAGENTA}Ada:{Fore.RESET}", response)
    finally:
        ada.save_session()
        print(f"{Fore.GREEN}Goodbye!{Fore.RESET}")

if __name__ == "__main__":
    asyncio.run(main())
