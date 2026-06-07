import pickle
from collections import defaultdict


def state_to_str(state, width, height):
    x_A, y_A, x_B, y_B, x_stag, y_stag, x_p1, y_p1, x_p2, y_p2 = state

    cells = defaultdict(set)
    cells[(x_A, y_A)].add("A")
    cells[(x_B, y_B)].add("B")
    cells[(x_stag, y_stag)].add("S")
    cells[(x_p1, y_p1)].add("P")
    cells[(x_p2, y_p2)].add("P")

    s = ""

    
    for y in range(height):
        row_str = "|"
        for x in range(width):
            content = cells[(x, y)]
            
            to_add = ""
            if not content:
                to_add += "." 
            else:
                if "A" in content:
                    to_add += "A"
                if "B" in content:
                    to_add += "B"
                if "S" in content:
                    to_add += "S"
                if "P" in content:
                    to_add += "P"  
            
            if len(to_add)==1:
                row_str+=f" {to_add} "
            if len(to_add)==2:
                row_str+=f"{to_add} "
        row_str += "|"
        s+=row_str+"\n"

    return s

class LLMStagHuntLog:
    def __init__(self, width, height, ask_for_cot=True, ask_for_critique=False):
        self.episode_log = []
        self.width = width
        self.height = height
        self.ask_for_critique = ask_for_critique
        self.ask_for_cot = ask_for_cot
    
    def add_log(self, step_log):
        self.episode_log.append({
            "state": step_log["state"],
            "state_description": step_log["state_description"],
            "decision": step_log["decision"],
            "reward": step_log["reward"]
        })

        if self.ask_for_cot:
            self.episode_log[-1]["cot"] = step_log["cot"]

        if self.ask_for_critique:
            self.episode_log[-1]["critique"] = step_log["critique"]
            self.episode_log[-1]["revised_decision"] = step_log["revised_decision"]
    
    def save(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_name):
        with open(file_name, "rb") as f:
            log = pickle.load(f)
        return log

    def dump(self):
        s=""
        for i, log in enumerate(self.episode_log):
            s+=f"Step: {i+1}\n"
            s+=f"{state_to_str(log["state"], self.width, self.height)}\n\n"
            s+=f"State description: {log["state_description"]}\n\n"
            if self.ask_for_cot:
                s+=f"CoT: {log["cot"]}\n"
            s+=f"Decision: {log["decision"]}\n"
            
            if self.ask_for_critique:
                s+=f"\nCritique: {log["critique"]}\n"
                s+=f"Revised Decision: {log["revised_decision"]}\n"

            s+=f"Reward: {log["reward"]}\n\n"
        return s