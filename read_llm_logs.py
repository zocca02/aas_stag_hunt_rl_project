from utils.llm_utils import LLMStagHuntLog
import numpy as np

def print_llm_stats(base_dir, file_base_name, n_files):
    file_names = [f"{base_dir}/{file_base_name}_{i+1}_ag2.pkl" for i in range(n_files)]
    logs = [LLMStagHuntLog.load(fn) for fn in file_names]

    episode_rewards = []
    
    for log in logs:
        episode_rewards.append(np.stack([tsl["reward"] for tsl in log.episode_log]))

    episode_total_rewards = np.stack([er.sum() for er in episode_rewards])
    episode_stags = np.stack([(er==5.0).sum() for er in episode_rewards])
    episode_foragings = np.stack([(er==1.0).sum() for er in episode_rewards])
    episode_maulings = np.stack([(er==-0.5).sum() for er in episode_rewards])

    print("##############################################################")
    print(f"Stats for {" ".join(file_base_name.split("_"))}")
    print("##############################################################")

    print("\n-------------------------------")
    print(f"Rewards: {episode_total_rewards}")
    mean = episode_total_rewards.mean()
    std = episode_total_rewards.std()
    best = episode_total_rewards.max()
    worst = episode_total_rewards.min()
    print(f"Mean: {mean:.2f}\t\tStd: {std:.2f}\t\tBest: {best:.2f}\t\tWorst: {worst:.2f}")

    print("\n-------------------------------")
    print(f"Stags: {episode_stags}")
    mean = episode_stags.mean()
    std = episode_stags.std()
    best = episode_stags.max()
    worst = episode_stags.min()
    print(f"Mean: {mean:.2f}\t\tStd: {std:.2f}\t\tBest: {best:.2f}\t\tWorst: {worst:.2f}")

    print("\n-------------------------------")
    print(f"Foragings: {episode_foragings}")
    mean = episode_foragings.mean()
    std = episode_foragings.std()
    best = episode_foragings.max()
    worst = episode_foragings.min()
    print(f"Mean: {mean:.2f}\t\tStd: {std:.2f}\t\tBest: {best:.2f}\t\tWorst: {worst:.2f}")

    print("\n-------------------------------")
    print(f"Maulings: {episode_maulings}")
    mean = episode_maulings.mean()
    std = episode_maulings.std()
    best = episode_maulings.max()
    worst = episode_maulings.min()
    print(f"Mean: {mean:.2f}\t\tStd: {std:.2f}\t\tBest: {best:.2f}\t\tWorst: {worst:.2f}")

print_llm_stats("llm_logs", "mistral_3_3B_reasoning", 10)
print()
print()
print_llm_stats("llm_logs", "mistral_3_8B_reasoning", 10)
print()
print()
print_llm_stats("llm_logs", "mistral_3_14B_reasoning", 10)


print()
print()


print_llm_stats("llm_logs", "mistral_3_3B_reasoning_vs_dqn", 10)
print()
print()
print_llm_stats("llm_logs", "mistral_3_8B_reasoning_vs_dqn", 10)
print()
print()
print_llm_stats("llm_logs", "mistral_3_14B_reasoning_vs_dqn", 10)