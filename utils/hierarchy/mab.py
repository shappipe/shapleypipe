import math
import numpy as np
from typing import List, Dict, Union, Optional
import multiprocessing as mp


class UCB1_Bandit:
    def __init__(self, arms: List[int], exploration_constant: float = 2.0,
                 manager = None):
        """
        UCB1 Multi-armed Bandit.

        Args:
            arms (list): List of arm identifiers (in our case, operator indices).
            exploration_constant (float): The 'C' parameter to balance exploration/exploitation.
            manager: A multiprocessing.Manager instance if used in parallel context.
        """
        self.arms = arms
        self.exploration_constant = exploration_constant
        self.is_parallel = manager is not None

        if manager:
            self._visits = manager.dict({arm: 0 for arm in arms})
            self._rewards = manager.dict({arm: 0.0 for arm in arms})
            self._total_visits = manager.Value('i', 0)
            self.lock = manager.Lock()
        else:
            self._visits = {arm: 0 for arm in arms}
            self._rewards = {arm: 0.0 for arm in arms}
            self._total_visits = 0
            self.lock = None

    def select_arm(self, mask: Optional[list[int]]=None) -> int:
        """Selects an arm according to the UCB1 strategy."""
        if self.is_parallel and self.lock:
            with self.lock:
                if mask is not None and len(mask) > 0:
                    return self._select_arm_internal_masked(mask)
                else:
                    return self._select_arm_internal()
        else:
            if mask is not None and len(mask) > 0:
                return self._select_arm_internal_masked(mask)
            else:
                return self._select_arm_internal()
        
    def select_rand(self):
        return int(np.random.choice(self.arms))

    def _select_arm_internal(self) -> int:
        total_visits_val = self._total_visits.value if self.is_parallel else self._total_visits

        if total_visits_val == 0:
            return int(np.random.choice(self.arms))

        ucb_scores = {}
        for arm in self.arms:
            visits = self._visits[arm]
            if visits == 0:
                return arm

            avg_reward = self._rewards[arm] / visits
            exploration_term = self.exploration_constant * math.sqrt(
                math.log(total_visits_val) / visits
            )
            ucb_scores[arm] = avg_reward + exploration_term

        # return max(ucb_scores, key=ucb_scores.get)
        return max(ucb_scores, key=ucb_scores.get)

    def _select_arm_internal_masked(self, mask: list[int]) -> int:
        total_visits_val = self._total_visits.value if self.is_parallel else self._total_visits

        if total_visits_val == 0:
            return int(np.random.choice(mask))

        ucb_scores = {}
        for arm in mask:
            visits = self._visits[arm]
            if visits == 0:
                return arm

            avg_reward = self._rewards[arm] / visits
            exploration_term = self.exploration_constant * math.sqrt(
                math.log(total_visits_val) / visits
            )
            ucb_scores[arm] = avg_reward + exploration_term

        # return max(ucb_scores, key=ucb_scores.get)
        return max(ucb_scores, key=ucb_scores.get)


    def update(self, arm: int, reward: float):
        """Updates the state of an arm after it has been played."""
        if arm not in self.arms:
            raise ValueError(f"Arm {arm} not in bandit arms {self.arms}")

        if not isinstance(reward, (int, float)) or math.isnan(reward):
            raise ValueError(f"Invalid reward: {reward}")

        if self.is_parallel and self.lock:
            with self.lock:
                self._update_internal(arm, reward)
        else:
            self._update_internal(arm, reward)

    def _update_internal(self, arm: int, reward: float):
        self._visits[arm] += 1
        self._rewards[arm] += reward

        if self.is_parallel:
            self._total_visits.value += 1
        else:
            self._total_visits += 1

    def get_best_arm(self, mask: Optional[list[int]]=None) -> int:
        """Returns the arm with the highest average reward."""
        if self.is_parallel and self.lock:
            with self.lock:
                if mask:
                    return self._get_best_arm_internal_masked(mask)
                else:
                    return self._get_best_arm_internal()
        else:
            if mask:
                return self._get_best_arm_internal_masked(mask)
            else:
                return self._get_best_arm_internal()

    def _get_best_arm_internal(self) -> int:
        if all(self._visits[arm] == 0 for arm in self.arms):
            return int(np.random.choice(self.arms))

        avg_rewards = {}
        for arm in self.arms:
            visits = self._visits[arm]
            if visits > 0:
                avg_rewards[arm] = self._rewards[arm] / visits
            else:
                avg_rewards[arm] = float('-inf')

        return max(avg_rewards, key=avg_rewards.get)
    
    def _get_best_arm_internal_masked(self, mask: list[int]) -> int:
        if all(self._visits[arm] == 0 for arm in mask):
            return np.random.choice(mask)

        avg_rewards = {}
        for arm in mask:
            visits = self._visits[arm]
            if visits > 0:
                avg_rewards[arm] = self._rewards[arm] / visits
            else:
                avg_rewards[arm] = float('-inf') 

        return max(avg_rewards, key=avg_rewards.get)

    def get_statistics(self) -> Dict[str, Union[int, float, Dict]]:
        if self.is_parallel and self.lock:
            with self.lock:
                return self._get_statistics_internal()
        else:
            return self._get_statistics_internal()

    def _get_statistics_internal(self) -> Dict[str, Union[int, float, Dict]]:
        total_visits = self._total_visits.value if self.is_parallel else self._total_visits

        arm_stats = {}
        for arm in self.arms:
            visits = self._visits[arm]
            total_reward = self._rewards[arm]
            avg_reward = total_reward / visits if visits > 0 else 0.0
            arm_stats[arm] = {
                'visits': visits,
                'total_reward': total_reward,
                'avg_reward': avg_reward
            }

        return {
            'total_visits': total_visits,
            'num_arms': len(self.arms),
            'exploration_constant': self.exploration_constant,
            'arm_statistics': arm_stats
        }

    def reset(self):
        if self.is_parallel and self.lock:
            with self.lock:
                self._reset_internal()
        else:
            self._reset_internal()

    def _reset_internal(self):
        for arm in self.arms:
            self._visits[arm] = 0
            self._rewards[arm] = 0.0

        if self.is_parallel:
            self._total_visits.value = 0
        else:
            self._total_visits = 0
