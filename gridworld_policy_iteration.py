"""
This program implements Policy Iteration in a Gridworld with:
- Obstacles (impassable black cells)
- Reward flags (gold cells that give a bonus)
- Visual interface with matplotlib
- Path overlay from a start cell to a terminal
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Button, Slider
from dataclasses import dataclass

A_UP, A_RIGHT, A_DOWN, A_LEFT = 0, 1, 2, 3
ACTIONS = [A_UP, A_RIGHT, A_DOWN, A_LEFT]
A2DELTA = {
    A_UP: (-1, 0),
    A_RIGHT: (0, 1),
    A_DOWN: (1, 0),
    A_LEFT: (0, -1),
}

@dataclass
class Gridworld:
    H: int = 5
    W: int = 5
    terminals: tuple = ((0, 4), (4, 4), (1,3))
    step_reward: float = -1.0
    obstacles: tuple = ((2, 2),)
    reward_flags: tuple = ((1, 3),)
    #reward_flags: tuple = ()
    flag_reward: float = 5.0

    def in_bounds(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W
    def is_terminal(self, s):
        return s in self.terminals
    def is_obstacle(self, s):
        return s in self.obstacles
    def is_flag(self, s):
        return s in self.reward_flags

    def next_state(self, s, a):
        if self.is_terminal(s):
            return s # stay if already terminal
        dr, dc = A2DELTA[a]
        nr, nc = s[0] + dr, s[1] + dc
        if not self.in_bounds(nr, nc) or (self.is_obstacle((nr, nc)) and not self.is_terminal((nr, nc))):
            return s
        return (nr, nc)

    def reward(self, s, a, s_next):
        # entering a terminal gives +10
        if self.is_terminal(s_next) and not self.is_terminal(s):
            # special case: if it's also a flag, give bonus + terminal reward
            if self.is_flag(s_next):
                return 10.0 + self.flag_reward 
            return 10.0
        # stepping on a flag (but not terminal)
        if self.is_flag(s_next):
            return self.flag_reward
        return self.step_reward

def transition_dist(env, s, a, noise):
    if env.is_terminal(s):
        return [(s, 1.0)]
    perp = {A_UP: [A_LEFT, A_RIGHT], A_DOWN: [A_LEFT, A_RIGHT], A_LEFT: [A_UP, A_DOWN], A_RIGHT: [A_UP, A_DOWN]}[a]
    main_p = 1 - noise
    slip_p = noise / 2
    outcomes = []
    for aa, p in [(a, main_p), (perp[0], slip_p), (perp[1], slip_p)]:
        s_next = env.next_state(s, aa)
        outcomes.append((s_next, p))
    from collections import defaultdict
    agg = defaultdict(float)
    for s_next, p in outcomes:
        agg[s_next] += p
    return list(agg.items())

def debug_policy_along_path(env, pi, start, max_steps=50):
    """Print detailed info while following pi from start."""
    s = start
    print("DEBUG PATH START:", start)
    for step in range(max_steps):
        print(f"\nStep {step}: s = {s}, terminal? {env.is_terminal(s)}, obstacle? {env.is_obstacle(s)}")
        if env.is_terminal(s):
            print(" -> Already terminal. Stopping.")
            break

        probs = pi[s[0], s[1], :]
        best = np.flatnonzero(np.isclose(probs, probs.max()))
        print(" policy probs:", np.round(probs, 3), " best action(s):", best)

        # For each action, show what transition(s) are possible and reward(s)
        for a in ACTIONS:
            outs = transition_dist(env, s, a, noise=0.0)  # deterministic view
            print(f"  action {a}: ", end="")
            for s_next, p in outs:
                r = env.reward(s, a, s_next)
                print(f" -> {s_next} (p={p:.2f}, r={r})", end="")
            print("")

        # chosen action and actual next_state according to env.next_state
        a_chosen = int(np.argmax(pi[s[0], s[1], :]))
        s_next = env.next_state(s, a_chosen)
        print(" chosen action:", a_chosen, " next_state:", s_next, " terminal?", env.is_terminal(s_next))

        # if next_state == s, check neighbors for a terminal (diagnostic)
        if s_next == s:
            print("  next_state == s (stuck). Checking neighbors for reachable terminals:")
            found_terminal = False
            for a in ACTIONS:
                cand = env.next_state(s, a)
                print("   action", a, " ->", cand, " terminal?", env.is_terminal(cand))
                if env.is_terminal(cand):
                    found_terminal = True
            print("  neighbor terminal found?:", found_terminal)

        # append / step
        if s_next == s:
            print(" Stuck. Stopping path trace.")
            break
        s = s_next
    else:
        print("Reached max_steps without terminal.")

    print("DEBUG PATH END. Final s:", s, "is_terminal?", env.is_terminal(s))

def random_policy(env):
    pi = np.zeros((env.H, env.W, 4))
    for r in range(env.H):
        for c in range(env.W):
            if env.is_terminal((r, c)) or env.is_obstacle((r, c)):
                continue
            pi[r, c, :] = 1 / 4
    return pi

def policy_evaluation(env, V, pi, gamma, noise):
    V_new = V.copy()
    """
    TODO: Implement one swep policy evaluation to compute V(s) given policy pi:
    update V(s) given policy pi, the environment dynamics, and gamma
    """
    return V_new

def policy_improvement(env, V, gamma, noise):
    pi_new = np.zeros((env.H, env.W, 4))
    """
    TODO: Implement policy improvement: update pi greedily based on current V(s).
    """
    return pi_new, True

def policy_iteration(env, gamma, noise, max_eval_iters=100, tol=1e-6):
    V = np.zeros((env.H, env.W))
    pi = random_policy(env)
    stable = False
    """
    TODO: Combine policy evaluation and improvement to perform policy iteration.
    """
    return V, pi

def extract_path(env, pi, start, max_steps=50):
    path = [start]
    s = start
    for _ in range(max_steps):
        # Choose the best action according to the policy
        a = np.argmax(pi[s[0], s[1], :])
        s_next = env.next_state(s, a)
        path.append(s_next)

        if env.is_terminal(s_next):
            break  # stop if current state is terminal

        # Stop if agent is stuck
        if s_next == s:
           break

        s = s_next        
    return path

class Viewer:
    def __init__(self):
        self.env = Gridworld()
        self.start = (4, 0)
        self.gamma = 0.95
        self.noise = 0.0 # 0.2
        self.V = np.zeros((self.env.H, self.env.W))
        self.pi = random_policy(self.env)
        self.path = []
        # setup matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(bottom=0.23)
        self.cmap = colors.LinearSegmentedColormap.from_list("vals", ["#f7fbff", "#08306b"])
        # Buttons
        ax_iter = plt.axes([0.08, 0.08, 0.20, 0.08])
        ax_run = plt.axes([0.30, 0.08, 0.28, 0.08])
        ax_path = plt.axes([0.60, 0.08, 0.15, 0.08])
        self.btn_iter = Button(ax_iter, "Iterate once")
        self.btn_run = Button(ax_run, "Run to convergence")
        self.btn_path = Button(ax_path, "Show Path")
        # Sliders
        ax_gamma = plt.axes([0.10, 0.02, 0.35, 0.03])
        ax_noise = plt.axes([0.58, 0.02, 0.35, 0.03])
        self.s_gamma = Slider(ax_gamma, "gamma", 0.50, 0.99, valinit=self.gamma)
        self.s_noise = Slider(ax_noise, "noise", 0.0, 0.5, valinit=self.noise)
        # Connect events
        self.btn_iter.on_clicked(self.on_iter)
        self.btn_run.on_clicked(self.on_run)
        self.btn_path.on_clicked(self.on_path)
        self.s_gamma.on_changed(self.on_hyper)
        self.s_noise.on_changed(self.on_hyper)
        self.redraw()
    def on_hyper(self, _):
        self.gamma = self.s_gamma.val
        self.noise = self.s_noise.val
        self.redraw()
    def on_iter(self, _):
        for _ in range(15):
            self.V = policy_evaluation(self.env, self.V, self.pi, self.gamma, self.noise)
        self.pi, _ = policy_improvement(self.env, self.V, self.gamma, self.noise)
        self.redraw()
    def on_run(self, _):
        self.V, self.pi = policy_iteration(self.env, self.gamma, self.noise)
        debug_policy_along_path(self.env, self.pi, self.start)
        self.redraw()
    def on_path(self, _):
        self.path = extract_path(self.env, self.pi, start=self.start)
        self.redraw()
    def redraw(self):
        self.ax.clear()
        self.ax.set_title("Gridworld with Path")
        # Draw grid lines
        self.ax.set_xticks(np.arange(-.5, self.env.W, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.env.H, 1), minor=True)
        self.ax.grid(which="minor", color="#aaaaaa", linewidth=0.6)
        
        # Draw value function heatmap
        self.ax.imshow(self.V, cmap=self.cmap, origin="upper")
        # Draw terminals (green outline)
        for (tr, tc) in self.env.terminals:
            self.ax.add_patch(plt.Rectangle((tc-0.5, tr-0.5), 1, 1, fill=False, linewidth=3.0, edgecolor="#00aa00"))
        # Draw obstacles (black)  
        for (or_, oc) in self.env.obstacles:
            self.ax.add_patch(plt.Rectangle((oc-0.5, or_-0.5), 1, 1, color="black"))
        # Draw reward flags (gold)
        for (fr, fc) in self.env.reward_flags:
            self.ax.add_patch(plt.Rectangle((fc-0.5, fr-0.5), 1, 1, color="gold"))
        # Draw start state (red outline)
        sr, sc = self.start
        self.ax.add_patch(plt.Rectangle((sc-0.5, sr-0.5), 1, 1, fill=False, linewidth=3.0, edgecolor="red")                          )       

        # Draw path (red circles and line)
        if self.path:
            xs = [c for r, c in self.path]
            ys = [r for r, c in self.path]
            self.ax.plot(xs, ys, marker='o', color='red')
        self.fig.canvas.draw_idle()

def main():
    Viewer()
    plt.show()

if __name__ == "__main__":
    main()
