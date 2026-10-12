# Part 4.2: Snake DQN

`agent.py` here is Patrick Loeber's
[snake-ai-pytorch](https://github.com/patrickloeber/snake-ai-pytorch)
`agent.py` (MIT License) with one change: training stops after
`MAX_GAMES = 300` games instead of running forever, and the final
score/mean-score plot stays open for the screenshot.

It needs `game.py`, `model.py` and `helper.py` from that repo, so copy it
over the original `agent.py` in your clone and run it there:

```
cd D:\CPSC254
git clone https://github.com/patrickloeber/snake-ai-pytorch.git
copy CPSC254-Assignment2\snake_dqn\agent.py snake-ai-pytorch\agent.py
cd snake-ai-pytorch
pip install torch pygame matplotlib ipython
python agent.py
```

Close the plot window when training finishes to end the script.
