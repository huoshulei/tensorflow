from Maze_env import Maze
from Rl_brain import QLearningTable
from Rl_brain import SarsaTable
from Rl_brain import SarsaLamdbaTable


def update():
    for episode in range(100):
        # initial observation
        observation = 环境.reset()

        while True:
            # fresh env
            环境.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = 环境.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    环境.destroy()


def update_sarsa():
    for episode in range(100):
        # initial observation
        observation = 环境.reset()
        # RL choose action based on observation
        action = RL.choose_action(str(observation))
        while True:
            # fresh env
            环境.render()

            # RL take action and get next observation and reward
            observation_, reward, done = 环境.step(action)

            # RL choose action based on observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation
            observation = observation_
            action = action_
            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    环境.destroy()


if __name__ == "__main__":
    print("阿德飒飒大苏打")
    环境 = Maze()
    RL = QLearningTable(actions=list(range(环境.n_actions)))
    # RL = SarsaTable(actions=list(range(环境.n_actions)))
    # RL = SarsaLamdbaTable(actions=list(range(环境.n_actions)))
    环境.after(100, update)
    # env.after(100, update_sarsa())
    环境.mainloop()
