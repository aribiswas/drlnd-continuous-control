# Continuous Control of Robot Arm

The goal of this project is to train a reinforcement learning agent to perform continuous control of a 2-DOF robotic arm.

## The Environment

For this project, you will work with the Reacher environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

There are multiple instances of the robot arm within this environment. Generating experiences through nultiple experiences improves the training of the reinforcement learning agent.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. The above animation shows the performance of a trained agent.

## Install dependencies

To run this project, you must have Python 3.6, Pytorch and Unity ML-Agents toolkit installed. Follow the instructions in Udacity's Deep Reinforcement Learning Nanodegree [repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install the required dependencies.

You must also use one of the following Unity environments for this project. Download the environment specific to your platform and place it in the same folder as this project. Unzip the file and extract the contents in the same folder.

* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

## Running the code

You can run the code from any Python IDE or a terminal.

* To run the code in a Jupyter notebook, open the **Continuous_Control.ipynb** notebook. This notebook goes through starting the Unity environment, creating the agent, training and simulation.
* To run from a terminal, run **reacher_train_ddpg.py** to train the agent, or **reacher_sim_ddpg.py** to watch a trained agent.

<pre><code>
python reacher_train_ddpg.py
python reacher_sim_ddpg.py
</code></pre>

For both the above steps, you must change the **file_name** variable in the code to match the appropriate platform.

## Further reading

If you are interested to read about the DDPG algorithm then read the project **Report.md**. It contains explanation on the algorithm, implementation and outlines what you may expect in the training process.

*Ref: Deep Reinforcement Learning Nanodegree resources.*
