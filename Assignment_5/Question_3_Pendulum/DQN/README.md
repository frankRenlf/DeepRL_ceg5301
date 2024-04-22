# DQN

The codes for DQN are adapted from [Kchu](https://github.com/Kchu/DeepRL_PyTorch)

The default number of parallel threads is set to 1 because the percentage of the sampling period in all traning time is very small. In other words, the sampling costs very little time, and hence the optimization space is very narrow even by employing parallel coding technique.

## Installing Dependency:
You can use anaconda to create a python3 environment:

```bash
conda env create -f environment.yml
```

If some error messages from Anaconda are raised, you could choose to install the required python3 package manually. Run the following command with CMD in Windows or Shell in Linux or MacOS:

```bash
pip3 install pytorch pygame gym opencv_python
```

How to use
Enter the DQN directory, and run the python3 command 'python3 train.py':

```bash
cd DQN-pytorch # 
python3 train.py
```

When testing the bulit environment, you could let the code idle with the following command:

```bash
python3 train.py --idling
```

When you run these codes, it can automatically create two subdirectories under the current directory: ./data/model/ & ./data/plots/. These two directories are used to store the models and the results.

After training, you can plot the results by running result_show.py with appropriate parameters.

## References:
Human-level control through deep reinforcement learning (DQN) | [Paper](https://www.nature.com/articles/nature14236) | [Code](https://github.com/buaawgj/DQN-pytorch/dqn.py) |