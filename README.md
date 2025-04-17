# elevator_domain
This is a machine learning based elevator control project for NUS MSc in Robotics module CS5446: AI Planning and Decision Making, Assignment 2 Q3. By Assignment Group A33.

This project aims to deploy an reinforcement learning agent that learn and control a lift, using the PyRDDL elevator domain gym.

## Setup
### 1. Clone the repo
```
git clone https://github.com/PiusLim373/elevator_domain.git
```
### 2. Setup using Conda
```
cd elevator_domain
conda env create -f requirement.yml
conda activate cs5446-elevator-domain
```
This will create a conda environment cs5446-elevator-domain with necessary packages installed to run the project.

## Training with Custom PPO Algorithm
```
python ppo_custom_train.py
```

During / after the training, models checkpoint will periodically saved to the  `saves/` folder.

## Testing with Trained Model
```
python ppo_custom_test.py
```
A sample trained model is provided, feel free to change it.

![](/docs/elevator_domain_test.png)