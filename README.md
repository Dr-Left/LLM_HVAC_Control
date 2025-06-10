# LLMControl: Facilitating LLM Agent to Control Building HVAC with Zero-shot Prompting

Author: Jingwei Zuo 
Mail: naohzjw@gmail.com

## Set up the environment

```bash
conda activate thesis
cd BEAR
pip install -e .
cd ../
pip install -e .
```

## Use the framework

`./run.sh` is to start the LLM framework. `./run_mpc.sh` is to start the MPC baseline. `./run_ppo.sh` is to start the PPO baseline.

The environments tested in the thesis: OfficeSmall & OfficeMedium (6 and 18 rooms in total respectively)

## Argument explanation

### Common Arguments (All Methods)
- `--build-type`: Type of building environment to simulate. Options: `OfficeSmall` (6 rooms), `OfficeMedium` (18 rooms).
- `--climate-zone`: Climate zone for the simulation. Options include: `Hot_Dry`, `Cool_Humid`, etc.
- `--city`: City location for weather data. E.g., `Tucson`, `Buffalo`.
- `--max-timestep`: Total number of timesteps to run the simulation (default: 240).
- `--time-reso`: Time resolution in seconds for each timestep (default: 3600, which is 1 hour).
- `--noise`: Adds Gaussian noise to the simulation for robustness testing (default: 0.0).

### LLM (llm.py) Arguments
- `--model`: LLM model to use for control decisions (default: "deepseek").
- `--prompt-style`: Style of prompt to use:
  - `cot_first`: Chain of Thought reasoning and then gives the result.
  - `cot_last`: Gives the result first, followed by Chain of Thought reasoning.
- `--history-method`: Method to select historical data for prompting:
  - `random`: Randomly selects historical data.
  - `highest_reward`: Selects historical data with highest rewards (default).
  - `none`: Does not include historical data.
- `--enable-hindsight`: When set, enables hindsight analysis in the prompts.

### PPO (ppo.py) Arguments
- `--train-steps`: Number of training steps for the PPO algorithm.
- `--learning-rate`: Learning rate for the optimizer (default: 3e-4).
- `--eval`: When set, runs evaluation after training.
- Additional PPO-specific parameters (from stable-baselines3):
  (You have to manually set these parameters in the `ppo.py` file)
  - `n_steps`: Number of steps to run for each environment per update.
  - `batch_size`: Minibatch size for training.
  - `n_epochs`: Number of epochs when optimizing the surrogate loss.
  - `gamma`: Discount factor for rewards.
  - `gae_lambda`: Factor for trade-off of bias vs variance in Generalized Advantage Estimator.
  - `clip_range`: Parameter for limiting policy update size.
  - `ent_coef`: Entropy coefficient for encouraging exploration.
  - `vf_coef`: Value function coefficient for the loss calculation.

### MPC (mpc.py) Arguments
- `--horizon`: Prediction horizon for MPC algorithm (default: 12 steps).


## Plotting

Use `plot.py` to plot the results. 

```bash
python plot.py -i SIM_ID
```

## Notes

### Environment Observation Vector
The observation vector provided to control algorithms consists of:
- `temperature_of_all_rooms`: Current temperature in each room (length = roomnum)
- `outside_temperature`: Current outside temperature (length = 1)
- `Global_horizontal_irradiance` (GHI): Solar radiation data for each room (length = roomnum)
- `temperature_of_the_ground`: Ground temperature (length = 1)
- `occupancy_power`: Power consumption from occupancy in each room (length = roomnum)

The vector is concatenated as follows:
```python
self.state = np.concatenate(
    (
        X_new,                                  # Room temperatures
        self.OutTemp[self.epochs].reshape(-1,), # Outside temperature
        ghi_repeated,                           # GHI values
        self.GroundTemp[self.epochs].reshape(-1), # Ground temperature
        occ_repeated,                           # Occupancy power
    ),
    axis=0,
)
```

### Actions
Actions are HVAC control intensities for each room, ranging from 0-10, where:
- Values are normalized to [0,1] during environment processing
- Higher values mean more cooling/heating power
- The system automatically determines heating/cooling mode based on target temperature

### Reward Function
The reward function balances temperature comfort and energy consumption:
- Rewards increase when room temperatures are close to target temperatures
- Penalties apply for energy consumption from HVAC usage
- The framework allows customizing the reward function trade-offs

### Results
Simulation results are saved in the `results/` directory with unique IDs. Use the plotting tool to visualize:
```bash
python plot.py -i SIM_ID
```
This generates visualizations of temperature control performance, energy consumption, and rewards over time.
