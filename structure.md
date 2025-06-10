./methods/ include all the experiment methods.
    llm/ include all the files related to llm
        llm.py is the executive entry file of llm method
    utils/ integrates all utilities such as chatting modules, logging modules and plotting modules.
    ppo.py is the executive entry file of ppo method

./models stores all PPO models.
./results 
    data/ stores all experiment results (h5 files) to be read for plotting
    figures/ stores all the figures (which could be generated again by running plot.py with the data)
    demonstration/ is deprecated
    old_result/ is deprecated
./rewards/ stores PPO reward lists (deprecated?)
./scripts/ stores PPO running scripts (deprectated)

./run.sh is to run llm method
./run_ppo.sh is to run ppo method

./BEAR/ is a forked git repo, for environment simulation.



