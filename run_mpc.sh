# python methods/mpc.py \
#     --build-type OfficeMedium \
#     --climate-zone Hot_Dry \
#     --city Tucson \
#     --max-timestep 240 \
#     --time-reso 360 \
#     --horizon 12  # Using a 12-step prediction horizon (equivalent to ~1.2 hours with time_reso=360) 

python methods/mpc.py \
    --build-type OfficeMedium \
    --climate-zone Cool_Humid \
    --city Buffalo \
    --max-timestep 240 \
    --time-reso 360 \
    --horizon 12 \
    --noise 2  # add noise to the MPC estimation results

# python methods/mpc.py \
#     --build-type OfficeSmall \
#     --climate-zone Hot_Dry \
#     --city Tucson \
#     --max-timestep 240 \
#     --time-reso 360 \
#     --horizon 12  # Using a 12-step prediction horizon (equivalent to ~1.2 hours with time_reso=360) 

# python methods/mpc.py \
#     --build-type OfficeSmall \
#     --climate-zone Cool_Humid \
#     --city Buffalo \
#     --max-timestep 240 \
#     --time-reso 360 \
#     --horizon 12  # Using a 12-step prediction horizon (equivalent to ~1.2 hours with time_reso=360) 
