# python methods/llm/llm.py \
#     --build-type="OfficeMedium" \
#     --climate-zone="Cool_Humid" \
#     --city="Buffalo" \
#     --model="qwen-max" \
#     --max-timestep=240 \
#     --time-reso=360 \
#     --prompt-style="cot_first" \
#     --history-method="highest_reward" \
#     --enable-hindsight

# python methods/llm/llm.py \
#     --build-type="OfficeMedium" \
#     --climate-zone="Cool_Humid" \
#     --city="Buffalo" \
#     --model="qwen-max" \
#     --max-timestep=24 \
#     --time-reso=3600 \
#     --prompt-style="cot_first" \
#     --history-method="highest_reward" \
#     --enable-hindsight

python methods/llm/llm.py \
    --build-type="OfficeMedium" \
    --climate-zone="Cool_Humid" \
    --city="Buffalo" \
    --model="qwen-max" \
    --max-timestep=240 \
    --time-reso=360 \
    --prompt-style="cot_first" \
    --history-method="highest_reward" \
    --enable-hindsight \
    --noise 2 \
    2>&1 | tee logs/buffalo_cool_humid_office_medium_noise=2.0_qwen-max_240_360_cot_first_highest_reward_hindsight.log

# python methods/llm/llm.py \
#     --build-type="OfficeMedium" \
#     --climate-zone="Cool_Humid" \
#     --city="Buffalo" \
#     --model="qwen-max" \
#     --max-timestep=240 \
#     --time-reso=360 \
#     --prompt-style="cot_first" \
#     --history-method="highest_reward"

# python methods/llm/llm.py \
#     --build-type="OfficeMedium" \
#     --climate-zone="Cool_Humid" \
#     --city="Buffalo" \
#     --model="qwen-max" \
#     --max-timestep=240 \
#     --time-reso=360 \
#     --prompt-style="cot_first" \
#     --history-method="none"
