#!/bin/bash

# Define arrays for each parameter
# build_types=("OfficeSmall" "OfficeMedium" "OfficeLarge")
build_types=("OfficeSmall")
# climate_zones=("Cold_Dry" "Hot_Dry" "Mixed_Dry" "Warm_Dry")
# climate_zones=("Mixed_Dry" "Warm_Dry")
climate_zones=("Hot_Dry")
cities=("Tucson")

for build_type in "${build_types[@]}"; do
  for climate_zone in "${climate_zones[@]}"; do
    for city in "${cities[@]}"; do
      python methods/ppo.py \
        --build_type $build_type \
        --climate_zone $climate_zone \
        --city $city \
        --num_steps 100 \
        --eval
    done
  done
done