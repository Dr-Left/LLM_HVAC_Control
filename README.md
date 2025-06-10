```
conda activate thesis
cd BEAR
pip install -e .
cd ../
pip install -e .
```
observation vector:
[temperature_of_all_rooms (x roomnum), outside_temperature, Global _horizontal_irradiance (GHI, x roomnum), temperature_of_the_ground, occupancy_power (x roomnum)]

```
self.state = np.concatenate(
            (
                X_new,
                self.OutTemp[self.epochs].reshape(
                    -1,
                ),
                ghi_repeated,
                self.GroundTemp[self.epochs].reshape(-1),
                occ_repeated,
            ),
            axis=0,
        )
```

# question: occupancy_power: 1 number or an array?

Currrently: initialize with (21,23) temperature