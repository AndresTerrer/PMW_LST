Consecutive swaths of measurements from polar orbiting satellites partially overlap at high latitudes ( > 60 degrees) and at the beginning/end of the day

For WindSat we use the early observations and do not overwrite unless an observation is missing. 
so that all geophysical parameters for a given 0.25-deg cell refer to the same observational time

Reasoning: 
WindSat is more complicated than other satellites because each frequency channel (and thus each geophysical parameter) has a distinct viewing geometry and therefore a slightly different time. If we were to overwrite early WindSat observations with later observations, it would cause large data gaps at high latitudes in the low resolution channels, especially at swath edges.
https://www.remss.com/missions/windsat/#overwriting_note


Frequency channels and polarizations :

Frequencies: 
0 -- 6.8 GHz
1 -- 10.7 GHz
2 -- 18.7 GHz (Ku)
3 -- 23.8 GHz
4 -- 37.0 GHz (Ka)


Polarization: 
0 -- V
1 -- H
(except for 6.8 and 23.8 GHz): 
2 -- P (+45º)
3 -- M (-45º)
4 -- L (Circular Left)
5 -- R (Circular Right)






https://images.remss.com/~RSS-TB/intercalibration/windsat_TB_maps_daily_025deg/RSS_WINDSAT_DAILY_TBTOA_MAPS_2017_01_01.nc
