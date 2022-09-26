## Usage of the code:

### To run ZOSPI without MDN, run with

python3 ZOSPI_train.py -g 0 -sm random_local -en Hopper-v2 -ts 1000000 -eps 0.1 -ns 50 -r 0

### To run ZOSPI with MDN, run with

python3 MoG_train.py -g 0 -en Humanoid-v2 -ts 1000000 -alg SPI_MoG -n 10 -eps 0.0 -r 0