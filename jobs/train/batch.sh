# wikipedia
./jobs/train/primer.sh 'wikipedia' 'bert'    'adam' 250 0.000001 32 10 0.0 1.0 32 4320 longgpu
./jobs/train/primer.sh 'wikipedia' 'roberta' 'adam' 250 0.000001 32 10 0.0 1.0 32 4320 longgpu
./jobs/train/primer.sh 'wikipedia' 'xlnet'   'adam' 250 0.000001 16 10 0.0 1.0 32 4320 longgpu

# hatebase
./jobs/train/primer.sh 'hatebase' 'bert'    'adam' 250 0.000001 32 50 0.0 1.0 32 4320 longgpu
./jobs/train/primer.sh 'hatebase' 'roberta' 'adam' 250 0.000001 32 50 0.0 1.0 32 4320 longgpu
./jobs/train/primer.sh 'hatebase' 'xlnet'   'adam' 250 0.000001 16 50 0.0 1.0 32 4320 longgpu

# civil_comments
./jobs/train/primer.sh 'civil_comments' 'bert'    'adam' 250 0.000001 32 10 0.0 1.0 32 10080 longgpu
./jobs/train/primer.sh 'civil_comments' 'roberta' 'adam' 250 0.000001 32 10 0.0 1.0 32 10080 longgpu
./jobs/train/primer.sh 'civil_comments' 'xlnet'   'adam' 250 0.000001 16 10 0.0 1.0 32 10080 longgpu

# nuclear_energy
./jobs/train/primer.sh 'nuclear_energy' 'bert'    'adam' 250 0.00006 32 10 0.0 1.0 32 4320 longgpu
./jobs/train/primer.sh 'nuclear_energy' 'roberta' 'adam' 250 0.00006 32 10 0.0 1.0 32 4320 longgpu
./jobs/train/primer.sh 'nuclear_energy' 'xlnet'   'adam' 250 0.00006 32 10 0.0 1.0 32 4320 longgpu

# climate-change_waterloo
./jobs/train/primer.sh 'climate-change_waterloo' 'bert'    'adam' 250 0.00006 32 15 0.0 1.0 32 4320 longgpu
./jobs/train/primer.sh 'climate-change_waterloo' 'roberta' 'adam' 250 0.00006 32 15 0.0 1.0 32 4320 longgpu
./jobs/train/primer.sh 'climate-change_waterloo' 'xlnet'   'adam' 250 0.00006 32 15 0.0 1.0 32 4320 longgpu

# imdb
./jobs/train/primer.sh 'imdb' 'bert'    'adam' 128 0.00002 64 5 0.0 1.0 32 4320 longgpu
./jobs/train/primer.sh 'imdb' 'roberta' 'adam' 128 0.00004 64 5 0.0 1.0 32 4320 longgpu
./jobs/train/primer.sh 'imdb' 'xlnet'   'adam' 128 0.00002 64 5 0.0 1.0 32 4320 longgpu

# sst
./jobs/train/primer.sh 'sst' 'bert'    'adam' 128 0.000006 32 10 0.0 1.0 32 4320 longgpu
./jobs/train/primer.sh 'sst' 'roberta' 'adam' 128 0.000006 32 10 0.0 1.0 32 4320 longgpu
./jobs/train/primer.sh 'sst' 'xlnet'   'adam' 128 0.000006 32 10 0.0 1.0 32 4320 longgpu

# fnc1
./jobs/train/primer.sh 'fnc1' 'uclmr' 'adam' 250 0.001 500 1000 0.00001 5.0 32 4320 longgpu
