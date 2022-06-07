# wikipedia
./jobs/attack/primer.sh 'abuse' 'wikipedia' 'wikipedia' 'bert'    250 32 500 32 4320 longgpu
./jobs/attack/primer.sh 'abuse' 'wikipedia' 'wikipedia' 'roberta' 250 32 500 32 4320 longgpu
./jobs/attack/primer.sh 'abuse' 'wikipedia' 'wikipedia' 'xlnet'   250 16 500 32 4320 longgpu

# hatebase
./jobs/attack/primer.sh 'abuse' 'hatebase' 'hatebase' 'bert'    250 32 500 32 4320 longgpu
./jobs/attack/primer.sh 'abuse' 'hatebase' 'hatebase' 'roberta' 250 32 500 32 4320 longgpu
./jobs/attack/primer.sh 'abuse' 'hatebase' 'hatebase' 'xlnet'   250 16 500 32 4320 longgpu

# civil_comments
./jobs/attack/primer.sh 'abuse' 'civil_comments' 'civil_comments' 'bert'    250 32 500 32 4320 longgpu
./jobs/attack/primer.sh 'abuse' 'civil_comments' 'civil_comments' 'roberta' 250 32 500 32 4320 longgpu
./jobs/attack/primer.sh 'abuse' 'civil_comments' 'civil_comments' 'xlnet'   250 16 500 32 4320 longgpu

# wikipedia_personal
./jobs/attack/primer.sh 'abuse' 'wikipedia_personal' 'hatebase' 'bert' 250 32 500 32 4320 longgpu
./jobs/attack/primer.sh 'abuse' 'wikipedia_personal' 'hatebase' 'roberta' 250 32 500 32 4320 longgpu
./jobs/attack/primer.sh 'abuse' 'wikipedia_personal' 'hatebase' 'xlnet' 250 32 500 32 4320 longgpu

# wikipedia_aggression
./jobs/attack/primer.sh 'abuse' 'wikipedia_aggression' 'hatebase' 'roberta' 250 32 500 32 4320 longgpu

# nuclear_energy
./jobs/attack/primer.sh 'sentiment' 'nuclear_energy' 'nuclear_energy' 'bert'    250 32 500 32 4320 longgpu
./jobs/attack/primer.sh 'sentiment' 'nuclear_energy' 'nuclear_energy' 'roberta' 250 32 500 32 4320 longgpu
./jobs/attack/primer.sh 'sentiment' 'nuclear_energy' 'nuclear_energy' 'xlnet'   250 32 500 32 4320 longgpu

# climate-change_waterloo
$ccw = 'climate-change_waterloo'  # run this first
./jobs/attack/primer.sh 'sentiment' $ccw $ccw 'bert'    250 32 500 32 4320 longgpu
./jobs/attack/primer.sh 'sentiment' $ccw $ccw 'roberta' 250 32 500 32 4320 longgpu
./jobs/attack/primer.sh 'sentiment' $ccw $ccw 'xlnet'   250 32 500 32 4320 longgpu

# imdb
./jobs/attack/primer.sh 'sentiment' 'imdb' 'imdb' 'bert'    128 64 500 32 4320 longgpu
./jobs/attack/primer.sh 'sentiment' 'imdb' 'imdb' 'roberta' 128 64 500 32 4320 longgpu
./jobs/attack/primer.sh 'sentiment' 'imdb' 'imdb' 'xlnet'   128 64 500 32 4320 longgpu

# sst
./jobs/attack/primer.sh 'sentiment' 'sst' 'sst' 'bert'    128 32 500 32 4320 longgpu
./jobs/attack/primer.sh 'sentiment' 'sst' 'sst' 'roberta' 128 32 500 32 4320 longgpu
./jobs/attack/primer.sh 'sentiment' 'sst' 'sst' 'xlnet'   128 32 500 32 4320 longgpu

# fnc1
./jobs/attack/primer.sh 'fake_news' 'fnc1' 'fnc1' 'uclmr' 250 500 500 32 4320 longgpu

# run all attack variants
./jobs/attack/primer.sh 'sentiment' 'sst' 'sst' 'bert'    128 32 500 32 4320 longgpu use_variants

# template for specific configuration, replace with desired values
./jobs/attack/primer_single.sh [task] [data] [tmtd] [model] [seq_len] [batch] [toolchn] [attack] [queries] 32 4320 longgpu

# example
./jobs/attack/primer_single.sh 'abuse' 'wikipedia_personal' 'hatebase' 'roberta' 250 32 'textattack' 'input_reduction' 500 32 4320 longgpu
