# wikipedia
./jobs/ava_jobs_attack/primer.sh 'abuse' 'wikipedia' 'wikipedia' 'bert'    250 32 500 ava-s0
./jobs/ava_jobs_attack/primer.sh 'abuse' 'wikipedia' 'wikipedia' 'roberta' 250 32 500 ava-s0
./jobs/ava_jobs_attack/primer.sh 'abuse' 'wikipedia' 'wikipedia' 'xlnet'   250 16 500 ava-s0

# hatebase
./jobs/ava_jobs_attack/primer.sh 'abuse' 'hatebase' 'hatebase' 'bert'    250 32 500 ava-s0
./jobs/ava_jobs_attack/primer.sh 'abuse' 'hatebase' 'hatebase' 'roberta' 250 32 500 ava-s0
./jobs/ava_jobs_attack/primer.sh 'abuse' 'hatebase' 'hatebase' 'xlnet'   250 16 500 ava-s0

# civil_comments
./jobs/ava_jobs_attack/primer.sh 'abuse' 'civil_comments' 'civil_comments' 'bert'    250 32 500 ava-s0
./jobs/ava_jobs_attack/primer.sh 'abuse' 'civil_comments' 'civil_comments' 'roberta' 250 32 500 ava-s0
./jobs/ava_jobs_attack/primer.sh 'abuse' 'civil_comments' 'civil_comments' 'xlnet'   250 16 500 ava-s0

# wikipedia_personal
./jobs/ava_jobs_attack/primer.sh 'abuse' 'wikipedia_personal' 'hatebase' 'roberta' 250 32 500 ava-s0

# wikipedia_aggression
./jobs/ava_jobs_attack/primer.sh 'abuse' 'wikipedia_aggression' 'hatebase' 'roberta' 250 32 500 ava-s0

# nuclear_energy
./jobs/ava_jobs_attack/primer.sh 'sentiment' 'nuclear_energy' 'nuclear_energy' 'bert'    250 32 500 ava-s0
./jobs/ava_jobs_attack/primer.sh 'sentiment' 'nuclear_energy' 'nuclear_energy' 'roberta' 250 32 500 ava-s0
./jobs/ava_jobs_attack/primer.sh 'sentiment' 'nuclear_energy' 'nuclear_energy' 'xlnet'   250 32 500 ava-s0

# climate-change_waterloo
$ccw = 'climate-change_waterloo'  # run this first
./jobs/ava_jobs_attack/primer.sh 'sentiment' $ccw $ccw 'bert'    250 32 500 ava-s0
./jobs/ava_jobs_attack/primer.sh 'sentiment' $ccw $ccw 'roberta' 250 32 500 ava-s0
./jobs/ava_jobs_attack/primer.sh 'sentiment' $ccw $ccw 'xlnet'   250 32 500 ava-s0

# imdb
./jobs/ava_jobs_attack/primer.sh 'sentiment' 'imdb' 'imdb' 'bert'    128 64 500 ava-s0
./jobs/ava_jobs_attack/primer.sh 'sentiment' 'imdb' 'imdb' 'roberta' 128 64 500 ava-s0
./jobs/ava_jobs_attack/primer.sh 'sentiment' 'imdb' 'imdb' 'xlnet'   128 64 500 ava-s0

# sst
./jobs/ava_jobs_attack/primer.sh 'sentiment' 'sst' 'sst' 'bert'    128 32 500 ava-s0
./jobs/ava_jobs_attack/primer.sh 'sentiment' 'sst' 'sst' 'roberta' 128 32 500 ava-s0
./jobs/ava_jobs_attack/primer.sh 'sentiment' 'sst' 'sst' 'xlnet'   128 32 500 ava-s0

# fnc1
./jobs/ava_jobs_attack/primer.sh 'fake_news' 'fnc1' 'fnc1' 'uclmr' 250 500 500 ava-s0


# template for specific configuration, replace with desired values
./jobs/ava_jobs_attack/primer_single.sh [task] [data] [tmtd] [model] [seq_len] [batch] [toolchn] [attack] [queries] [nodelist]

# example
./jobs/ava_jobs_attack/primer_single.sh 'abuse' 'wikipedia_personal' 'hatebase' 'roberta' 250 32 'textattack' 'bae' 500 ava-s0
