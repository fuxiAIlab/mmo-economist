for seed in 0 1 2 3 4
do
  for cfg in config_50_50 config_20_80 config_80_20
    do
      for adjust in  planner
      do
	      python tf_train.py --adj $adjust --cfg $cfg --num-iter 300 --seed $seed --phase 1
	      python tf_train.py --adj $adjust --cfg $cfg --num-iter 700 --seed $seed --phase 2 --restore runs/phase_1_planner_seed_$seed/$cfg/iter_299/checkpoint_300/checkpoint-300
	    for adjust in fixed random-asy random-syn greedy-equ greedy-pro
	    do
	      python tf_train.py --adj $adjust --cfg $cfg --num-iter 1000 --seed $seed --phase 1
	    done
    done
  done
done
