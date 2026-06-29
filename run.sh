
for use_gt in False
do
    for reward_type in "discrepancy" # standard -> standard reward, discrepancy -> discrepancy reward
    do

        for w in 0 
        do
            for model in   "deeplabv3"
            do  
                for dataset in "VOC2012"
                do
                    for action_space in "reg"  # standard -> RFPAR, reg -> SegPAR
                    do
                        for p in 10e-4 # sparsity level
                        do
                            echo "Running ${dataset} ${model} with bound=100"
                            python main.py --config configs_attack/${dataset}/config_${model}.py \
                                --majority None \
                                --bound 100 \
                                --patient 1 \
                                --process_name "${model}_${dataset}_attack_100_use_gt_${use_gt}_model_info_${w}" \
                                --cuda_device "cuda:0" \
                                --use_gt $use_gt \
                                --reward_type $reward_type \
                                --show_effect True \
                                --w $w \
                                --rl_learning_rate 1e-05\
                                --it_max 1000\
                                --action_space $action_space\
                                --batch 4 \
                                --attack_pixel $p
                        done
                    done
                done
            done
        done
    done
done

