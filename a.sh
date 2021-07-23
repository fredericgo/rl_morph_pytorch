envs=("ant_b" "ant5" "ant6" "ant7" "ant8")

for((i=0; i<${#envs[@]}; i++))

do
        echo ${envs[i]}
        python tools/rollout_to_buffer.py --env-name ${envs[i]}-v0\
               --actor_path data/policies/${envs[i]}_rnd.torch\
               --memory_path ${envs[i]}_rnd.memory\
               --num_steps 5000
done