python main.py --backbone gcn  --dataset squirrel --lr 0.001 --num_layers 3 \
    --hidden_channels 64 --weight_decay 5e-4 --dropout 0.3   \
    --method hyp_transformer --ours_layers 1 --use_graph --ours_use_act --ours_use_residual --num_heads 1 \
    --alpha 0.5  --device 0 --runs 1 --seed 1234

python main.py --backbone gcn --dataset chameleon --lr 0.001 --num_layers 5 \
    --hidden_channels 64 --weight_decay 0.001 --dropout 0  \
    --method hyp_transformer --ours_layers 1 --use_graph  --num_heads 1 --ours_use_residual --ours_use_act \
    --alpha 0.5 --ours_dropout 0 --ours_weight_decay 0 --device 0 --runs 1 --epochs 500 --patience 400 --seed 1234

python main.py --backbone gcn --dataset film --lr 0.02 --num_layers 9 \
    --hidden_channels 64 --weight_decay 0.001 --dropout 0.2  \
    --method hyp_transformer --ours_layers 1 --use_graph  --num_heads 1 --ours_use_residual --ours_use_act \
    --alpha 0.5 --ours_dropout 0 --ours_weight_decay 0 --device 0 --runs 1 --epochs 500 --patience 400 --seed 1234 