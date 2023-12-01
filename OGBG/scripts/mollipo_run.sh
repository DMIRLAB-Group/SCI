python ../mollipo.py --dataset_name lipo --device 0 --epochs 100 --batch_size 200 --p_dropout 0.2 --fingerprint_dim 200 --emb_dim1 100 --emb_dim2 39 --radius 3 --T 2 --per_task_output_units_num 2 --learning_rate 2.8 --weight_decay 5 --gh_sparity_para 1.0 --g1_loss_para 1.0 --gh_sparity_loss_para 1 --gn_sparity_loss_para 1 --Lr_para 2 --x1_loss_para 10 --KL_para 1 --k1_loss_para 1.0 --yg_loss_para 1 --random_seed 66 --path ../data/ogbg_mollipo --save_path ../saved_models/model_
