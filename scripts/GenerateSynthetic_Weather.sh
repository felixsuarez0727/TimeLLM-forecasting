input_len=128
steps=96

python generate_synthetic.py \
  --task_name long_term_forecast \
  --data_path './dataset/weather/weather.csv' \
  --input_len $input_len \
  --generate_steps $steps \
  --model_checkpoint './checkpoints/long_term_forecast_weather_512_96_TimeLLM_Weather_ftM_sl128_ll48_pl96_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-TimeLLM-Weather-LowMem/checkpoint' \
  --output_path './synthetic_output' \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --pred_len 96 \
  --d_model 16 \
  --n_heads 8 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 32 \
  --moving_avg 25 \
  --factor 3 \
  --dropout 0.1 \
  --embed 'timeF' \
  --activation 'gelu' \
  --output_attention False \
  --patch_len 16 \
  --stride 8 \
  --prompt_domain 0 \
  --llm_model 'GPT2' \
  --llm_dim 768 \
  --llm_layers 4 \
  --features 'M'
