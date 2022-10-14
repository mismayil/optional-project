t5_mesh_transformer \
    --model_dir=${MODEL_DIR} \
    --t5_tfds_data_dir=${DATA_DIR} \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="learning_rate_schedules/constant_0_001.gin" \
    --gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
    --gin_param="utils.run.mesh_devices = ['gpu:0']" \
    --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
    --gin_param="tsv_dataset_fn.filename = '${DATA_FILE}'"
