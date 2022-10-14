${T5_ENV}/bin/t5_mesh_transformer \
    --model_dir="${T5_DIR}/model/large/" \
    --t5_tfds_data_dir=${DATA_DIR} \
    --gin_file="${T5_DIR}/model/large/operative_config.gin" \
    --gin_file="learning_rate_schedules/constant_0_001.gin" \
    --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
    --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
    --gin_param="tsv_dataset_fn.filename = '${DATA_FILE}'"
