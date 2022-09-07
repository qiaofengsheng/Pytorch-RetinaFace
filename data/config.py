cfg_mobilenet = {
    'model_size': 0.25,
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,
    'stage_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'input_channels': 32,
    'output_channels': 64
}