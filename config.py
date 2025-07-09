# training config
# n_step = 1000000
n_step = 192000
scheduler_checkpoint_step = 20000
log_checkpoint_step = 1970
gradient_accumulate_every = 1
lr = 4e-5
decay = 0.9
minf = 0.5
optimizer = "adam"  # adamw or adam
n_workers = 4

# load------------------------------------------------------------------------------------------------------------
load_model = True
load_step = True
alpha = 1.0
beta = 0.0128
device = 0

# diffusion config
pred_mode = 'noise'
loss_type = "l1"
iteration_step = 20000
sample_steps = 200
embed_dim = 64
dim_mults = (1, 2, 3, 4, 5, 6)
hyper_dim_mults = (4, 4, 4)
context_channels = 3
clip_noise = "none"
val_num_of_batch = 1
additional_note = ""
vbr = False
context_dim_mults = (1, 2, 3, 4)
sample_mode = "ddim"
var_schedule = "linear"  # beta取法
aux_loss_type = "lpips"
compressor = "big"

# dataset----------------------------------------------------------------------------------------
data_path = 'cityscape_dataset'
dataset_name = "Cityscape"
# data_path = './'
# dataset_name = "KITTI_Stereo"
resize = [128,256]
# data config
data_config = {
    "dataset_name": "vimeo",
    "data_path": "*",
    "sequence_length": 1,
    "img_size": 256,
    "img_channel": 3,
    "add_noise": False,
    "img_hz_flip": False,
}

train_batch_size = 16



# result_root = "./kitti_result_distribute_dual"
# train_path = './kitti_result_distribute_dual/train_log_alpha0.9_beta0.1024'
# val_path = './kitti_result_distribute_dual/val_log_alpha0.9_beta0.1024'
#---------------------------------------------------------------------------------------
# result_root = "./kitti_result_distribute_froze"
# train_path = './kitti_result_distribute_froze/train_log_alpha1.0_beta0.0128'
# val_path = './kitti_result_distribute_froze/val_log_alpha1.0_beta0.0128'
# test_path = './kitti_result_distribute_froze/LP_0.0256'

result_root = "cityscape_result_distribute_froze"
# train_path = './cityscape_result_distribute_froze/train_log_alpha1.0_beta0.0128'
# val_path = './cityscape_result_distribute_froze/val_log_alpha1.0_beta0.0064'
test_path = './cityscape_result_distribute_froze/LP_0.0128___'

# result_root = "cityscape_result_CDC_froze"
# train_path = './cityscape_result_CDC_froze/train_log_alpha1.0_beta0.0128'
# val_path = './cityscape_result_CDC_froze/val_log_alpha1.0_beta0.0128'
# test_path = './cityscape_result_CDC_froze/alpha_0.0128'

# result_root = "./kitti_result_distribute_froze"
# train_path = './kitti_result_distribute_froze/train_log_alpha1.0_beta0.0064'
# val_path = './kitti_result_distribute_froze/val_log_alpha1.0_beta0.0064'
# test_path = './kitti_result_distribute_froze/test_log_alpha1.0_beta0.0512'


tensorboard_root = "./result/tensorboard"
