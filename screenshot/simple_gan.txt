'''
##########################
Load galaxy image data
##########################

++ X_img.shape: (1866, 50, 50, 3)

++ image_size: 50

##########################
Load targets
##########################

++ HSC_ids: [43158176442374224 43158176442374373 43158176442374445 ...
 43159155694916013 43159155694916476 43159155694917496]

++ df.head() :
                   photo_z  log_mass
HSC_id                              
43158584464268619   0.4810   9.17510
43158721903220850   0.0050   7.64999
43158447025313043   0.0857   7.94762
43158584464268728   0.1315   8.21643
43158447025292832   0.0701   7.86620

means:  [0.21093612 8.62739865]

std:    [0.30696933 0.63783586]

++ y_standard.shape : (1866, 2)

##########################
Run GAN
##########################
2019-04-16 21:23:07.863074: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-04-16 21:23:07.950511: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:964] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-04-16 21:23:07.951222: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties: 
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 0000:00:04.0
totalMemory: 11.17GiB freeMemory: 11.09GiB
2019-04-16 21:23:07.951259: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-04-16 21:23:08.274445: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-04-16 21:23:08.274529: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2019-04-16 21:23:08.274545: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2019-04-16 21:23:08.275165: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10747 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
 [*] Reading checkpoints...
 [*] Failed to find a checkpoint
 [!] Load failed...
Epoch: [ 0] [  28/  29] time: 9.3616, d_loss: 107.78681183, g_loss: 342.433349610
Epoch: [ 1] [  28/  29] time: 17.6019, d_loss: 164.63529968, g_loss: 518.520141608
Epoch: [ 2] [  28/  29] time: 25.8896, d_loss: 73.43810272, g_loss: 301.975402836
Epoch: [ 3] [  28/  29] time: 33.8825, d_loss: 7.49694204, g_loss: 744.85888672734
Epoch: [ 4] [  28/  29] time: 41.5999, d_loss: 24.78225708, g_loss: 302.64285278
Epoch: [ 5] [  28/  29] time: 49.6418, d_loss: 20.41371155, g_loss: 341.33825684
Epoch: [ 6] [  28/  29] time: 57.4665, d_loss: 10.24314880, g_loss: 662.3884277384
Epoch: [ 7] [  28/  29] time: 65.9247, d_loss: 5.41382122, g_loss: 912.6527099609
Epoch: [ 8] [  28/  29] time: 73.7139, d_loss: 5.73435831, g_loss: 940.350585942
Epoch: [ 9] [  28/  29] time: 81.9780, d_loss: 12.29454422, g_loss: 730.17492676
...
..
.
'''