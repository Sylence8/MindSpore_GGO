import numpy as np
import os
import logging
from tqdm import tqdm
from glob import glob
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Model
from mindspore.train.callback import Callback
from mindspore.dataset.engine import d_dataset

# Define the neural network model
class GGOModel(nn.Cell):
    def __init__(self, opt):
        super(GGOModel, self).__init__()
        self.model, _ = generate_model(opt)
        self.fc = nn.Dense(opt.num_classes)

    def construct(self, x):
        if opt.model == 'resnet':
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)
        else:  # densenet
            features = self.model.features(x)
            out = ops.functional.relu(features, inplace=True)
            out = ops.functional.adaptive_avg_pool3d(out, output_size=(1, 1, 1)).view(features.size(0), -1)
        out = self.fc(out)
        return out

class MyCallback(Callback):
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step = cb_params.cur_step_num
        if step % 100 == 0:
            print(f'Step {step} finished.')

def reset_weights(net):
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d) or isinstance(cell, nn.Dense):
            cell.weight.set_data(Tensor(cell.weight.default_input))
            cell.bias.set_data(Tensor(cell.bias.default_input))

if __name__ == '__main__':
    try:
        # Initialize the opts
        opt = parse_opts()
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        
        # Create the model
        net = GGOModel(opt)
        
        # Load the pretrained parameters
        model_path = opt.model_path
        if model_path == 'random':
            reset_weights(net)
        else:
            param_dict = load_checkpoint(model_path)
            load_param_into_net(net, param_dict)
        
        # Define the data loader
        test_path = opt.test_path
        test_dataset = GGODataIter(data_file=test_path, phase='test', crop_size=opt.sample_size, crop_depth=opt.sample_duration, aug=opt.aug, sample_phase=opt.sample, classifier_type=opt.clt)
        test_loader = d_dataset.MindDataset(test_dataset, num_parallel_workers=12, shuffle=True)
        
        # Create a Model for evaluation
        model = Model(net)
        model.eval()
        
        # Perform inference
        features = []
        labels = []

        print("Starting inference...")
        for data, label, names in tqdm(test_loader):
            data = data.asnumpy()
            data = Tensor(data).as_in_context(context.get_context())
            
            out = model(data)
            pred_arr = out.asnumpy()
            
            features.append(pred_arr)
            labels.append(label)

        print("Inference finished!")
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        np.save(os.path.join(excels_path, f'features_{target_dir}.npy'), features)
        np.save(os.path.join(excels_path, f'labels_{target_dir}.npy'), labels)
    except Exception as e:
        print(f"An error occurred: {e}")
