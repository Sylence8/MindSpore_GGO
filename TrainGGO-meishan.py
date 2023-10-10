import numpy as np
from mindspore import context, nn
from mindspore.nn import DataParallel
from mindspore import Tensor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train import Model
from mindspore.ops import operations as P
from mindspore import dtype as mstype
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from DataIter import GGODataIter
from mindspore.dataset import DataLoader

from opts import parse_opts
from layers import *
from metrics import *
from collections import OrderedDict

import logging
from sklearn.metrics import confusion_matrix


def train(model, data_loader, optimizer, loss, epoch, logger):
    train_loss = []
    lr = optimizer.learning_rate.asnumpy()
    for i, (data, target, names) in enumerate(data_loader):

        data = Tensor(data, dtype=mstype.float32)
        target = Tensor(target, dtype=mstype.float32)

        out = model(data)
        cls, pen_term = loss(out, target)
        optimizer.clear_grad()
        cls.backward()
        optimizer.step()
        if out.shape[1] > 1:
            pred = nn.Sigmoid()(out)
            pred_arr = np.argmax(pred.asnumpy(), axis=1)
            label_arr = np.argmax(target.asnumpy(), axis=1)
            train_acc = np.mean(label_arr == pred_arr)
        else:
            pred = nn.Sigmoid()(out[:, :1])
            train_acc = acc_metric(pred.asnumpy(), target.asnumpy())

        train_loss.append(cls.asnumpy())

        if i % 20 == 0:
            logger.info(f"Training: Epoch {epoch}: {i}th batch, loss {cls.asnumpy():2.4f}, acc {train_acc:2.4f}, lr: {lr[0]:2.6f}")

    return np.mean(train_loss)



def test(model, data_loader, loss, epoch, lr, max_acc, max_auc, acc_max, auc_max, max_acc_auc, logger):
    test_acc = []
    loss_lst = []

    pred_lst = []
    label_lst = []
    isave = False
    isave_acc_lst = False
    isave_auc_lst = False
    isave_acc_auc_lst = False
    pred_target_dict = {}

    for i, (data, target, names) in enumerate(data_loader):
        data = Tensor(data, dtype=mstype.float32)
        target = Tensor(target, dtype=mstype.float32)
        data = data.asnumpy()
        target = target.asnumpy()
        data = Tensor(data, dtype=mstype.float32)
        target = Tensor(target, dtype=mstype.float32)
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        out = model(data)
        cls, pen_term = loss(out, target)
        loss_lst.append(cls.asnumpy())

        pred = nn.Sigmoid()(out)
        pred_arr = pred.asnumpy()
        label_arr = target.asnumpy()

        if out.shape[1] > 1:
            _acc = np.mean(np.argmax(label_arr, axis=1) == np.argmax(pred_arr, axis=1))

        else:
            _acc = acc_metric(pred_arr, label_arr)

        for i in range(pred_arr.shape[0]):
            pred_target_dict[names[i]] = [pred_arr[i], label_arr[i]]
        pred_lst.append(pred_arr)
        label_lst.append(label_arr)
        test_acc.append(_acc)

    test_loss = np.mean(loss_lst)

    label_com = np.concatenate(label_lst, axis=0)
    pred_com = np.concatenate(pred_lst, axis=0)
    num_classi = label_com.shape[1]

    if num_classi > 1:
        auc = 0
        prec = 0
        recall = 0
        f1_score = 0

        num_f = 0
        for i in range(num_classi):
            try:
                auc_t, prec_t, recall_t = confusion_matrics(
                    label_com[:, i].tolist(), pred_com[:, i].tolist())
                f1_score_t = 2 * (prec_t * recall_t) / (prec_t + recall_t)

            except:
                auc_t, prec_t, recall_t = 0, 0, 0
                f1_score_t = 0
                raise

            auc += auc_t
            prec += prec_t
            recall += recall_t
            f1_score += f1_score_t

        auc = auc / num_classi
        prec = prec / num_classi
        recall = recall / num_classi
        f1_score = f1_score / num_classi
        acc = np.mean(np.argmax(label_com, axis=1) == np.argmax(pred_com, axis=1))
    else:
        pred_lst = pred_com[:, 0].tolist()
        label_lst = label_com[:, 0].tolist()
        auc, prec, recall = confusion_matrics(label_lst, pred_lst)
        bin_pred = np.where(np.array(pred_lst) > 0.5, 1, 0)
        tn, fp, fn, tp = confusion_matrix(label_lst, bin_pred).ravel()
        sen = tp/(tp+fn+1e-8)
        spe = tn/(tn+fp+1e-8)
        HMoPN = 2*spe*recall/(spe+recall+1e-8)
        f1_score = 2 * (prec * recall) / (prec + recall+1e-8)
        acc = acc_metric(pred_com[:, 0], label_com[:, 0])

    if acc > max_acc:
        max_acc = acc
        max_auc = auc
        isave = True
        isave_acc_lst = True
    elif acc == max_acc and auc > max_auc:
        max_acc = acc
        max_auc = auc
        isave = True
        isave_acc_lst = True

    if auc > auc_max:
        auc_max = auc
        acc_max = acc
        isave = True
        isave_auc_lst = True
    elif auc == max_auc and acc > max_acc:
        auc_max = auc
        acc_max = acc
        isave = True
        isave_auc_lst = True

    if (acc + auc) > max_acc_auc:
        max_acc_auc = acc + auc
        isave = True
        isave_acc_auc_lst = True

    logger.info(f"Testing: Epoch {epoch}:{i}th batch, learning rate {lr} loss {test_loss:2.4f}, acc {acc:2.4f}, auc {auc:2.4f}, precision {prec:2.4f}, recall {recall:2.4f}")
    logger.info(f"ANDingg: The sensitivity is {sen}, f1 score is {f1_score}, specialty is {spe}, HMoPN is {HMoPN}")
    return max_acc, max_auc, acc_max, auc_max, max_acc_auc, test_loss, isave, pred_target_dict, isave_acc_lst, isave_auc_lst, isave_acc_auc_lst

if __name__ == '__main__':
    # Initialize the opts
    opt = parse_opts()
    opt = parse_opts()
    # opt.mean = get_mean(1)
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    # opt.sample_duration = 16
    opt.scales = [opt.initial_scale]
    # make directory
    target_dir = opt.save_dir
    results_dict_path = "results/" + target_dir
    if not os.path.exists(results_dict_path):
        os.makedirs(results_dict_path)
    curve_path = "metric_curves/"+target_dir
    if not os.path.exists(curve_path):
        os.makedirs(curve_path)
    save_dir = "saved_models/" + target_dir + \
        "/size_%d/" % opt.sample_size + opt.model + "_%d" % opt.model_depth
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # construct training data iterator
    # 1.logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    file_handler = logging.FileHandler(
        results_dict_path+"/logModel.txt")  # 文件句柄
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()  # 流句柄
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    context.set_context(mode=context.GRAPH_MODE, device_target=opt.device_target)  # 设置MindSpore的运行模式和设备
    
    # train_path = opt.data_dir + "train_%d.npy" %opt.num_valid
    train_path = "/data/zly/GGOPreprocess/train.npy"
    train_iter = GGODataIter(
        data_file=train_path,
        phase='train',
        crop_size=opt.sample_size,
        crop_depth=opt.sample_duration,
        aug=opt.aug,
        sample_phase=opt.sample,
        classifier_type=opt.clt)

    train_loader = DataLoader(
        train_iter,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True)
    
    # construct testing data iterator
    # test_path = opt.data_dir + "valid_%d.npy" %opt.num_valid
    test_path = "/data/zly/GGOPreprocess/valid.npy"
    test_iter = GGODataIter(data_file=test_path,
                            phase='test',
                            crop_size=opt.sample_size,
                            crop_depth=opt.sample_duration,
                            classifier_type=opt.clt)

    test_loader = DataLoader(
        test_iter,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True)

    valid_path = "/data/zly/GGOPreprocess/test.npy"
    valid_iter = GGODataIter(data_file=valid_path,
                             phase='test',
                             crop_size=opt.sample_size,
                             crop_depth=opt.sample_duration,
                             classifier_type=opt.clt)

    val_loader = DataLoader(
        valid_iter,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True)
    
    # construct testing data iterator
#     val_path = opt.data_dir + "test.npy" # %opt.num_valid
#     val_iter = GGODataIter(data_file=val_path,
#         phase='test',
#         crop_size=opt.sample_size,
#         crop_depth=opt.sample_duration,
#         classifier_type=opt.clt)

#     val_loader = DataLoader(
#         val_iter,
#         batch_size = opt.batch_size,
#         shuffle = True,
#         num_workers = 32,
#         pin_memory=True)

    model, policies = generate_model(opt)


    if opt.tf:
        logger.info("Loading pretrained parameters...")
        load_parameters = torch.load(
            "/data/zly/DeepGGO/pretrained_model/resnet-50-kinetics.pth")['state_dict']
        curmodel_dict = model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in load_parameters.items():
            name = 'module.' + k
            if 'module.conv1.weight' in name:
                new_state_dict[name] = v[:, 0:1, :, :, :]
            else:
                new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)  # 加载预训练参数
        
    model = nn.DataParallel(model)
    model.set_train(True)

    if opt.clt == 0:
        if "FP" in opt.save_dir:
            if "FP1" in opt.save_dir:
                loss = FPLoss1()
            else:
                loss = FPLoss()
        elif "RC" in opt.save_dir:
            loss = RCLoss()
        elif "CWFocalLoss" in opt.save_dir:
            logger.info("CWFocalLoss")
            loss = CWFocalLoss()
        elif "FocalLoss" in opt.save_dir:
            logger.info("FocalLoss")
            loss = FocalLoss()
        elif "AUCP" in opt.save_dir:
            loss = AUCPLoss()
        elif "CWAUCH" in opt.save_dir:
            logger.info("CWAUCH")
            loss = CWAUCHLoss(alpha=opt.alpha, lamb=opt.lamb)
        elif "AUCH" in opt.save_dir:
            logger.info("AUCH")
            loss = AUCHLoss()
        elif "CWCEL" in opt.save_dir:
            logger.info("CWCEL")
            loss = CWCELoss()
        else:
            logger.info("CEL")
            loss = CEL()

    elif opt.clt == 2:
        if "FP" in opt.save_dir:
            if "FP1" in opt.save_dir:
                loss = FPLoss1()
            else:
                loss = FPLoss()
        elif "RC" in opt.save_dir:
            loss = RCLoss()
        elif "CWFocalLoss" in opt.save_dir:
            logger.info("CWFocalLoss")
            loss = CWFocalLoss()
        elif "FocalLoss" in opt.save_dir:
            logger.info("FocalLoss")
            loss = FocalLoss()
        elif "CWAUCH" in opt.save_dir:
            logger.info("CWAUCH")
            loss = CWAUCHLoss(alpha=opt.alpha, lamb=opt.lamb)
        elif "AUCH" in opt.save_dir:
            logger.info("AUCH")
            loss = AUCHLoss(alpha=opt.alpha, lamb=opt.lamb)
        elif "AUCP" in opt.save_dir:
            loss = AUCPLoss()
        elif "CWCEL" in opt.save_dir:
            logger.info("CWCEL")
            loss = CWCELoss()
        else:
            logger.info("Loss")
            loss = Loss()

    else:
        logger.info("Modality 1")
        if "FGMCWCEL" in opt.save_dir:
            logger.info("FGMCWCEL")
            loss = FGMCWCEL()
        elif "CWFocalLoss" in opt.save_dir:
            logger.info("CWFocalLoss")
            loss = CWFocalLoss()
        elif "FocalLoss" in opt.save_dir:
            logger.info("FocalLoss")
            loss = FocalLoss()
        elif "FGMCWAUCHLoss" in opt.save_dir:
            logger.info("FGMCWAUCHLoss")
            loss = FGMCWAUCHLoss(alpha=opt.alpha, lamb=opt.lamb)
        elif "SoftmaxAUCHLoss" in opt.save_dir:
            logger.info("SoftmaxAUCHLoss")
            loss = SoftmaxAUCHLoss()
        elif "MAUCH" in opt.save_dir:
            logger.info("MAUCH")
            loss = MAUCHLoss()
        elif "CWCEL" in opt.save_dir:
            logger.info("MCWCEL")
            loss = MCWCEL()
        elif "MCWAUCHLoss" in opt.save_dir:
            logger.info("MCWAUCHLoss")
            loss = MCWAUCHLoss(alpha=opt.alpha, lamb=opt.lamb)
        elif "SoftMaxloss" in opt.save_dir:
            logger.info("SoftMaxloss")
            loss = SoftMaxLoss()
        else:
            loss = CEL()

    loss = loss.to("cuda")

    optimizer = torch.optim.Adam(
         model.parameters(),
         lr=opt.lr,
         weight_decay = 1e-4)
    
    max_acc = 0
    max_auc = 0
    acc_max = 0
    auc_max = 0
    max_acc_auc = 0

    max_acc_dict = {}
    max_acc_auc_dict = {}
    max_auc_dict = {}

    num_acc = 0
    num_auc = 0
    num_acc_auc = 0

    train_loss_list = []
    test_loss_list = []
    
    for epoch in range(opt.start_epoch, opt.epochs):
        model.train()
        train_loss = train(model, train_loader, optimizer, loss, epoch, logger)
        model.eval()
        max_acc, max_auc, acc_max, auc_max, max_acc_auc, test_loss, isave, pred_target_dict, isave_acc_lst, isave_auc_lst, isave_acc_auc_lst = test(
            model, test_loader, loss, epoch, opt.lr, max_acc, max_auc, acc_max, auc_max, max_acc_auc, logger)
        each_train_loss = {}
        each_test_loss = {}

        if isave_acc_lst:
            max_acc_dict = pred_target_dict
            logger.info("Temp is saving acc_lst")

            valid_max_acc, valid_max_auc, valid_acc_max, valid_auc_max, valid_max_acc_auc, valid_test_loss, valid_isave, valid_pred_target_dict, valid_isave_acc_lst, valid_isave_auc_lst, valid_isave_acc_auc_lst = test(
                model, val_loader, loss, epoch, opt.lr, max_acc, max_auc, acc_max, auc_max, max_acc_auc, logger)
            results = {}
            results['max_acc'] = valid_max_acc
            results['max_auc'] = valid_max_auc
            results['acc_max'] = valid_acc_max
            results['auc_max'] = valid_auc_max
            results['max_acc_auc'] = valid_max_acc_auc
            results['max_dict'] = valid_pred_target_dict
            results_dict_path = "results/" + target_dir

            if not os.path.exists(results_dict_path):
                os.makedirs(results_dict_path)
            num_acc += 1
            np.save(results_dict_path+"/valid_acc.npy", results)

        if isave_auc_lst:
            max_auc_dict = pred_target_dict
            logger.info("Temp is saving auc_lst")

            valid_max_acc, valid_max_auc, valid_acc_max, valid_auc_max, valid_max_acc_auc, valid_test_loss, valid_isave, valid_pred_target_dict, valid_isave_acc_lst, valid_isave_auc_lst, valid_isave_acc_auc_lst = test(
                model, val_loader, loss, epoch, opt.lr, max_acc, max_auc, acc_max, auc_max, max_acc_auc, logger)
            results = {}
            results['max_acc'] = valid_max_acc
            results['max_auc'] = valid_max_auc
            results['acc_max'] = valid_acc_max
            results['auc_max'] = valid_auc_max
            results['max_acc_auc'] = valid_max_acc_auc
            results['max_auc_dict'] = valid_pred_target_dict
            results_dict_path = "results/" + target_dir

            if not os.path.exists(results_dict_path):
                os.makedirs(results_dict_path)

            num_auc += 1
            np.save(results_dict_path+"/valid_auc.npy", results)

        if isave_acc_auc_lst:
            max_acc_auc_dict = pred_target_dict
            logger.info("Temp is saving acc_auc_lst")

            valid_max_acc, valid_max_auc, valid_acc_max, valid_auc_max, valid_max_acc_auc, valid_test_loss, valid_isave, valid_pred_target_dict, valid_isave_acc_lst, valid_isave_auc_lst, valid_isave_acc_auc_lst = test(
                model, val_loader, loss, epoch, opt.lr, max_acc, max_auc, acc_max, auc_max, max_acc_auc, logger)
            results = {}
            results['max_acc'] = valid_max_acc
            results['max_auc'] = valid_max_auc
            results['acc_max'] = valid_acc_max
            results['auc_max'] = valid_auc_max
            results['max_acc_auc'] = valid_max_acc_auc
            results['max_acc_auc_dict'] = valid_pred_target_dict

            num_acc_auc += 1
            np.save(results_dict_path+"/valid_acc_auc.npy", results)

        # isave = isave_acc_lst or isave_auc_lst or isave_acc_auc_lst

    if isave:
        save_path = os.path.join(save_dir, f'{epoch:03d}.ckpt')
        model.export(save_path, format='MINDIR', file_name='model')

    each_train_loss = {}
    each_test_loss = {}

    logger.info(f'Epoch {epoch}, the max acc is {max_acc:.4f}, max auc is {max_auc:.4f}, the acc_max is {acc_max:.4f}, auc_max is {auc_max:.4f}\n')

    train_loss_list.append(each_train_loss)
    test_loss_list.append(each_test_loss)

    if epoch >= 50 and epoch % 30 == 0:
        opt.lr *= 0.1
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt.lr,
            weight_decay = 1e-4)

    train_path = "/data/zly/GGOPreprocess/train.npy"
    train_iter = GGODataIter(
        data_file=train_path,
        phase='train',
        crop_size=opt.sample_size,
        crop_depth=opt.sample_duration,
        aug=opt.aug,
        sample_phase=opt.sample,
        classifier_type=opt.clt)

    train_loader = DataLoader(
        train_iter,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True)

    np.save(os.path.join(curve_path, 'train_loss.npy'), np.array(train_loss_list))
    np.save(os.path.join(curve_path, 'test_loss.npy'), np.array(test_loss_list))

    results = {}
    results['max_acc'] = max_acc
    results['max_auc'] = max_auc
    results['acc_max'] = acc_max
    results['auc_max'] = auc_max
    results['max_acc_auc'] = max_acc_auc
    results['max_dict'] = max_acc_dict
    results['max_auc_dict'] = max_auc_dict
    results['max_acc_auc_dict'] = max_acc_auc_dict

    np.save(os.path.join(results_dict_path, f'valid_{opt.num_valid}.npy'), results)
    logger.info(f'The max acc is {max_acc:.4f}, max auc is {max_auc:.4f}')
