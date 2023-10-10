import numpy as np
from mindspore import context, nn, Model, Tensor, ops, Parameter
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.transforms.py_transforms import Compose
from mindspore.common.initializer import TruncatedNormal
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.ops import operations as P

from data_iter import GGODataIter
from model import generate_model
from opts import parse_opts
from layers import *
from metrics import *
from collections import OrderedDict

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

def train(model, data_loader, optimizer, loss, epoch, logger):
    train_loss = []
    lr = optimizer.get_lr()[0]

    for i, (data, target, names) in enumerate(data_loader):
        data = Tensor(data.numpy(), dtype=mindspore.float32)
        target = Tensor(target.numpy(), dtype=mindspore.float32)

        out = model(data)
        cls, pen_term = loss(out, target)

        optimizer.clear_grad()
        cls.backward()
        optimizer.step()

        if out.shape[1] > 1:
            pred = ops.Sigmoid()(out)
            pred_arr = pred.argmax(axis=1).asnumpy()
            label_arr = target.argmax(axis=1).asnumpy()
            train_acc = np.mean(label_arr == pred_arr)
        else:
            pred = ops.Sigmoid()(out[:, :1])
            train_acc = acc_metric(pred.asnumpy(), target.asnumpy())

        train_loss.append(cls.asnumpy().item())

        if i % 20 == 0:
            logger.info(f"Training: Epoch {epoch}: {i}th batch, loss {cls.asnumpy().item():.4f}, acc {train_acc:.4f}, lr: {lr:.6f}")

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
        data = Tensor(data.numpy(), dtype=mindspore.float32)
        target = Tensor(target.numpy(), dtype=mindspore.float32)

        data = ops.Cast()(data, mindspore.float32)
        target = ops.Cast()(target, mindspore.float32)

        data = data.to("cuda")
        target = target.to("cuda")

        out = model(data)
        cls, pen_term = loss(out, target)
        loss_lst.append(cls.asnumpy())

        pred = ops.Sigmoid()(out)
        pred_arr = pred.asnumpy()
        label_arr = target.asnumpy()

        if out.shape[1] > 1:
            _acc = np.mean(np.argmax(label_arr, axis=1) == np.argmax(pred_arr, axis=1))
        else:
            _acc = acc_metric(pred_arr[:, 0], label_arr[:, 0])

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

    logger.info(f"Testing: Epoch {epoch}:{i}th batch, learning rate {lr:.6f} loss {test_loss:.4f}, acc {acc:.4f}, auc {auc:.4f}, precision {prec:.4f}, recall {recall:.4f}")
    logger.info(f"ANDingg: The sensitivity is {sen:.4f}, f1 score is {f1_score:.4f}, specialty is {spe:.4f}, HMoPN is {HMoPN:.4f}")

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

    train_path = "/data/zly/dataset/train.npy"
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
    test_path = "/data/zly/dataset/valid.npy"
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

    valid_path = "/data/zly/dataset/test.npy"
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

    model, policies = generate_model(opt)


    if opt.tf:
        logger.info("Loading pretrained parameters...")
        load_parameters = torch.load(
            "/data/zly/DeepGGO/pretrained_model/resnet-50-kinetics.pth")['state_dict']
        # model.load_state_dict(load_parameters)
        curmodel_dict = model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in load_parameters.items():
            #import pdb;pdb.set_trace()
            name = 'module.' + k
            if 'module.conv1.weight' in name:
                new_state_dict[name] = v[:, 0:1, :, :, :]
            else:
                new_state_dict[name] = v

    model = nn.DataParallel(model.cuda())

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
            # logger.info("Loss")
            # loss = Loss()
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
            # logger.info("CEL")
            # loss = CEL()

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

    loss = loss.cuda()

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=opt.lr,
    #     momentum=0.9,
    #     weight_decay=1e-4)

    optimizer = nn.Adam(
        params=model.trainable_params(),
        learning_rate=opt.lr,
        weight_decay=1e-4)

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
        state_dict = model.parameters_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].asnumpy()

        ckpt_name = os.path.join(save_dir, '{}.ckpt'.format(epoch))
        mox.file.copy_parallel(ckpt_name, os.path.join(save_dir, 'checkpoint/'))

    # writer.add_scalar('metric_curves/loss/train',train_loss,epoch)
    # writer.add_scalar('metric_curves/loss/test',test_loss,epoch)
    each_train_loss[epoch] = train_loss
    each_test_loss[epoch] = test_loss

    logger.info("Epoch %d, the max acc is %2.4f, max auc is %2.4f, the acc_max is %2.4f, auc_max is %2.4f" % (
        epoch, max_acc, max_auc, acc_max, auc_max))
    logger.info("\n")

    train_loss_list.append(each_train_loss)
    test_loss_list.append(each_test_loss)

    if epoch >= 50 and epoch % 30 == 0:
        opt.lr = opt.lr * 0.1
        optimizer = nn.Adam(
            params=model.trainable_params(),
            learning_rate=opt.lr,
            weight_decay=1e-4)

    train_path = "/data/zly/dataset/train.npy"
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

    np.save(curve_path+"/train_loss.npy", train_loss_list)
    np.save(curve_path+"/test_loss.npy", test_loss_list)

    results = {}
    results['max_acc'] = max_acc
    results['max_auc'] = max_auc
    results['acc_max'] = acc_max
    results['auc_max'] = auc_max
    results['max_acc_auc'] = max_acc_auc
    results['max_dict'] = max_acc_dict
    results['max_auc_dict'] = max_auc_dict
    results['max_acc_auc_dict'] = max_acc_auc_dict

    np.save(results_dict_path+"/valid_%d.npy" % opt.num_valid, results)
    logger.info("The max acc is %2.4f, max auc is %2.4f" % (max_acc, max_auc))
