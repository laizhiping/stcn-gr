import os
import time
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from solver.utils import log, data_reader
from solver.stcn import stcn

class Solver():
    def __init__(self, args):
        super(Solver, self).__init__()
        self.args = args
        # self.init_seed()
        self.init_device()
        self.check_dirs()
        self.get_logger_writer()
        self.dump_info()

    def dump_info(self):
        for k, v in vars(self.args).items():
            self.logger.info(f"{k}: {v}")

        num_gestures= len(self.args.gestures)
        model = stcn.STCN(num_channels=1, num_points=self.args.num_channels, num_classes=num_gestures)
        self.logger.info(f"{model}")

    def init_device(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def init_seed(self):
        torch.cuda.manual_seed_all(1)
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def check_dirs(self):
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)
        if not os.path.exists(self.args.tb_path):
            os.makedirs(self.args.tb_path)
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

    def get_logger_writer(self):
        t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        file_name = self.args.dataset_name + f"-{t}"
        self.logger = log.get_logger(self.args.log_path, file_name+".log")
        self.writer = SummaryWriter(os.path.join(self.args.tb_path, file_name))


    def start(self, task):
        self.logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.logger.info(f"Task {task} {self.args.stage} ")
        if task == "intra_session":
            self.intra_session()
        elif task == "inter_session":
            self.inter_session()
        elif task == "inter_subject":
            self.inter_subject()
        else:
            raise ValueError

        self.logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    def inter_session(self):
        pass
    def inter_subject(self):
        pass

    def intra_session(self):
        subjects = self.args.subjects
        sessions = self.args.sessions
        gestures = self.args.gestures
        trials = self.args.trials
        train_trials = self.args.train_trials
        test_trials = self.args.test_trials

        if self.args.stage == "pretrain":
            pass
        elif self.args.stage == "train":
            accuracy = np.zeros((len(subjects), len(sessions)))
            for i, subject in enumerate(subjects):
                for j, session in enumerate(sessions):
                    self.logger.info(f"Begin training subject {subject} session {session}")
                    self.logger.info(f"{subject}, {session}, {gestures}, {train_trials}, {test_trials}")

                    num_gestures= len(self.args.gestures)
                    model = stcn.STCN(num_channels=1, num_points=self.args.num_channels, num_classes=num_gestures)
                    model = nn.DataParallel(model)
                    model.to(self.device)
                    # path = os.path.join(self.args.model_path, f"pretrain-{self.args.dataset_name}.pkl")
                    # if self.args.need_pretrain:
                    #     if os.path.exists(path):
                    #         self.model.load_state_dict(torch.load(path))
                    #         self.logger.info("load pretrain model successfully")

                    train_loader = self.get_loader([subject], [session], gestures, train_trials)
                    test_loader = self.get_loader([subject], [session], gestures, test_trials)
                    trial_acc = self.train(subject, session, model, train_loader, test_loader)
                    accuracy[i][j] = trial_acc
            self.logger.info(f"All session accuracy:\n {accuracy}")
            self.logger.info(f"All subject average accuracy:\n {accuracy.mean()}")


        elif self.args.stage == "test":
            accuracy = np.zeros((len(subjects), len(sessions)))
            for i, subject in enumerate(subjects):
                for j, session in enumerate(sessions):
                    num_gestures= len(self.args.gestures)
                    model = stcn.STCN(num_channels=1, num_points=self.args.num_channels, num_classes=num_gestures)
                    model.to(self.device)

                    path = os.path.join(self.args.model_path, f"{self.args.dataset_name}_{subject}_{session}_{self.args.window_size}_{self.args.window_step}.pkl")
                    if os.path.exists(path):
                        model.load_state_dict(torch.load(path))
                        test_trials = self.args.test_trials
                        test_loader = self.get_loader([subject], [session], gestures, test_trials)
                        criterion = torch.nn.CrossEntropyLoss()
                        metric = self.test(model, test_loader, criterion)
                        test_last_acc = metric["accuracy"]
                        self.logger.info(f"Test subject {subject} session {session}: {test_last_acc}")
                        accuracy[i][j] = test_last_acc
                    else:
                        self.logger.info(f"{path} not exists")
                        exit(1)
            self.logger.info(f"All session accuracy:\n{accuracy}\n Avg: {accuracy.mean()}")

        else:
            raise ValueError


    def get_loader(self, subjects, sessions, gestures, trials):
        dataset = data_reader.DataReader(subjects, sessions, gestures, trials, self.args)

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=0)
        return data_loader


    def train(self, subject, session, model, train_loader, test_loader):
        criterion = torch.nn.CrossEntropyLoss()
        if self.args.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones, gamma=self.args.gamma)

        self.logger.info(f"Begin train")

        path = os.path.join(self.args.model_path, f"{self.args.dataset_name}_{subject}_{session}_{self.args.window_size}_{self.args.window_step}.pkl")
        if os.path.exists(path):
            num_gestures= len(self.args.gestures)
            last_model = stcn.STCN(num_channels=1, num_points=self.args.num_channels, num_classes=num_gestures)
            last_model.to(self.device)
            last_model.load_state_dict(torch.load(path))
            metric = self.test(last_model, test_loader, criterion)
            last_acc = metric["accuracy"]
        else:
            last_acc = 0
        metric = self.test(model, test_loader, criterion)
        init_acc = metric["accuracy"]

        self.logger.info("Initial: {}, last {}".format(init_acc, last_acc))

        best_acc = last_acc
        for epoch in range(1, self.args.num_epochs+1):
            model.train()
            epoch_loss = 0
            correct = 0
            true_label = []
            pred_label = []
            train_loader.dataset.shuffle()
            for step, (x, y) in enumerate(train_loader):
                x = x.float().to(self.device)
                y = y.to(self.device)
                output = model(x)
                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                correct += (output.argmax(dim=1) == y).sum()
                true_label.extend(y.tolist())
                pred_label.extend(output.argmax(dim=1).tolist())
            scheduler.step()
            train_acc = 1.0 * correct / len(train_loader.dataset)
            metric = self.test(model, test_loader, criterion)
            if True or epoch == self.args.num_epochs:
                # self.draw_confusion(true_label, pred_label, "train")
                # self.draw_confusion(metric["true_label"], metric["pred_label"], "test")
                pass

            if metric["accuracy"] > best_acc:
                best_acc = metric["accuracy"]
                torch.save(model.state_dict(), path)

            self.writer.add_scalars(main_tag="loss", tag_scalar_dict={"train_loss": epoch_loss, "valid_loss": metric["loss"]},
                               global_step=epoch)
            self.writer.add_scalars(main_tag="accuracy", tag_scalar_dict={"train_accuracy": train_acc, "valid_acc": metric["accuracy"]},
                               global_step= epoch)
            self.logger.info("Epoch [{:5d}/{:5d}]\t train_loss: {:.08f}\t test_loss: {:.08f}\t\t train_accuracy: {:.06f} [{}/{}]\t test_accuracy: {:.06f} [{}/{}]"
                        .format(epoch, self.args.num_epochs, epoch_loss, metric["loss"],
                                train_acc, correct, len(train_loader.dataset),
                                metric["accuracy"], metric["correct"], metric["all"]))

        self.logger.info(f"Best: {best_acc}")
        return best_acc

    def test(self, model, test_loader, criterion):
        model.eval()
        metric = {}
        loss = 0
        correct = 0
        true_label = []
        pred_label = []
        for (x, y) in test_loader:
            x = x.float().to(self.device)
            y = y.to(self.device)
            output = model(x)
            loss1 = criterion(output, y)
            loss += loss1.item()
            correct += (output.argmax(dim=1) == y).sum()

            true_label.extend(y.tolist())
            pred_label.extend(output.argmax(dim=1).tolist())
            del loss1
            del output

        metric["correct"] = correct
        metric["all"] = len(test_loader.dataset)
        metric["accuracy"] = correct*1.0 / len(test_loader.dataset)
        metric["loss"] = loss
        metric["pred_label"] = pred_label
        metric["true_label"] = true_label
        return metric

    def draw_confusion(self, true_label, pred_label, title):
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import recall_score
        import matplotlib.pyplot as plt

        classes = list(set(true_label))
        classes.sort()
        confusion = confusion_matrix(true_label, pred_label)
        for first_index in range(len(confusion)):
            for second_index in range(len(confusion[first_index])):
                print("{:3d}".format(confusion[first_index][second_index]), end=" ")
            print("\n")


        # print(len(true_label))
        # print(len(pred_label))
        # plt.figure()
        # plt.imshow(confusion, cmap=plt.cm.Blues)
        # plt.title(title)
        # indices = range(len(classes))
        # plt.xticks(indices, classes)
        # plt.yticks(indices, classes)
        # plt.colorbar()
        # plt.xlabel('pred_label')
        # plt.ylabel('true_label')
        # for first_index in range(len(confusion)):
        #     for second_index in range(len(confusion[first_index])):
        #         plt.text(first_index, second_index, confusion[first_index][second_index])
        # plt.show()

    def get_optimizer(self):
        criterion = torch.nn.CrossEntropyLoss()

        if self.args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.milestones, gamma=self.args.gamma)

