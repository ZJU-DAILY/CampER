import copy
import pickle
import struct
import sys
import time
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler
from model.base_model_1 import*
from utils import *
from data_ol import MyDataset
import torch.utils.data as Data
import socket


def train(args, loader2, eval_loader2, set_size=0, start=0):
    begin = time.time()
    fix_seed(args.seed)
    PROJ_DIR = abspath(dirname(__file__))
    task = args.task
    task = task.replace('/', '_')
    if not os.path.exists(join(PROJ_DIR, 'flog2')):
        os.mkdir(join(PROJ_DIR, 'flog2'))
    if not os.path.exists(join(PROJ_DIR, 'flog2', task)):
        os.mkdir(join(PROJ_DIR, 'flog2', task))
    id2tokens2 = id2tokens(args.path2)
    index2id = dict()
    id2index = dict()
    topk = args.topk

    device = torch.device(args.device)
    model = BertEncoder(args).to(device)  # encoder q
    _model = BertEncoder(args).to(device)  # encoder k
    _model.update(model)  # moto update
    iteration = 0
    lr = args.lr
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    # if args.fp16:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "localhost"
    try:
        s.connect((host, args.port))  #
    except Exception:
        print('[!] Server not found or not open')
        sys.exit()
    ids_2, vector_2 = list(), list()
    with torch.no_grad():
        model.eval()
        for sample_id_2, (tuple_token_2, tuple_id_2) in enumerate(eval_loader2):
            tuple_vector_2 = model(tuple_token_2)
            tuple_vector_2 = tuple_vector_2.squeeze().detach().cpu().numpy()
            vector_2.append(tuple_vector_2)
            tuple_id_2 = tuple_id_2.squeeze().tolist()
            if isinstance(tuple_id_2, int):
                tuple_id_2 = [tuple_id_2]
            ids_2.extend(tuple_id_2)

    v2 = np.vstack(vector_2).astype(np.float32)
    v2 = preprocessing.normalize(v2)
    self_sim_score = torch.tensor(v2.dot(v2.T))
    self_dist, self_topk = torch.topk(self_sim_score, k=topk, dim=1)
    for i, x in enumerate(ids_2):
        index2id[i] = x
        id2index[x] = i

    ids_2, vector_2 = list(), list()


    for round_id in range(start, args.rounds):
        # adjust_learning_rate(optimizer, round_id, lr)
        epoch_loss = []

        scaler = GradScaler()
        for iter in range(args.local_ep):
            adjust_learning_rate(optimizer, int(round_id) * int(iter+1), lr)
            batch_loss = []
            for batch_id, batch in tqdm(enumerate(loader2)):
                tuple_id = batch[-1]
                index = list(np.vectorize(id2index.get)(tuple_id.numpy()))
                topk_index = self_topk.numpy()[index]
                topk_index = np.delete(topk_index, 0, axis=1)
                topk_id = np.vectorize(index2id.get)(topk_index)
                topk_id = topk_id.reshape(topk_id.shape[0] * topk_id.shape[1])
                topk_tokens = list(map(id2tokens2.get, topk_id))
                x = np.array(topk_tokens)

                # rand_id = random.sample(id2tokens2.keys(), topk)
                # topk_tokens = list(map(id2tokens2.get, rand_id))
                # x = np.array(topk_tokens)

                if len(batch) == 3:
                    tuple_tokens, tuple_pos_tokens, _ = batch  # T(batch_size, 256) T(batch_size, neg_num, 256) T(batch_size)
                    pos1 = tuple_tokens
                    pos2 = tuple_pos_tokens

                else:
                    tuple_tokens, tuple_pos_tokens, tuple_neg_tokens, _ = batch
                    pos1 = tuple_tokens
                    pos2 = tuple_pos_tokens
                    neg_dk = tuple_neg_tokens

                optimizer.zero_grad()
                with torch.no_grad():
                    _model.eval()
                    pos_2 = _model(pos2.squeeze(0))
                    hard_neg = torch.LongTensor(x).to(device)
                    neg_value = _model(hard_neg)
                    del pos2
                with autocast():
                    pos_1 = model(pos1.squeeze(0))
                    # contrastive
                    loss = model.contrastive_loss(pos_1, pos_2, neg_value)

                del pos_1
                del pos_2
                del neg_value
                iteration += 1

                scaler.scale(loss).backward()
                if args.dp_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.dp_clip, norm_type=1)
                scaler.step(optimizer)
                scaler.update()
                _model.update(model)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        match_dic1 = dict()
        match_dic2 = dict()

        def match_loader(path):
            match = []
            p = open(path, 'r')
            i = 0
            for line in p:
                id_1, id_2 = line.strip().split(' ')
                match.append((int(id_1), int(id_2)))
                i += 1
            return match

        TrueMatch = match_loader(args.match_path)

        for (e1, e2) in TrueMatch:
            match_dic1[e1] = e2
            match_dic2[e2] = e1
        if args.dp_epsilon != -1:
            add_noise(model, args.dp_mechanism, args.lr, args.dp_clip, args.dp_epsilon, args.batch_size, dp_delta=None)

        model_weight = model.state_dict()


        w = pickle.dumps(model_weight)
        need_recv_size = len(w)

        recv = b""
        while need_recv_size > 0:
            x = s.recv(min(0xfffffffffff, need_recv_size))
            recv += x
            need_recv_size -= len(x)
        begin = time.time()
        s.sendall(w)

        weight_A = pickle.loads(recv)

        w_avg = copy.deepcopy(model_weight)
        for key in model_weight.keys():
            w_avg[key] += weight_A[key]
            w_avg[key] = torch.div(w_avg[key], 2)
        # update parameters
        model.load_state_dict(w_avg)
        _model.update(model)

        ids_2, vector_2 = list(), list()

        with torch.no_grad():
            model.eval()
            for sample_id_2, (tuple_token_2, tuple_id_2) in enumerate(eval_loader2):
                tuple_vector_2 = model(tuple_token_2)
                tuple_vector_2 = tuple_vector_2.squeeze().detach().cpu().numpy()
                vector_2.append(tuple_vector_2)
                tuple_id_2 = tuple_id_2.squeeze().tolist()
                if isinstance(tuple_id_2, int):
                    tuple_id_2 = [tuple_id_2]
                ids_2.extend(tuple_id_2)
        v2 = np.vstack(vector_2).astype(np.float32)
        v2 = preprocessing.normalize(v2)


        if round_id == int(args.rounds)-1:
            li = []
            for i, x in enumerate(ids_2):
                if x not in match_dic2:
                    li.append(i)
            v2_ = np.delete(v2, li, axis=0)
            np.savetxt("./round_{}_WA_FED_Amazon.txt".format(round_id), v2_, fmt='%f', delimiter=',')

        self_sim_score = torch.tensor(v2.dot(v2.T))
        self_dist, self_topk = torch.topk(self_sim_score, k=topk, dim=1)
        v2 = pickle.dumps(v2)
        header = struct.pack('i', len(v2))
        s.send(header)
        s.sendall(v2)

        ids_2 = pickle.dumps(ids_2)
        header = struct.pack('i', len(ids_2))
        s.send(header)
        s.sendall(ids_2)

    s.close()
    end = time.time()


    print("running time: ", end - begin)





