import copy
import pickle
import struct

from torch.cuda.amp import GradScaler
from model.base_model import *

from utils import *
import socket


def train(args, loader1, eval_loader1, set1_id2t, set2_id2t, set_size=0, start=0):
    begin = time.time()
    fix_seed(args.seed)
    PROJ_DIR = abspath(dirname(__file__))
    task = args.task
    task = task.replace('/', '_')

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
    id2tokens1 = id2tokens(args.path1)

    match_dic1 = dict()
    match_dic2 = dict()
    for (e1, e2) in TrueMatch:
        match_dic1[e1] = e2
        match_dic2[e2] = e1

    if not os.path.exists(join(PROJ_DIR, 'exp_log')):
        os.mkdir(join(PROJ_DIR, 'exp_log'))

    if not os.path.exists(join(PROJ_DIR, 'exp_log', task)):
        os.mkdir(join(PROJ_DIR, 'exp_log', task))

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

    host = "localhost"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, args.port))
    s.listen(5)
    print("listening...")
    conn, addr = s.accept()  # acc connect
    print('[+] Connected with', addr)

    ids_1, vector_1 = list(), list()
    with torch.no_grad():
        model.eval()
        for sample_id_1, (tuple_token_1, tuple_id_1) in enumerate(eval_loader1):
            tuple_vector_1 = model(tuple_token_1)
            tuple_vector_1 = tuple_vector_1.squeeze().detach().cpu().numpy()
            vector_1.append(tuple_vector_1)
            tuple_id_1 = tuple_id_1.squeeze().tolist()
            if isinstance(tuple_id_1, int):
                tuple_id_1 = [tuple_id_1]
            ids_1.extend(tuple_id_1)

    v1 = np.vstack(vector_1).astype(np.float32)
    v1 = preprocessing.normalize(v1)
    self_sim_score = torch.tensor(v1.dot(v1.T))
    self_dist, self_topk = torch.topk(self_sim_score, k=topk, dim=1)

    for i, x in enumerate(ids_1):
        index2id[i] = x
        id2index[x] = i

    filename = os.path.join(PROJ_DIR, 'exp_log',
                            task + '/rand={}_bsize={}_rounds={}_'
                                   'clip={}_epsilon={}.txt'.format(
                                args.topk,
                                str(args.batch_size),
                                str(args.rounds),
                                str(args.dp_clip),
                                str(args.dp_epsilon)
                            ))

    for round_id in range(start, args.rounds):


        epoch_loss = []

        scaler = GradScaler()
        # Local Update
        for iter in range(args.local_ep):
            adjust_learning_rate(optimizer, int(round_id) * int(iter + 1), lr)
            batch_loss = []
            for batch_id, batch in tqdm(enumerate(loader1)):  # data from table1
                tuple_id = batch[-1]
                index = list(np.vectorize(id2index.get)(tuple_id.numpy()))
                topk_index = self_topk.numpy()[index]
                topk_index = np.delete(topk_index, 0, axis=1)
                topk_id = np.vectorize(index2id.get)(topk_index)
                topk_id = topk_id.reshape(topk_id.shape[0] * topk_id.shape[1])
                topk_tokens = list(map(id2tokens1.get, topk_id))
                x = np.array(topk_tokens)
                # rand_id = random.sample(id2tokens1.keys(), topk)
                # topk_tokens = list(map(id2tokens1.get, rand_id))
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.dp_clip,  norm_type=1)
                scaler.step(optimizer)
                scaler.update()

                _model.update(model)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if args.dp_epsilon != -1:
            add_noise(model, args.dp_mechanism, args.lr, args.dp_clip, args.dp_epsilon, args.batch_size, dp_delta=None)

        model_weight = model.state_dict()
        local_loss = sum(epoch_loss) / len(epoch_loss)

        w = pickle.dumps(model_weight)
        need_recv_size = len(w)
        conn.sendall(w)
        recv = b""
        while need_recv_size > 0:
            x = conn.recv(min(0xffffffff, need_recv_size))
            recv += x
            need_recv_size -= len(x)

        weight_B = pickle.loads(recv)

        w_avg = copy.deepcopy(model_weight)
        for key in model_weight.keys():
            w_avg[key] += weight_B[key]
            w_avg[key] = torch.div(w_avg[key], 2)
        # update parameters
        model.load_state_dict(w_avg)

        _model.update(model)

        # evaluate
        print('round: {} loss: {}'.format(round_id, local_loss))
        with open(filename, 'a+') as f:
            f.write('round: {} loss: {}\n'.format(round_id, local_loss))

        ids_1, vector_1 = list(), list()
        with torch.no_grad():
            model.eval()
            for sample_id_1, (tuple_token_1, tuple_id_1) in enumerate(eval_loader1):
                tuple_vector_1 = model(tuple_token_1)
                tuple_vector_1 = tuple_vector_1.squeeze().detach().cpu().numpy()
                vector_1.append(tuple_vector_1)
                tuple_id_1 = tuple_id_1.squeeze().tolist()
                if isinstance(tuple_id_1, int):
                    tuple_id_1 = [tuple_id_1]
                ids_1.extend(tuple_id_1)
        v1 = np.vstack(vector_1).astype(np.float32)
        v1 = preprocessing.normalize(v1)

        self_sim_score = torch.tensor(v1.dot(v1.T))
        self_dist, self_topk = torch.topk(self_sim_score, k=topk, dim=1)

        # recv v2 from party2
        header_struct = conn.recv(4)  # 4 length
        unpack_res = struct.unpack('i', header_struct)
        need_recv_size = unpack_res[0]
        recv = b""
        while need_recv_size > 0:
            x = conn.recv(min(0xfffffffffff, need_recv_size))
            recv += x
            need_recv_size -= len(x)
        v2 = pickle.loads(recv)

        header_struct = conn.recv(4)  # 4 length
        unpack_res = struct.unpack('i', header_struct)
        need_recv_size = unpack_res[0]
        recv = b""
        while need_recv_size > 0:
            xx = conn.recv(min(0xfffffffffff, need_recv_size))
            recv += xx
            need_recv_size -= len(xx)
        ids_2 = pickle.loads(recv)

        inverse_ids_1, inverse_ids_2 = dict(), dict()
        for idx, _id in enumerate(ids_1):
            inverse_ids_1[_id] = idx  # entity id to index
        for idx, _id in enumerate(ids_2):
            inverse_ids_2[_id] = idx  # entity id to index

        sim_score = torch.tensor(v1.dot(v2.T))
        distA, topkA = torch.topk(sim_score, k=2, dim=1)
        distB, topkB = torch.topk(sim_score, k=2, dim=0)
        topkB = topkB.t()

        lenA = topkA.shape[0]
        PseudoMatch = []
        for e1_index in range(lenA):
            e2_index = topkA[e1_index][0].item()
            if e1_index == topkB[e2_index][0].item():
                PseudoMatch.append((ids_1[e1_index], ids_2[e2_index]))

        match_dic = {}  # dict A->B
        invers_match_dic = {}  # dict B->A
        PseudoMatch_dic = {}  # dict A->B
        invers_PseudoMatch_dic = {}  # dict B->A
        len_pm = len(PseudoMatch)
        for pair in TrueMatch:  # list
            if pair[0] not in match_dic:
                match_dic[pair[0]] = []  # one left may be matched to multi-right entity
            match_dic[pair[0]].append(pair[1])

        for pair in TrueMatch:
            if pair[1] not in invers_match_dic:
                invers_match_dic[pair[1]] = []  # one left may be matched to multi-right entity
            invers_match_dic[pair[1]].append(pair[0])

        for pair in PseudoMatch:
            PseudoMatch_dic[pair[0]] = [pair[1]]

        for pair in PseudoMatch:
            invers_PseudoMatch_dic[pair[1]] = [pair[0]]

        wrong_match = 0
        extra_match = 0
        candidate = 0
        tp = 0
        tp_sim = []

        PseudoMatch_dic_copy = PseudoMatch_dic.copy()
        for e1 in PseudoMatch_dic:
            e2 = PseudoMatch_dic[e1][0]
            if e1 in match_dic:
                if e2 in match_dic[e1]:
                    tp += 1
                    tp_sim.append(sim_score[inverse_ids_1[e1]][inverse_ids_2[e2]])
                else:
                    wrong_match += 1
                candidate += 1
            else:
                if e2 in invers_match_dic:
                    wrong_match += 1
                    candidate += 1
                else:
                    extra_match += 1

        try:
            pre = round(tp / candidate, 3)
        except ZeroDivisionError:
            pre = 0.0
        recall = round(tp / len(TrueMatch), 3)
        try:
            f1 = round(2 * pre * recall / (pre + recall), 3)
        except ZeroDivisionError:
            f1 = 0.0
        print("TrueMatch.Sie: ", len(TrueMatch))
        print("PseudoMatch.Sie: ", len(PseudoMatch))
        print("Candidates: ", candidate)
        print("Precision: {}".format(pre))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("Wrong_Match: {}".format(wrong_match))
        print("Extra_Match: {}".format(extra_match))

        with open(filename, 'a+') as f:
            f.write("TrueMatch.Sie: {}\n".format(len(TrueMatch)))
            f.write("PseudoMatch.Sie: {}\n".format(len(PseudoMatch)))
            f.write("Pre: {}\n".format(pre))
            f.write("Rec: {}\n".format(recall))
            f.write("F1: {}\n".format(f1))
            f.write("Wrong_Match: {}\n".format(wrong_match))
            f.write("Extra_Match: {}\n".format(extra_match))

    conn.close()
    s.close()
    end = time.time()

    print("running time: ", end - begin)
    with open(filename, 'a+') as f:
        f.write("Running time: {}\n".format(end - begin))
