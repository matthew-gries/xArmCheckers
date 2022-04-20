from typing import Tuple, List
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import os

from xarm_checkers.alphago_zero.NeuralNet import NeuralNet
from xarm_checkers.alphago_zero.utils import dotdict, AverageMeter
from xarm_checkers.learning.checkers import CheckersGame, CheckersGameState
from xarm_checkers.learning.checkers_nn_arch import CheckersNN
from xarm_checkers.checkers.utils import game_as_numpy

args = dotdict({
    'lr': 0.001,
    'epoch': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 5,
    'residual_block_count': 5
})

class CheckersNNWrapper(NeuralNet):

    def __init__(self, game: CheckersGame):
        self.game = game
        self.device = 'cuda' if args.cuda else 'cpu'
        self.network = CheckersNN(args)

        if args.cuda:
            self.network = self.network.cuda()

    def canonical_board_into_nn_rep(self, board: CheckersGameState) -> torch.Tensor:
        """
        Convert the canonical representation of the board into a game state representation that the neural network can use.
        Uses an approach similar to the game of Go, but each positional layer is doubled such that there is a positional layer
        for normal pieces and a positional layer for kinged pieces

        The top most layers will represent the positions of player 1's pieces
        """
        # 5 channels, 2 types of pieces x 2 players + layer to determine which player
        nn_rep = np.zeros((args.num_channels, 8, 8))
        # set current player in the last layer, if player 1 set these to all ones
        if board.whose_turn() == 1:
            nn_rep[1] = np.ones((8, 8))
        
        # fill in the players
        # loop is a leftover from when there were game histories
        for i, board in enumerate([board]):
            normal_idx_p1 = 2*i
            kings_idx_p1 = 2*i + 1
            board_as_np = game_as_numpy(board, as_board=True)
            normal_pieces_p1 = np.where(board_as_np==1, 1, 0).astype(np.float64)
            king_pieces_p1 = np.where(board_as_np==2, 1, 0).astype(np.float64)
            nn_rep[normal_idx_p1] = normal_pieces_p1
            nn_rep[kings_idx_p1] = king_pieces_p1
            normal_idx_p2 = 2*i + ((args.num_channels - 1) // 2)
            kings_idx_p2 = 2*i + ((args.num_channels - 1) // 2) + 1
            board_as_np = game_as_numpy(board, as_board=True)
            normal_pieces_p2 = np.where(board_as_np==-1, 1, 0).astype(np.float64)
            king_pieces_p2 = np.where(board_as_np==-2, 1, 0).astype(np.float64)
            nn_rep[normal_idx_p2] = normal_pieces_p2
            nn_rep[kings_idx_p2] = king_pieces_p2
    
        # convert to torch tensor and return
        return torch.from_numpy(nn_rep).to(device=self.device, dtype=torch.float64) 

    # Taken from othello example
    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    # Taken from othello example
    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]        

    def train(self, examples: List[Tuple[CheckersGameState, np.ndarray, float]]):
        # Convert boards into neural network representations
        optimizer = optim.Adam(self.network.parameters())

        for epoch in range(args.epoch):
            print('EPOCH ::: ' + str(epoch + 1))
            self.network.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')

            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards_list = [self.canonical_board_into_nn_rep(board) for board in boards]
                boards = torch.zeros((len(boards_list), args.num_channels, 8, 8)).to(self.device, dtype=torch.float64)
                for i, board in enumerate(boards_list):
                    boards[i] = board
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.network(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board: CheckersGameState) -> Tuple[np.ndarray, np.ndarray]:
        # preparing input
        board_rep = self.canonical_board_into_nn_rep(board)
        board_rep = board_rep.unsqueeze(dim=0).float()
        if args.cuda:
            board_rep = board_rep.contiguous().cuda()
        self.network.eval()
        with torch.no_grad():
            pi, v = self.network(board_rep)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.network.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.network.load_state_dict(checkpoint['state_dict'])
