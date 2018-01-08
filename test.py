import _pickle as cPickle
from model import MemNet
from solver import Solver


#=============================
dec_path = './dec_map.pkl'
emb_path = './emb_map.pkl'
#=============================

def main():

    print('Restoring map...')
    dec_map = cPickle.load(open(dec_path, 'rb'))

    emb_map = cPickle.load(open(emb_path, 'rb'))

    input_size = len(emb_map[0])

    print('Bulid Model...')
    model = MemNet(emb_map,
                    input_size = input_size,
                    embed_size = 300,
                    n_hop = 6,
                    memory_size = 20,
                    sentence_size = 50,
                    option_size = 10,
                    decode_map = dec_map)

    print('Bulid Solver...')
    solver = Solver(model,
                    eval_batch_size = 128,
                    test_record_path = './record/test/',
                    test_examples = 10000,
                    restore_path = './checkpoint/',
                    print_step = 10)

    solver.test()

if __name__ == '__main__':
    main()
