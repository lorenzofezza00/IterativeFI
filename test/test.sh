#!/usr/bin/env bash

# parser.add_argument("-data", choices=available_datasets, required=True, help=f"{','.join(available_datasets)}")
# parser.add_argument("-root", type=str, help="root datapath", default="../../data/")
# parser.add_argument("-net", type=str, help="network", default=None)
# parser.add_argument("-weights_path", help="weights_path", default="./networks/weights")
# parser.add_argument("-results_path", help="results_path", default="./results/")
# # parser.add_argument("-preds_path", help="preds_path", default="./preds/")

# parser.add_argument("-pilot", type=int, help="pilot", default = 200)
# parser.add_argument("-eps", type=float, help="epsilon", default = 0.005)
# parser.add_argument("-conf", type=float, help="confidence", default = 0.95)
# parser.add_argument("-p0", type=float, help="epsilon", default = 0.5)
# parser.add_argument("-block", type=int, help="pilot", default=50)
# parser.add_argument("-budget_cap", type=int, help="budget_cap", default=None)
# parser.add_argument("-seed", type=int, help="seed", default=0)

python main.py -data banknote -root ../data -net banknote_mlp -weights_path ./networks/weights/best_banknote_mlp.pt -results_path ./results