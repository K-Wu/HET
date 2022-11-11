from .train_dgl import RGAT_main_procedure, RGAT_parse_args

if __name__ == "__main__":
    args = RGAT_parse_args()
    print(args)
    RGAT_main_procedure(args, dgl_model_flag=False)
