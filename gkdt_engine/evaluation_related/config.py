import os
import sys
import time
import datetime
import argparse
from yacs.config import CfgNode
import logging
from tensorboardX import SummaryWriter
import yaml

def parse_list_str_via_yaml(args, cfg):
    for per_source in ['TRAIN_DATA', 'VAL_DATA', 'TEST_DATA']:
        per_source_tmp = ('DATASET.'+per_source)  # e.g., 'DATASET.TEST_DATA'
        if per_source_tmp in args.opts:
            idx = args.opts.index(per_source_tmp)
            argument_str = args.opts[idx + 1]
            list_tmp = yaml.safe_load(argument_str)  # parse list of lists via json
            cfg['DATASET'][per_source] = list_tmp  # put list of lists to cfg
            del args.opts[idx:(idx+2)]  # delete
    return args, cfg

def update_config():
    parser = argparse.ArgumentParser(description='Open-prompted keypoint detection.')
    #./experiments/configs/openkd_autoname.yaml
    parser.add_argument('--autoname_keys', nargs='*', default=[])
    parser.add_argument('--autoname_labels', nargs='*', default=[])
    parser.add_argument('--cfg_file', type=str, default='./experiments/configs/openkd.yaml', help='config file')
    parser.add_argument('--eval_only', action="store_true", default=False)
    parser.add_argument('--input_file', default=None)
    parser.add_argument('--output_pred_file', default=None)
    parser.add_argument('opts', help='see yaml config files for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = CfgNode.load_cfg(open(args.cfg_file))
    
    #---------------------------
    # parse dataset configuration (list of lists) and put them into cfg
    args, cfg = parse_list_str_via_yaml(args, cfg)
    # add EVAL_ONLY to cfg
    cfg.EVAL_ONLY = args.eval_only  # False, training phase; True, test phase
    # add input_file to cfg
    cfg.input_file = args.input_file 
    # add output_pred_file to cfg
    cfg.output_pred_file = args.output_pred_file 
    #---------------------------
    
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    print('-------------autoname key-labels start---------------')
    print(args.autoname_keys)
    print(args.autoname_labels)
    assert len(args.autoname_keys) == len(args.autoname_labels), 'keys/labels number should be same.'
    cfg.AUTONAME.KEYS = args.autoname_keys
    cfg.AUTONAME.LABELS = args.autoname_labels
    print('-------------autoname key-labels end---------------')

    return cfg, args

def create_loggers(cfg, cfg_str, create_dir=True, init_logger=True, init_tb=True):
    output_dir = cfg.OUTPUT_DIR
    output_model_dir = os.path.join(output_dir, 'model')
    output_log_dir = os.path.join(output_dir, 'log')
    output_tb_log_dir = os.path.join(output_dir, 'tb_log')

    if create_dir == True:
        if os.path.exists(output_dir) == False:
            try:
                os.makedirs(output_dir)
            except:
                print('==>Cannot create folder at %s. Folder may already exist.'%(output_dir))
        for p in [output_model_dir, output_log_dir, output_tb_log_dir]:
            if os.path.exists(p) == False:
                try:
                    os.makedirs(p)
                except:
                    print('==>Cannot create folder at %s. Folder may already exist.'%(p))

    # set up logger
    if init_logger:
        date_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger_path = os.path.join(output_log_dir, cfg_str+'_{}.log'.format(date_str))
        # head = '%(asctime)-15s %(message)s'  # print format: datetime + messages
        head = '%(message)s'  # print format: messages only
        logging.basicConfig(
            filename=str(logger_path),
            level=logging.INFO,
            format=head,
            # filemode='a',
            )
        logger = logging.getLogger()
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        #-------------------------------------------
        # Simple class to redirect stdout to logging
        class PrintToLogger:
            def __init__(self, level=logging.INFO):
                self.level = level
                
            def write(self, message):
                if message.strip():  # Skip empty messages
                    logging.log(self.level, message.strip())
                    
            def flush(self):
                pass  # Required for file-like object
        # Redirect stdout to use our logging system
        sys.stdout = PrintToLogger(logging.INFO)
        sys.stderr = PrintToLogger(logging.ERROR)
        #-------------------------------------------
    else:
        logger = None

    # set up tensorboard writer
    if init_tb:
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        tb_writer_path = os.path.join(output_tb_log_dir, '{}-{}'.format(time_str, cfg_str))
        tb_writer = SummaryWriter(tb_writer_path)
    else:
        tb_writer = None

    return output_model_dir, logger, tb_writer