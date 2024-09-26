import os
import mindspore as ms
from mindspore import Tensor, context, ops, nn
from mindspore.communication import init, get_rank, get_group_size
import logging
import socket

logger = logging.getLogger(__name__)

def setup_for_distributed(is_master):
    import warnings

    builtin_warn = warnings.warn

    def warn(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_warn(*args, **kwargs)

    # Log warnings only once
    warnings.warn = warn
    warnings.simplefilter("once", UserWarning)

    if not is_master:
        logging.disable()

def is_main_process():
    return get_rank() == 0

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def init_distributed_mode(args):
    if 'MS_ROLE' in os.environ and os.environ['MS_ROLE'] == 'MS_WORKER':
        # Job started by msrun
        args.rank = int(os.environ['RANK_ID']) if 'RANK_ID' in os.environ else 0
        args.world_size = int(os.environ['RANK_SIZE'])
    else:
        logger.info('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    context.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    init()

    logger.info('| distributed init (rank {}): {}'.format(args.rank, args.dist_url))

    setup_for_distributed(args.rank == 0)
    # TODO: barrier barrier()

# FIXME
def barrier():
    """
    A barrier that blocks each process until all processes reach this point.
    """
    sync_tensor = ms.Tensor([1], dtype=ms.float32)
    ops.AllReduce(ops.ReduceOp.SUM)(sync_tensor)