from datetime import datetime

from parser.utils.logging import init_logger, logger


class CMD(object):
    def __call__(self, args):
        self.args = args
        timestamp = datetime.now().isoformat().split(".")[0]
        self.timestamp = str.replace(timestamp, ':', '_')
        init_logger(logger, f"{args.save}/{args.mode}_D{self.timestamp}.log")
