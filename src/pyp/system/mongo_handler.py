import getpass
import logging
import sys
from datetime import datetime
from socket import gethostname

from bson import InvalidDocument
from pymongo.collection import Collection

try:
    from pymongo import MongoClient as Connection
except ImportError:
    from pymongo import Connection

if sys.version_info[0] >= 3:
    unicode = str


class MongoFormatter(logging.Formatter):
    def format(self, record):
        """Format exception object as a string"""
        data = record.__dict__.copy()

        if record.args:
            msg = record.msg % record.args
        else:
            msg = record.msg

        data.update(
            username=getpass.getuser(),
            time=datetime.now(),
            host=gethostname(),
            message=msg,
            args=tuple(unicode(arg) for arg in record.args),
        )
        if "exc_info" in data and data["exc_info"]:
            data["exc_info"] = self.formatException(data["exc_info"])
        return data


class MongoHandler(logging.Handler):
    """ Custom log handler

    Logs all messages to a mongo collection. This  handler is
    designed to be used with the standard python logging mechanism.
    """

    @classmethod
    def to(
        cls,
        collection,
        db="mongolog",
        host="localhost",
        port=None,
        username=None,
        password=None,
        level=logging.NOTSET,
    ):
        """ Create a handler for a given  """
        return cls(collection, db, host, port, username, password, level)

    def __init__(
        self,
        collection,
        db="mongolog",
        host="localhost",
        port=None,
        username=None,
        password=None,
        level=logging.NOTSET,
    ):
        """ Init log handler and store the collection handle """
        logging.Handler.__init__(self, level)
        if isinstance(collection, str):
            connection = Connection(host, port)
            if username and password:
                connection[db].authenticate(username, password)
            self.collection = connection[db][collection]
        elif isinstance(collection, Collection):
            self.collection = collection
        else:
            raise TypeError(
                "collection must be an instance of basestring or " "Collection"
            )
        self.formatter = MongoFormatter()

    def emit(self, record):
        """ Store the record to the collection. Async insert """
        try:
            self.collection.insert_one(self.format(record))
        except InvalidDocument as e:
            logging.error("Unable to save log record: %s", e.message, exc_info=True)
