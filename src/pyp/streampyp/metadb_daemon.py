import time
from collections import namedtuple

from pymongo import MongoClient


def open(**kwargs):
    return MetaDB(**kwargs)


class MetaDB:
    def __init__(self, timeout_ms=30000):

        # connect to the database
        hostname = "login.cryoem.duke.edu"
        hostname = "10.183.5.180"
        port = 27017
        self._client = MongoClient(
            "mongodb://%s:%d" % (hostname, port),
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
            socketTimeoutMS=timeout_ms,
        )
        self._db = self._client.micromon

        # get our collections
        self._sessions = self.Sessions(self._db)
        self._micrographs = self.Micrographs(self._db)
        self._twod_classes = self.TwoDClassesCollection(self._db)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        # close the database connection
        self._client.close()
        self._client = None
        pass

    class Sessions:
        def __init__(self, db):
            self._collection = db.sessions

        def _filter(self, group_id, session_id):
            return {"_id": "%s/%s" % (group_id, session_id)}

        def get(self, group_id, session_id):
            with self._collection.find(self._filter(group_id, session_id)) as cursor:
                try:
                    return cursor.next()
                except StopIteration:
                    return None

        def write(self, group_id, session_id, doc):
            self._collection.replace_one(
                filter=self._filter(group_id, session_id), replacement=doc, upsert=True
            )

        def update(self, group_id, session_id, update_doc):
            self._collection.update_one(
                filter=self._filter(group_id, session_id), update=update_doc
            )

    class Micrographs:
        def __init__(self, db):
            self._collection = db.micrographs

        def _filter(self, group_id, session_id, micrograph_id):
            return {"_id": "%s/%s/%s" % (group_id, session_id, micrograph_id)}

        def _filter_session(self, group_id, session_id):
            return {"$and": [{"groupId": group_id}, {"sessionId": session_id}]}

        def delete_all(self, group_id, session_id):
            self._collection.delete_many(
                filter=self._filter_session(group_id, session_id)
            )

        def count(self, group_id, session_id):
            return self._collection.count_documents(
                self._filter_session(group_id, session_id)
            )

        def get(self, group_id, session_id, micrograph_id):
            with self._collection.find(
                self._filter(group_id, session_id, micrograph_id)
            ) as cursor:
                try:
                    return cursor.next()
                except StopIteration:
                    return None

        def write(self, group_id, session_id, micrograph_id, doc):
            self._collection.replace_one(
                filter=self._filter(group_id, session_id, micrograph_id),
                replacement=doc,
                upsert=True,
            )

        def update(self, group_id, session_id, micrograph_id, update_doc):
            self._collection.update_one(
                filter=self._filter(group_id, session_id, micrograph_id),
                update=update_doc,
                upsert=True,
            )

    class TwoDClassesCollection:
        def __init__(self, db):
            self._collection = db.twoDClasses

        def _filter(self, group_id, session_id, twodclasses_id):
            return {"_id": "%s/%s/%s" % (group_id, session_id, twodclasses_id)}

        def _filter_session(self, group_id, session_id):
            return {"$and": [{"groupId": group_id}, {"sessionId": session_id}]}

        def delete_all(self, group_id, session_id):
            self._collection.delete_many(
                filter=self._filter_session(group_id, session_id)
            )

        def count(self, group_id, session_id):
            return self._collection.count_documents(
                self._filter_session(group_id, session_id)
            )

        def get(self, group_id, session_id, twodclasses_id):
            with self._collection.find(
                self._filter(group_id, session_id, twodclasses_id)
            ) as cursor:
                try:
                    return cursor.next()
                except StopIteration:
                    return None

        def write(self, group_id, session_id, twodclasses_id, doc):
            self._collection.replace_one(
                filter=self._filter(group_id, session_id, twodclasses_id),
                replacement=doc,
                upsert=True,
            )

        def update(self, group_id, session_id, twodclasses_id, update_doc):
            self._collection.update_one(
                filter=self._filter(group_id, session_id, twodclasses_id),
                update=update_doc,
                upsert=True,
            )

    def session(self, group_id, session_id):
        return MetaDB.Session(self, group_id, session_id)

    class Session:
        def __init__(self, db, group_id, session_id):
            self._db = db
            self.group_id = group_id
            self.session_id = session_id

        def _get(self):
            return self._db._sessions.get(self.group_id, self.session_id)

        def _write(self, doc):
            return self._db._sessions.write(self.group_id, self.session_id, doc)

        def _update(self, update_doc):
            return self._db._sessions.update(self.group_id, self.session_id, update_doc)

        def exists(self):
            """
			Return True iff the session has already been created.
			"""
            return self._get() is not None

        def create(self, created):
            """
			Create a new session, or re-create an existing session.

			:param int created: Creation time of the session, in milliseconds since the Unix epoch
			"""
            self._write(
                {
                    # static metadata
                    "_id": "%s/%s" % (self.group_id, self.session_id),
                    "group": self.group_id,
                    "name": self.session_id,
                    "created": created,
                    # defaults for dynamic metadata
                    "updated": None,
                    "micrographs": [],
                    "twoDClasses": [],
                    # TEMP: set a scanned date, so the website scanner won't try to scan this session
                    "scanned": 5,
                }
            )

            # clear the micrographs and 2D classes for this session
            self._db._micrographs.delete_all(self.group_id, self.session_id)
            self._db._twod_classes.delete_all(self.group_id, self.session_id)

        def count_micrographs(self):
            """
			Count the number of micrographs that have been created,
			not the number of micrograph IDs saved to the session.

			:rtype: int
			"""
            return self._db._micrographs.count(self.group_id, self.session_id)

        def get_pyp_config(self):
            """
			:rtype: list[str]
			"""
            return self._get()["pyp"]

        def set_pyp_config(self, config):
            """
			:type config: dict[str,str]
			"""
            self._update({"$set": {"pyp": config}})

        def get_micrograph_ids(self):
            """
			:rtype: list[str]
			"""
            return self._get()["micrographs"]

        def set_micrograph_ids(self, ids):
            """
			:type ids: list[str]
			"""
            self._update({"$set": {"micrographs": ids}})

        def append_micrograph_id(self, id):
            """
			:type id: str
			"""
            self._update({"$push": {"micrographs": id}})

        def get_updated(self):
            """
			Gets the 'updated' timestamp for the session, in milliseconds since the Unix epoch
			:rtype: int
			"""
            return self._get()["updated"]

        def set_updated_now(self):
            """
			Sets the 'updated' timestamp for the session.
			"""
            timestamp = int(round(time.time() * 1000))
            self._update({"$set": {"updated": timestamp}})

        def get_twodclasses_ids(self):
            """
			:rtype: list[str]
			"""
            return self._get()["twoDClasses"]

        def set_twodclasses_ids(self, ids):
            """
			:type ids: list[str]
			"""
            self._update({"$set": {"twoDClasses": ids}})

        def append_twodclasses_id(self, id):
            """
			:type id: str
			"""
            self._update({"$push": {"twoDClasses": id}})

        def micrograph(self, id):
            return MetaDB.Micrograph(self._db, self, id)

        def twodclasses(self, id):
            return MetaDB.TwoDClasses(self._db, self, id)

    class Micrograph:
        def __init__(self, db, session, id):
            self._db = db
            self.session = session
            self.id = id

        def _get(self):
            return self._db._micrographs.get(
                self.session.group_id, self.session.session_id, self.id
            )

        def _write(self, doc):
            return self._db._micrographs.write(
                self.session.group_id, self.session.session_id, self.id, doc
            )

        def _update(self, update_doc):
            return self._db._micrographs.update(
                self.session.group_id, self.session.session_id, self.id, update_doc
            )

        def exists(self):
            """
			Return True iff the micrograph has already been created.
			"""
            return self._get() is not None

        def create(self, created):
            """
			Create a new micrograph, or re-create an existing micrograph.

			:param int created: Creation time of the micrograph, in milliseconds since the Unix epoch
			"""
            self._write(
                {
                    "_id": "%s/%s/%s"
                    % (self.session.group_id, self.session.session_id, self.id),
                    "groupId": self.session.group_id,
                    "sessionId": self.session.session_id,
                    "timestamp": created,
                }
            )

        def get_ctf(self):
            """
			:rtype: CTF
			"""
            return CTF(**self._get()["ctf"])

        def set_ctf(self, ctf):
            """
			:type ctf: CTF
			"""
            self._update({"$set": {"ctf": ctf._asdict()}})

        def get_avgrot(self):
            """
			:rtype: list[AVGROT]
			"""
            return [AVGROT(**x) for x in self._get()["avgrot"]["samples"]]

        def set_avgrot(self, avgrot):
            """
			:type avgrot: list[AVGROT]
			"""
            self._update(
                {"$set": {"avgrot": {"samples": [x._asdict() for x in avgrot]}}}
            )

        def get_xf(self):
            """
			:rtype: list[XF]
			"""
            return [XF(**x) for x in self._get()["xf"]["samples"]]

        def set_xf(self, xf):
            """
			:type xf: list[XF]
			"""
            self._update({"$set": {"xf": {"samples": [x._asdict() for x in xf]}}})

        def get_boxx(self):
            """
			:rtype: list[BOXX]
			"""
            return [BOXX(**x) for x in self._get()["boxx"]["samples"]]

        def set_boxx(self, boxx):
            """
			:type boxx: list[BOXX]
			"""
            self._update({"$set": {"boxx": {"particles": [x._asdict() for x in boxx]}}})

    class TwoDClasses:
        def __init__(self, db, session, id):
            self._db = db
            self.session = session
            self.id = id

        def _get(self):
            return self._db._twod_classes.get(
                self.session.group_id, self.session.session_id, self.id
            )

        def _write(self, doc):
            return self._db._twod_classes.write(
                self.session.group_id, self.session.session_id, self.id, doc
            )

        def _update(self, update_doc):
            return self._db._twod_classes.update(
                self.session.group_id, self.session.session_id, self.id, update_doc
            )

        def exists(self):
            """
			Return True iff the 2D Classes has not already been created.
			"""
            return self._get() is not None

        def create(self, created, width, height):
            """
			Create a new 2D classes image, or re-create an existing 2D classes image.

			:param int created: Creation time of the 2D classes image, in milliseconds since the Unix epoch
			:param int width: Width of the image, in pixels
			:param int height: Height of the image, in pixels
			"""
            self._write(
                {
                    "_id": "%s/%s/%s"
                    % (self.session.group_id, self.session.session_id, self.id),
                    "groupId": self.session.group_id,
                    "sessionId": self.session.session_id,
                    "timestamp": created,
                    "w": width,
                    "h": height,
                }
            )


CTF = namedtuple(
    "CTF",
    [
        "mean_df",
        "cc",
        "df1",
        "df2",
        "angast",
        "ccc",
        "x",
        "y",
        "z",
        "pixel_size",
        "voltage",
        "magnification",
        "cccc",
        "counts",
    ],
)

AVGROT = namedtuple(
    "AVGROT", ["freq", "avgrot_noastig", "avgrot", "ctf_fit", "quality_fit", "noise"]
)

XF = namedtuple("XF", ["mat00", "mat01", "mat10", "mat11", "x", "y"])

BOXX = namedtuple("BOXX", ["x", "y", "w", "h", "in_bounds", "cls"])
