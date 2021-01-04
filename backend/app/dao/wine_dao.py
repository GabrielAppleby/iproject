from app.dao.database import db


class WineDB(db.Model):
    __tablename__ = 'wine'
    uid = db.Column(db.Integer, primary_key=True)
    features = db.Column(db.LargeBinary())
    target = db.Column(db.FLOAT)

    def __repr__(self):
        return '<Wine {}>'.format(self.uid)
