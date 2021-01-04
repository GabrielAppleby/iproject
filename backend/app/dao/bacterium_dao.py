from app.dao.database import db


class BacteriumDB(db.Model):
    __tablename__ = 'bacteria'
    uid = db.Column(db.Integer, primary_key=True)
    features = db.Column(db.LargeBinary())
    target = db.Column(db.FLOAT)

    def __repr__(self):
        return '<Bacterium {}>'.format(self.uid)
