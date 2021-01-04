from app.dao.database import db


class FlowerDB(db.Model):
    __tablename__ = 'flowers'
    uid = db.Column(db.Integer, primary_key=True)
    features = db.Column(db.LargeBinary())
    target = db.Column(db.FLOAT)

    def __repr__(self):
        return '<Flower {}>'.format(self.uid)
