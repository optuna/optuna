"""empty message

Revision ID: v1.2.0.b
Revises: v1.2.0.a
Create Date: 2020-02-14 16:23:04.800808

"""
import json

from alembic import op
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.declarative import declarative_base

# revision identifiers, used by Alembic.
revision = 'v1.2.0.b'
down_revision = 'v1.2.0.a'
branch_labels = None
depends_on = None

# Model definition
MAX_INDEXED_STRING_LENGTH = 512
MAX_STRING_LENGTH = 2048
BaseModel = declarative_base()  # type: Any


class TrialModel(BaseModel):
    __tablename__ = 'trials'
    trial_id = sa.Column(sa.Integer, primary_key=True)
    number = sa.Column(sa.Integer, unique=True)


class TrialSystemAttributeModel(BaseModel):
    __tablename__ = 'trial_system_attributes'
    trial_system_attribute_id = sa.Column(sa.Integer, primary_key=True)
    trial_id = sa.Column(sa.Integer, sa.ForeignKey('trials.trial_id'))
    key = sa.Column(sa.String(MAX_INDEXED_STRING_LENGTH))
    value_json = sa.Column(sa.String(MAX_STRING_LENGTH))


def upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    with op.batch_alter_table("trials") as batch_op:
        batch_op.add_column(sa.Column('number', sa.Integer(), nullable=True, default=-1))

    try:
        number_records = session.query(TrialSystemAttributeModel) \
            .filter(TrialSystemAttributeModel.key == '_number').all()
        mapping = [{'trial_id': r.trial_id, 'number': json.loads(r.value_json)}
                   for r in number_records]
        session.bulk_update_mappings(TrialModel, mapping)
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def downgrade():
    with op.batch_alter_table("trials") as batch_op:
        batch_op.drop_column('number')
