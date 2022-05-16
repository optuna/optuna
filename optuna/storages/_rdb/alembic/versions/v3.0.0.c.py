"""Add intermediate_value_type column to represent +inf and -inf

Revision ID: v3.0.0.c
Revises: v3.0.0.b
Create Date: 2022-05-16 17:17:28.810792

"""
import enum

import numpy as np
from alembic import op
import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import orm


# revision identifiers, used by Alembic.
revision = "v3.0.0.c"
down_revision = "v3.0.0.b"
branch_labels = None
depends_on = None


BaseModel = declarative_base()
RDB_MAX_FLOAT = np.finfo(np.float32).max
RDB_MIN_FLOAT = np.finfo(np.float32).min


def _isinf(value: float) -> bool:
    return (
        np.isclose(value, RDB_MIN_FLOAT)
        or np.isclose(value, RDB_MIN_FLOAT)
        or np.isinf(value)  # for users who store inf/-inf at v2.10.0 or older.
    )


class IntermediateValueModel(BaseModel):
    class FloatTypeEnum(enum.Enum):
        FINITE_OR_NAN = 1
        INF_POS = 2
        INF_NEG = 3

    __tablename__ = "trial_intermediate_values"
    trial_intermediate_value_id = sa.Column(sa.Integer, primary_key=True)
    intermediate_value = sa.Column(sa.Float, nullable=True)
    intermediate_value_type = sa.Column(
        sa.Enum(FloatTypeEnum), nullable=False, default=FloatTypeEnum.FINITE_OR_NAN
    )


def upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)
    with op.batch_alter_table("trial_intermediate_values", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "intermediate_value_type",
                sa.Enum("FINITE_OR_NAN", "INF_POS", "INF_NEG", name="floattypeenum"),
                nullable=False,
                default="FINITE_OR_NAN",
            )
        )

    try:
        records = session.query(IntermediateValueModel).all()
        mapping = []
        for r in records:
            float_type: IntermediateValueModel.FloatTypeEnum
            if _isinf(r.intermediate_value):
                if r.intermediate_value > 0:
                    float_type = IntermediateValueModel.FloatTypeEnum.INF_POS
                else:
                    float_type = IntermediateValueModel.FloatTypeEnum.INF_NEG
            else:
                continue
            mapping.append(
                {
                    "trial_intermediate_value_id": r.trial_intermediate_value_id,
                    "float_type": float_type,
                }
            )
        session.bulk_update_mappings(IntermediateValueModel, mapping)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()


def downgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    try:
        records = session.query(IntermediateValueModel).all()
        mapping = []
        for r in records:
            if r.intermediate_value_type == IntermediateValueModel.FloatTypeEnum.FINITE_OR_NAN:
                continue

            _intermediate_value = r.intermediate_value
            if r.intermediate_value_type == IntermediateValueModel.FloatTypeEnum.INF_POS:
                _intermediate_value = RDB_MAX_FLOAT
            else:
                _intermediate_value = RDB_MIN_FLOAT

            mapping.append(
                {
                    "trial_intermediate_value_id": r.trial_intermediate_value_id,
                    "intermediate_value": _intermediate_value,
                }
            )
        session.bulk_update_mappings(IntermediateValueModel, mapping)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()

    with op.batch_alter_table("trial_intermediate_values", schema=None) as batch_op:
        batch_op.drop_column("intermediate_value_type")
