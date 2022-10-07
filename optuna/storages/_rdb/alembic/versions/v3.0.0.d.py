"""Handle inf/-inf for trial_values table.

Revision ID: v3.0.0.d
Revises: v3.0.0.c
Create Date: 2022-06-02 09:57:22.818798

"""
import enum

import numpy as np
from alembic import op
import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import orm
from typing import Optional
from typing import Tuple


# revision identifiers, used by Alembic.
revision = "v3.0.0.d"
down_revision = "v3.0.0.c"
branch_labels = None
depends_on = None


BaseModel = declarative_base()
RDB_MAX_FLOAT = np.finfo(np.float32).max
RDB_MIN_FLOAT = np.finfo(np.float32).min


FLOAT_PRECISION = 53


class TrialValueModel(BaseModel):
    class TrialValueType(enum.Enum):
        FINITE = 1
        INF_POS = 2
        INF_NEG = 3

    __tablename__ = "trial_values"
    trial_value_id = sa.Column(sa.Integer, primary_key=True)
    value = sa.Column(sa.Float(precision=FLOAT_PRECISION), nullable=True)
    value_type = sa.Column(sa.Enum(TrialValueType), nullable=False)

    @classmethod
    def value_to_stored_repr(
        cls,
        value: float,
    ) -> Tuple[Optional[float], TrialValueType]:
        if value == float("inf"):
            return (None, cls.TrialValueType.INF_POS)
        elif value == float("-inf"):
            return (None, cls.TrialValueType.INF_NEG)
        else:
            return (value, cls.TrialValueType.FINITE)

    @classmethod
    def stored_repr_to_value(cls, value: Optional[float], float_type: TrialValueType) -> float:
        if float_type == cls.TrialValueType.INF_POS:
            assert value is None
            return float("inf")
        elif float_type == cls.TrialValueType.INF_NEG:
            assert value is None
            return float("-inf")
        else:
            assert float_type == cls.TrialValueType.FINITE
            assert value is not None
            return value


def upgrade():
    bind = op.get_bind()

    sa.Enum(TrialValueModel.TrialValueType).create(bind, checkfirst=True)

    # MySQL and PostgreSQL supports DEFAULT clause like 'ALTER TABLE <tbl_name>
    # ADD COLUMN <col_name> ... DEFAULT "FINITE"', but seemingly Alembic
    # does not support such a SQL statement. So first add a column with schema-level
    # default value setting, then remove it by `batch_op.alter_column()`.
    with op.batch_alter_table("trial_values") as batch_op:
        batch_op.add_column(
            sa.Column(
                "value_type",
                sa.Enum("FINITE", "INF_POS", "INF_NEG", name="trialvaluetype"),
                nullable=False,
                server_default="FINITE",
            ),
        )
    with op.batch_alter_table("trial_values") as batch_op:
        batch_op.alter_column(
            "value_type",
            existing_type=sa.Enum("FINITE", "INF_POS", "INF_NEG", name="trialvaluetype"),
            existing_nullable=False,
            server_default=None,
        )
        batch_op.alter_column(
            "value",
            existing_type=sa.Float(precision=FLOAT_PRECISION),
            nullable=True,
        )

    session = orm.Session(bind=bind)
    try:
        records = (
            session.query(TrialValueModel)
            .filter(
                sa.or_(
                    TrialValueModel.value > 1e16,
                    TrialValueModel.value < -1e16,
                )
            )
            .all()
        )
        mapping = []
        for r in records:
            value: float
            if np.isclose(r.value, RDB_MAX_FLOAT) or np.isposinf(r.value):
                value = float("inf")
            elif np.isclose(r.value, RDB_MIN_FLOAT) or np.isneginf(r.value):
                value = float("-inf")
            else:
                value = r.value

            (
                stored_value,
                float_type,
            ) = TrialValueModel.value_to_stored_repr(value)
            mapping.append(
                {
                    "trial_value_id": r.trial_value_id,
                    "value_type": float_type,
                    "value": stored_value,
                }
            )
        session.bulk_update_mappings(TrialValueModel, mapping)
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
        records = session.query(TrialValueModel).all()
        mapping = []
        for r in records:
            if r.value_type == TrialValueModel.TrialValueType.FINITE:
                continue

            _value = r.value
            if r.value_type == TrialValueModel.TrialValueType.INF_POS:
                _value = RDB_MAX_FLOAT
            else:
                _value = RDB_MIN_FLOAT

            mapping.append(
                {
                    "trial_value_id": r.trial_value_id,
                    "value": _value,
                }
            )
        session.bulk_update_mappings(TrialValueModel, mapping)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()

    with op.batch_alter_table("trial_values", schema=None) as batch_op:
        batch_op.drop_column("value_type")
        batch_op.alter_column(
            "value",
            existing_type=sa.Float(precision=FLOAT_PRECISION),
            nullable=False,
        )

    sa.Enum(TrialValueModel.TrialValueType).drop(bind, checkfirst=True)
