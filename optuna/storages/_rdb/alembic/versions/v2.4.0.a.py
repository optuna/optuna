"""empty message

Revision ID: v2.4.0.a
Revises: v1.3.0.a
Create Date: 2020-11-17 02:16:16.536171

"""
from alembic import op
import sqlalchemy as sa

from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import orm

from optuna.storages._rdb.models import StudyModel
from optuna.storages._rdb.models import StudyDirectionModel
from optuna.storages._rdb.models import TrialIntermediateValueModel
from optuna.storages._rdb.models import TrialModel
from optuna.storages._rdb.models import TrialValueModel
from optuna.study import StudyDirection


# revision identifiers, used by Alembic.
revision = "v2.4.0.a"
down_revision = "v1.3.0.a"
branch_labels = None
depends_on = None

# Model definition
MAX_INDEXED_STRING_LENGTH = 512
MAX_STRING_LENGTH = 2048
BaseModel = declarative_base()


def upgrade():
    bind = op.get_bind()
    tables = Inspector.from_engine(bind).get_table_names()

    if "study_direction" not in tables:
        op.create_table(
            "study_direction",
            sa.Column("study_direction_id", sa.Integer(), nullable=False),
            sa.Column(
                "direction",
                sa.Enum("NOT_SET", "MINIMIZE", "MAXIMIZE", name="studydirection"),
                nullable=False,
            ),
            sa.Column("study_id", sa.Integer(), nullable=True),
            sa.Column("objective_id", sa.Integer(), nullable=True),
            sa.ForeignKeyConstraint(
                ["study_id"],
                ["studies.study_id"],
            ),
            sa.PrimaryKeyConstraint("study_direction_id"),
            sa.UniqueConstraint("study_id", "objective_id"),
        )

    if "trial_intermediate_values" not in tables:
        op.create_table(
            "trial_intermediate_values",
            sa.Column("trial_intermediate_values_id", sa.Integer(), nullable=False),
            sa.Column("trial_id", sa.Integer(), nullable=True),
            sa.Column("step", sa.Integer(), nullable=True),
            sa.Column("value", sa.Float(), nullable=True),
            sa.ForeignKeyConstraint(
                ["trial_id"],
                ["trials.trial_id"],
            ),
            sa.PrimaryKeyConstraint("trial_intermediate_values_id"),
            sa.UniqueConstraint("trial_id", "step"),
        )

    with op.batch_alter_table("trial_values", schema=None) as batch_op:
        batch_op.add_column(sa.Column("objective_id", sa.Integer(), nullable=True))
        batch_op.create_unique_constraint("value_constraint", ["trial_id", "objective_id"])

    session = orm.Session(bind=bind)
    try:

        class BackwardCompatibleStudyModel(StudyModel):
            direction = sa.Column(sa.Enum(StudyDirection), nullable=False)

        class BackwardCompatibleTrialValueModel(TrialValueModel):
            step = sa.Column(sa.Integer)

        class BackwardCompatibleTrialModel(TrialModel):
            value = sa.Column(sa.Float)

        studies_records = session.query(BackwardCompatibleStudyModel).all()
        objects = [
            StudyDirectionModel(study_id=r.study_id, direction=r.direction, objective_id=0)
            for r in studies_records
        ]
        session.bulk_save_objects(objects)

        intermediate_values_records = session.query(BackwardCompatibleTrialValueModel).all()
        objects = [
            TrialIntermediateValueModel(trial_id=r.trial_id, value=r.value, step=r.step)
            for r in intermediate_values_records
        ]
        session.bulk_save_objects(objects)

        TrialValueModel.__table__.delete()
        trials_records = session.query(BackwardCompatibleTrialModel).all()
        objects = [
            TrialValueModel(trial_id=r.trial_id, value=r.value, objective_id=0)
            for r in trials_records
        ]
        session.bulk_save_objects(objects)

        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()

    with op.batch_alter_table("studies", schema=None) as batch_op:
        batch_op.drop_column("direction")

    with op.batch_alter_table("trial_values", schema=None) as batch_op:
        batch_op.drop_column("step")

    with op.batch_alter_table("trials", schema=None) as batch_op:
        batch_op.drop_column("value")


def downgrade():
    with op.batch_alter_table("trials", schema=None) as batch_op:
        batch_op.add_column(sa.Column("value", sa.FLOAT(), nullable=True))

    with op.batch_alter_table("trial_values", schema=None) as batch_op:
        batch_op.add_column(sa.Column("step", sa.INTEGER(), nullable=True))

    with op.batch_alter_table("studies", schema=None) as batch_op:
        batch_op.add_column(sa.Column("direction", sa.VARCHAR(length=8), nullable=False))

    bind = op.get_bind()
    session = orm.Session(bind=bind)

    try:
        values_records = session.query(TrialValueModel).all()
        mapping = [{"trial_id": r.trial_id, "value": r.value} for r in values_records]
        session.bulk_update_mappings(TrialModel, mapping)

        intermediate_values_records = session.query(TrialIntermediateValueModel).all()
        mapping = [
            {"trial_id": r.trial_id, "value": r.value, "step": r.step}
            for r in intermediate_values_records
        ]
        session.bulk_update_mappings(TrialValueModel, mapping)

        direction_records = session.query(StudyDirectionModel).all()
        mapping = [{"study_id": r.study_id, "direction": r.direction} for r in direction_records]
        session.bulk_update_mappings(StudyModel, mapping)

        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()

    with op.batch_alter_table("trial_values", schema=None) as batch_op:
        batch_op.drop_constraint("value_constraint", type_="unique")
        batch_op.drop_column("objective_id")

    op.drop_table("trial_intermediate_values")
    op.drop_table("study_direction")
