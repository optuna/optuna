"""Change floating point precision and make intermediate_value nullable.

Revision ID: v3.0.0.b
Revises: v3.0.0.a
Create Date: 2022-04-27 16:31:42.012666

"""
from alembic import op
from sqlalchemy import Float


# revision identifiers, used by Alembic.
revision = "v3.0.0.b"
down_revision = "v3.0.0.a"
branch_labels = None
depends_on = None

FLOAT_PRECISION = 53


def upgrade():
    with op.batch_alter_table("trial_intermediate_values") as batch_op:
        batch_op.alter_column(
            "intermediate_value",
            type_=Float(precision=FLOAT_PRECISION),
            nullable=True,
        )
    with op.batch_alter_table("trial_params") as batch_op:
        batch_op.alter_column(
            "param_value",
            type_=Float(precision=FLOAT_PRECISION),
            existing_nullable=True,
        )
    with op.batch_alter_table("trial_values") as batch_op:
        batch_op.alter_column(
            "value",
            type_=Float(precision=FLOAT_PRECISION),
            existing_nullable=False,
        )


def downgrade():
    with op.batch_alter_table("trial_intermediate_values") as batch_op:
        batch_op.alter_column("intermediate_value", type_=Float, nullable=False)
    with op.batch_alter_table("trial_params") as batch_op:
        batch_op.alter_column("param_value", type_=Float, existing_nullable=True)
    with op.batch_alter_table("trial_values") as batch_op:
        batch_op.alter_column("value", type_=Float, existing_nullable=False)
